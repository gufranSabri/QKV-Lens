# Captures per-layer Q, K, V activations from a HuggingFace causal LM via
# forward hooks on q_proj/k_proj/v_proj -- HF exposes hidden_states and
# attentions for free but nothing for Q/K/V.
#
# Captured PRE-RoPE (the hook fires before RoPE is applied and before the
# head reshape): RoPE entangles content with absolute position, and we want
# content; V never receives RoPE so the distinction doesn't apply to it.
# The steering stage writes back at this exact point (output of the
# projection Linear), so detector-read and steerer-write coordinates match.
#
# GQA: K/V are narrower than Q in every model used here (e.g. Llama-3-8B
# D_q=4096, D_kv=1024). Read geometry from model.config, never hardcode it.
#
# With a KV cache, each attention module fires once at PREFILL (seq=prompt_len,
# the prompt's Q/K/V) and once per DECODE step (seq=1, the new token's). Only
# decode steps are recorded -- the (T, L, D) tensor should be one Q/K/V vector
# per layer per GENERATED token, never the prompt's.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn

VIEWS = ("Q", "K", "V")
_PROJ_FOR_VIEW = {"Q": "q_proj", "K": "k_proj", "V": "v_proj"}


@dataclass
class ModelGeometry:
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def d_q(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.head_dim

    def feature_dim(self, view: str) -> int:
        return self.d_q if view == "Q" else self.d_kv

    def __str__(self) -> str:
        return (
            f"L={self.n_layers} hidden={self.hidden_size} "
            f"heads={self.n_heads} kv_heads={self.n_kv_heads} "
            f"head_dim={self.head_dim} D_q={self.d_q} D_kv={self.d_kv}"
        )

    def valid_n_segments(self) -> list[int]:
        # Segment counts dividing every projection's feature dim (their gcd).
        # Qwen2.5-7B (L=28, D_kv=512) has no valid M == its layer count.
        from math import gcd

        g = 0
        for view in VIEWS:
            g = gcd(g, self.feature_dim(view))
        return [d for d in range(1, g + 1) if g % d == 0]

    def check_n_segments(self, n_segments: int) -> None:
        bad = [
            (v, self.feature_dim(v))
            for v in VIEWS
            if self.feature_dim(v) % n_segments != 0
        ]
        if not bad:
            return

        valid = self.valid_n_segments()
        nearby = sorted(valid, key=lambda d: abs(d - n_segments))[:6]
        detail = ", ".join(f"{v} has D={d}" for v, d in bad)
        raise ValueError(
            f"n_segments={n_segments} does not evenly divide the feature dim of: "
            f"{detail}. This model has L={self.n_layers} layers, D_q={self.d_q}, "
            f"D_kv={self.d_kv}. Set extract.n_segments to one of {sorted(nearby)} "
            f"in your config (closest valid choices to {n_segments})."
        )


def read_geometry(model) -> ModelGeometry:
    cfg = model.config
    cfg = getattr(cfg, "text_config", cfg)   # multimodal wrappers nest the text config

    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads   # no GQA -> kv == q heads

    return ModelGeometry(
        n_layers=cfg.num_hidden_layers,
        hidden_size=hidden,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
    )


def get_decoder_layers(model) -> nn.ModuleList:
    for path in (
        "model.layers",              # Llama, Mistral, Qwen, ...
        "model.model.layers",
        "transformer.h",             # GPT-2 family
        "model.decoder.layers",      # OPT
    ):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (nn.ModuleList, list)):
            return obj
    raise AttributeError(
        "could not locate decoder layers on this model; expected one of "
        "model.layers / model.model.layers / transformer.h / model.decoder.layers"
    )


def get_projection(model, layer_idx: int, view: str) -> nn.Module:
    # Shared by the read path (hooks below) and the future steering write
    # path, so both address the identical module.
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; valid views are {VIEWS}")
    layers = get_decoder_layers(model)
    return getattr(layers[layer_idx].self_attn, _PROJ_FOR_VIEW[view])


@dataclass
class QKVCapture:
    # Accumulates hook firings keyed by (view, layer), one growing list PER
    # BATCH SLOT so each sequence's own steps can be truncated to its own
    # true length (sequences finish decoding at different steps).
    n_layers: int
    views: tuple[str, ...]
    batch_size: int = 1
    steps: dict = field(default_factory=dict)
    _recording: bool = False
    # True while the slot's sequence is still decoding; set by the caller
    # before each step. A firing for a finished slot is not recorded.
    active: list = field(default_factory=list)

    def __post_init__(self):
        self.steps = {
            v: [[[] for _ in range(self.batch_size)] for _ in range(self.n_layers)]
            for v in self.views
        }
        self.active = [True] * self.batch_size

    def record(self, view: str, layer: int, out: torch.Tensor) -> None:
        # out: (B, seq, D); seq is always 1 here (one decode step per slot).
        if not self._recording:
            return
        if out.shape[1] != 1:
            raise RuntimeError(
                f"expected seq=1 per decode step, got seq={out.shape[1]}. "
                "This means a prefill-shaped tensor reached record() while "
                "recording was on -- the prefill/decode split is wrong."
            )
        # detach + move to CPU immediately -- GPU memory over a long generation
        cpu_out = out[:, 0].detach().to("cpu", torch.float32)  # (B, D)
        for slot in range(self.batch_size):
            if self.active[slot]:
                self.steps[view][layer][slot].append(cpu_out[slot])

    def stack(self) -> dict[str, dict[int, torch.Tensor]]:
        # {slot: (T_slot, L, D)} per view; T_slot varies across the batch
        # since slots stop accumulating once they finish.
        out: dict[str, dict[int, torch.Tensor]] = {}
        for view in self.views:
            per_slot: dict[int, torch.Tensor] = {}
            for slot in range(self.batch_size):
                per_layer = []
                for layer in range(self.n_layers):
                    chunks = self.steps[view][layer][slot]
                    if not chunks:
                        per_layer.append(torch.empty(0))   # zero tokens generated (immediate EOS)
                        continue
                    per_layer.append(torch.stack(chunks, dim=0))  # (T_slot, D)

                n_tok = {t.shape[0] for t in per_layer}
                if len(n_tok) != 1:
                    raise RuntimeError(
                        f"view {view} slot {slot}: layers disagree on token "
                        f"count: {sorted(n_tok)}. Some layers fired more often "
                        "than others for this sequence."
                    )
                if next(iter(n_tok)) == 0:
                    per_slot[slot] = torch.empty(0)
                else:
                    per_slot[slot] = torch.stack(per_layer, dim=1)  # (T_slot, L, D)
            out[view] = per_slot
        return out


@contextmanager
def qkv_hooks(model, batch_size: int = 1):
    # Attaches Q/K/V hooks for the context's duration; recording starts OFF so
    # a prefill pass can run unrecorded (capture_all toggles _recording).
    geom = read_geometry(model)
    layers = get_decoder_layers(model)
    if len(layers) != geom.n_layers:
        raise RuntimeError(
            f"config says {geom.n_layers} layers but found {len(layers)} modules"
        )

    capture = QKVCapture(n_layers=geom.n_layers, views=VIEWS, batch_size=batch_size)
    handles = []

    def make_hook(view: str, layer_idx: int):
        def hook(_module, _inputs, output):
            capture.record(view, layer_idx, output)
        return hook

    try:
        for layer_idx in range(geom.n_layers):
            for view in VIEWS:
                proj = get_projection(model, layer_idx, view)
                handles.append(proj.register_forward_hook(make_hook(view, layer_idx)))
        yield capture
    finally:
        for h in handles:
            h.remove()


def _resolve_eos_ids(eos_token_id, model) -> set[int]:
    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id
    if eos_token_id is None:
        return set()
    return set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])


def left_pad_batch(
    prompt_ids: list[torch.Tensor], pad_id: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Left-pad prompts to (B, max_len). Left (not right) padding keeps "the
    # last real token" at column -1 for every row, so one
    # logits[:, -1, :] gives the whole batch's next-token argmax at once.
    flat = [p.reshape(-1) for p in prompt_ids]
    max_len = max(p.shape[0] for p in flat)
    B = len(flat)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, p in enumerate(flat):
        n = p.shape[0]
        input_ids[i, max_len - n :] = p.to(device)
        attention_mask[i, max_len - n :] = 1
    return input_ids, attention_mask


@torch.no_grad()
def capture_all(
    model,
    input_ids,
    max_new_tokens: int = 64,
    eos_token_id=None,
    attention_mask: torch.Tensor | None = None,
    capture_generate_outputs: bool = False,
) -> list[tuple[dict[str, torch.Tensor], torch.Tensor]] | list[tuple[dict, torch.Tensor, dict]]:
    # Greedy-generates and captures per-layer Q/K/V for every generated token.
    # Drives the decode loop by hand (not model.generate) since generate()
    # gives no reliable hook for the prefill/decode boundary.
    #
    # input_ids: (B, prompt_len), left-padded; pass attention_mask from
    # left_pad_batch when B > 1.
    #
    # capture_generate_outputs: also captures hidden_states/attentions/logits
    # per decode step (for baselines other than QKV-Lens's own field, e.g.
    # hallushift's per-token features). Off by default since output_attentions
    # needs attn_implementation="eager" and a full (B, heads, seq, seq) tensor
    # per step. Requires B == 1 -- per-row slicing out of a padded batch isn't
    # implemented here; the caller forces batch_size=1 when this is set.
    #
    # Returns a list of B (activations, generated_ids) pairs, or with
    # capture_generate_outputs, (activations, generated_ids, generate_outputs)
    # triples where generate_outputs mirrors transformers'
    # GenerateDecoderOnlyOutput shape. activations["Q"] is (T_i, L, D_q),
    # ["K"]/["V"] are (T_i, L, D_kv); T_i varies per row (own EOS or
    # max_new_tokens), so results are not a padded tensor.
    if input_ids.ndim != 2:
        raise ValueError(f"expected input_ids (B, prompt_len), got {tuple(input_ids.shape)}")

    B = input_ids.shape[0]
    if capture_generate_outputs and B != 1:
        raise ValueError(
            f"capture_generate_outputs requires batch_size 1, got B={B} -- "
            "the caller must force batch_size to 1 when this is set."
        )
    device = input_ids.device
    eos_ids = _resolve_eos_ids(eos_token_id, model)

    extra_kwargs = (
        dict(output_hidden_states=True, output_attentions=True, output_logits=True)
        if capture_generate_outputs
        else {}
    )
    step_hidden_states: list[list] = [[] for _ in range(B)]
    step_attentions: list[list] = [[] for _ in range(B)]
    step_logits: list[list] = [[] for _ in range(B)]

    with qkv_hooks(model, batch_size=B) as capture:
        # ---- PREFILL: consume the prompt. NOT recorded. ----
        capture._recording = False
        out = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True, **extra_kwargs
        )
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

        if attention_mask is not None:
            running_mask = attention_mask
        else:
            running_mask = torch.ones_like(input_ids)

        # ---- DECODE: one token at a time. Recorded. ----
        # Token fed at step t is the one generated at step t-1 (or the
        # prompt's last-logit argmax for t=0), so activations at step t are
        # exactly what the model computed FOR that generated token.
        capture._recording = True
        generated: list[list[int]] = [[] for _ in range(B)]
        finished = [False] * B

        for _ in range(max_new_tokens):
            newly_finished = [
                (not finished[i]) and int(next_token[i, 0].item()) in eos_ids
                for i in range(B)
            ]
            for i, done in enumerate(newly_finished):
                if done:
                    finished[i] = True
            # A slot finished BEFORE this step must not be recorded on this
            # firing; a slot finishing ON this step still gets a real forward
            # pass but its token is EOS, so it's excluded from `generated` below.
            capture.active = [not f for f in finished]

            if all(finished):
                break

            # Run every slot's step, including finished ones (throwaway pass,
            # simpler than shrinking the batch; correctness-neutral since
            # `capture.active` / `generated` already exclude them).
            running_mask = torch.cat(
                [running_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1
            )
            out = model(
                input_ids=next_token,
                past_key_values=past,
                attention_mask=running_mask,
                use_cache=True,
                **extra_kwargs,
            )
            past = out.past_key_values

            for i in range(B):
                if not finished[i]:
                    generated[i].append(int(next_token[i, 0].item()))
                    if capture_generate_outputs:
                        step_hidden_states[i].append(
                            tuple(h[i : i + 1].detach().cpu() for h in out.hidden_states)
                        )
                        step_attentions[i].append(
                            tuple(a[i : i + 1].detach().cpu() for a in out.attentions)
                        )
                        # (batch, seq=1, vocab) -> (batch, vocab), matching
                        # GenerateDecoderOnlyOutput.logits for hallushift's
                        # probability_function (expects (vocab,) per step).
                        step_logits[i].append(out.logits[i : i + 1, -1].detach().cpu())

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        capture._recording = False

    if all(not g for g in generated):
        empty = [
            ({v: torch.empty(0) for v in VIEWS}, torch.empty(0, dtype=torch.long))
            for _ in range(B)
        ]
        if not capture_generate_outputs:
            return empty
        return [
            (acts, ids, {"hidden_states": (), "attentions": (), "logits": ()})
            for acts, ids in empty
        ]

    qkv_by_slot = capture.stack()  # {view: {slot: (T_slot, L, D)}}

    results = []
    for i in range(B):
        n_gen = len(generated[i])
        row_out = {}
        for view in VIEWS:
            tensor = qkv_by_slot[view][i]
            if tensor.numel() > 0 and tensor.shape[0] != n_gen:
                raise RuntimeError(
                    f"row {i} view {view}: captured {tensor.shape[0]} token "
                    f"activations but generated {n_gen} tokens. The "
                    "prefill/decode split or per-slot masking is wrong."
                )
            row_out[view] = tensor
        gen_ids_tensor = torch.tensor(generated[i], dtype=torch.long, device=device)
        if not capture_generate_outputs:
            results.append((row_out, gen_ids_tensor))
        else:
            results.append(
                (
                    row_out,
                    gen_ids_tensor,
                    {
                        "hidden_states": tuple(step_hidden_states[i]),
                        "attentions": tuple(step_attentions[i]),
                        "logits": tuple(step_logits[i]),
                    },
                )
            )
    return results
