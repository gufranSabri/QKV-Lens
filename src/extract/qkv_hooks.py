"""Capture per-layer Q, K and V activations from a HuggingFace causal LM.

HuggingFace exposes hidden states (`output_hidden_states=True`) and attention
probabilities (`output_attentions=True`) for free, but gives you NOTHING for the
Q/K/V projections -- they are internal to the attention module's forward(). This
module fills that gap with forward hooks on the projection Linears.

WHAT THE HOOK SEES
------------------
In every RoPE-family decoder (Llama, Mistral, Qwen, ...) the attention forward
begins:

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, ...)

so a forward hook on `q_proj` / `k_proj` / `v_proj` yields the projection output
BEFORE the head reshape and BEFORE RoPE, with shape (B, seq, n_heads*head_dim).

This is deliberate. We capture PRE-RoPE Q/K:
  - RoPE is a position-dependent rotation. Post-RoPE Q/K would entangle the
    token's *content* with its *absolute position*, so identical content at
    different positions would produce different fields. We want content.
  - V never receives RoPE at all, so for V the distinction does not exist and
    pre/post capture are identical.

QKV-Steer NOTE: the steering stage writes back at exactly this point -- the
output of the projection Linear, before RoPE and before the head reshape -- so
the coordinates the detector reads and the coordinates the steerer writes are
the same by construction. See src/steer/ for the write path.

GROUPED-QUERY ATTENTION (GQA)
-----------------------------
K and V are NARROWER than Q in every model we use:

    Llama-3-8B    D_q = 4096 (32 heads x 128)   D_kv = 1024 (8 kv-heads x 128)
    Mistral-7B    D_q = 4096 (32 heads x 128)   D_kv = 1024 (8 kv-heads x 128)
    Qwen2.5-7B    D_q = 3584 (28 heads x 128)   D_kv =  512 (4 kv-heads x 128)

The field builder pools each projection's D down to the same M segments, so all
three land on one channel axis -- only the pooling segment width differs. Never
hardcode any of these numbers; read them from `model.config`.

GENERATION SEMANTICS
--------------------
With a KV cache active, each attention module is called:
  - once during PREFILL, with seq = prompt_len   (the prompt's Q/K/V)
  - once per DECODE step, with seq = 1           (the new token's Q/K/V)

We record only the decode steps, so we end up with exactly one Q/K/V vector per
layer per GENERATED token -- precisely the (T, L, D) tensor we want. The prefill
activations are the prompt's, not the response's, and are discarded.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn

#: The three projections captured, in the channel order the feature field uses.
VIEWS = ("Q", "K", "V")
_PROJ_FOR_VIEW = {"Q": "q_proj", "K": "k_proj", "V": "v_proj"}


@dataclass
class ModelGeometry:
    """The per-model numbers the field builder needs. Read, never assumed."""

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
        # Q is d_q; K/V are the (GQA-narrower) d_kv.
        return self.d_q if view == "Q" else self.d_kv

    def __str__(self) -> str:
        return (
            f"L={self.n_layers} hidden={self.hidden_size} "
            f"heads={self.n_heads} kv_heads={self.n_kv_heads} "
            f"head_dim={self.head_dim} D_q={self.d_q} D_kv={self.d_kv}"
        )

    def valid_n_segments(self) -> list[int]:
        """Segment counts that evenly divide EVERY projection's feature dim.

        Q and K/V have different widths under GQA, so a legal M must divide
        their GCD. Notably Qwen2.5-7B (L=28, D_kv=512) has NO valid M equal to
        its layer count -- 28 does not divide 512 -- so it cannot use the
        default square field and must set extract.n_segments explicitly.
        """
        from math import gcd

        g = 0
        for view in VIEWS:
            g = gcd(g, self.feature_dim(view))
        return [d for d in range(1, g + 1) if g % d == 0]

    def check_n_segments(self, n_segments: int) -> None:
        """Raise with an actionable message if M cannot pool every projection."""
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
    """Extract layer/head geometry from a model config, with GQA fallbacks."""
    cfg = model.config
    # Some configs nest the text config (multimodal wrappers).
    cfg = getattr(cfg, "text_config", cfg)

    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    # head_dim is explicit on newer configs; otherwise it's the classic ratio.
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads
    # Models without GQA omit num_key_value_heads; then kv heads == q heads.
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads

    return ModelGeometry(
        n_layers=cfg.num_hidden_layers,
        hidden_size=hidden,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
    )


def get_decoder_layers(model) -> nn.ModuleList:
    """Locate the ModuleList of decoder layers across model wrappers."""
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
    """The `q_proj`/`k_proj`/`v_proj` Linear at `layer_idx`.

    Shared by the read path (hooks below) and the steering write path, so both
    address the identical module -- a coordinate the detector attributed to is
    guaranteed to name the module the steerer perturbs.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; valid views are {VIEWS}")
    layers = get_decoder_layers(model)
    return getattr(layers[layer_idx].self_attn, _PROJ_FOR_VIEW[view])


@dataclass
class QKVCapture:
    """Accumulates hook firings, keyed by (view, layer), split by decode step.

    Handles a batch of B sequences at once. Each decode step fires once per
    (view, layer) with a (B, 1, D) tensor; we keep one growing list PER BATCH
    SLOT so that after generation each sequence's own steps can be concatenated
    and truncated to its own true length (sequences finish at different steps).
    """

    n_layers: int
    views: tuple[str, ...]
    batch_size: int = 1
    # steps[view][layer][slot] -> list of (D,) tensors, one per decode step
    # while that slot was still active.
    steps: dict = field(default_factory=dict)
    _recording: bool = False
    # Per-slot: True while the sequence in that batch position is still being
    # decoded. Set by the caller before each decode step; a firing for a slot
    # that has already finished is NOT recorded (it would be past-EOS filler).
    active: list = field(default_factory=list)

    def __post_init__(self):
        self.steps = {
            v: [[[] for _ in range(self.batch_size)] for _ in range(self.n_layers)]
            for v in self.views
        }
        self.active = [True] * self.batch_size

    def record(self, view: str, layer: int, out: torch.Tensor) -> None:
        """out: (B, seq, D) -- the raw projection output for this firing.

        seq is always 1 here (one decode step -> one new position per slot);
        the caller's manual decode loop guarantees this, unlike model.generate.
        """
        if not self._recording:
            return
        if out.shape[1] != 1:
            raise RuntimeError(
                f"expected seq=1 per decode step, got seq={out.shape[1]}. "
                "This means a prefill-shaped tensor reached record() while "
                "recording was on -- the prefill/decode split is wrong."
            )
        # Detach immediately and move off the accelerator: keeping these on GPU
        # across a 100-token generation is what blows up memory.
        cpu_out = out[:, 0].detach().to("cpu", torch.float32)  # (B, D)
        for slot in range(self.batch_size):
            if self.active[slot]:
                self.steps[view][layer][slot].append(cpu_out[slot])

    def stack(self) -> dict[str, dict[int, torch.Tensor]]:
        """Concatenate captured steps into {slot: (T_slot, L, D)} per view.

        Slots stop accumulating once they finish (see `active`), so T_slot
        naturally varies across the batch -- that is the whole point of
        tracking it per slot rather than assuming one shared T.
        """
        out: dict[str, dict[int, torch.Tensor]] = {}
        for view in self.views:
            per_slot: dict[int, torch.Tensor] = {}
            for slot in range(self.batch_size):
                per_layer = []
                for layer in range(self.n_layers):
                    chunks = self.steps[view][layer][slot]
                    if not chunks:
                        # This slot generated zero tokens (immediate EOS).
                        per_layer.append(torch.empty(0))
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
    """Attach Q/K/V projection hooks for the duration of the context.

    Yields the QKVCapture. Recording is OFF on entry, so a prefill pass can run
    without being recorded; `capture_all` manages the `_recording` flag.

    Hooks are always removed on exit, including on exception.
    """
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
    """Left-pad a list of (1, len_i) or (len_i,) prompts to (B, max_len).

    Left-padding (not right) is what keeps "the last real token" at the same
    column (-1) for every sequence, so a single `logits[:, -1, :]` gives the
    correct next-token argmax for the whole batch at once. Right-padding would
    put that position at a different column per sequence and need per-row
    gathers.

    Returns (input_ids, attention_mask), both (B, max_len) on `device`.
    """
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
    """Greedy-generate and capture per-layer Q/K/V for every generated token.

    Args:
        input_ids: (B, prompt_len) LEFT-PADDED prompt token ids, on the model
            device. B == 1 and no padding is the common case; for B > 1, pass
            `attention_mask` from `left_pad_batch` so padded positions are
            excluded from attention.
        capture_generate_outputs: also request hidden_states/attentions/logits
            on every decode-step forward pass (needed by baselines other than
            QKV-Steer's own Q/K/V field, e.g. scripts/reproducing_baselines/hallushift's
            per-token Wasserstein/cosine/probability features). Off by default:
            output_attentions requires attn_implementation="eager" and returns
            a full (B, heads, seq_len, seq_len) tensor every step, so this is
            real extra memory/time callers that don't need it shouldn't pay.
            REQUIRES B == 1 -- the caller (run_extraction.py) forces batch_size
            to 1 whenever this is set, since slicing per-row hidden_states/
            attentions out of a padded batch is not implemented here.

    Returns:
        Without capture_generate_outputs: a list of B (activations, generated_ids)
        pairs, one per input row, exactly as before.

        With capture_generate_outputs: a list of B (activations, generated_ids,
        generate_outputs) triples, where generate_outputs is
        {"hidden_states": tuple, "attentions": tuple, "logits": tuple}, each a
        tuple of T_i per-step tensors -- the SAME shapes model.generate(...,
        output_hidden_states=True, output_attentions=True, output_logits=True)
        would return, so hallushift's own feature functions can consume them
        unmodified.

        activations["Q"] is (T_i, L, D_q), ["K"]/["V"] are (T_i, L, D_kv),
        generated_ids is (T_i,) -- that row's response tokens, prompt excluded.
        T_i is the number of tokens THAT ROW generated before its own EOS (or
        max_new_tokens), so it varies across the returned list; this is not a
        padded tensor.

    Only decode-step activations are recorded. The prefill pass -- which carries
    the PROMPT's activations, not the response's -- is explicitly excluded by
    leaving recording off until the prompt has been consumed.

    We drive the decode loop by hand rather than calling `model.generate`,
    because generate() gives us no reliable hook to distinguish prefill from
    decode; doing it manually makes the prefill/decode boundary explicit and
    unambiguous, which is exactly the thing that would otherwise silently
    corrupt every tensor we produce.
    """
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
    # Per-slot lists of per-step outputs, only populated when requested.
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
        # The token fed at step t is the token GENERATED at step t-1 (or the
        # argmax of the prompt's last logit, for t=0). So the activations
        # captured at step t are exactly what the model computed *for* that
        # generated token -- the correspondence we want between a field and a
        # response token.
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
            # Slots already finished BEFORE this step must not have their
            # activations recorded on this firing (nothing new to capture for
            # them); slots finishing ON this step still get a real logits row
            # from a real forward pass, but their token IS eos, so it is never
            # appended to `generated` and is excluded below.
            capture.active = [not f for f in finished]

            if all(finished):
                break

            # Append the current padding-mask column, then run the step for
            # EVERY slot (finished ones just get a throwaway forward pass --
            # simpler and correctness-neutral vs. shrinking the batch, since
            # already-finished slots are excluded from `capture.active` and
            # their `generated` list is simply never appended to below).
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
                        # Move off-device immediately, same reasoning as
                        # QKVCapture.record: keeping these on GPU across a
                        # 100-token generation is what blows up memory.
                        step_hidden_states[i].append(
                            tuple(h[i : i + 1].detach().cpu() for h in out.hidden_states)
                        )
                        step_attentions[i].append(
                            tuple(a[i : i + 1].detach().cpu() for a in out.attentions)
                        )
                        # (batch, seq=1, vocab) -> (batch, vocab): matches
                        # transformers' GenerateDecoderOnlyOutput.logits shape
                        # exactly (seq already squeezed there), which is what
                        # hallushift's probability_function assumes when it
                        # does F.softmax(logit[0], dim=0) expecting (vocab,).
                        step_logits[i].append(out.logits[i : i + 1, -1].detach().cpu())

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        capture._recording = False

    if all(not g for g in generated):
        # Every row emitted EOS immediately: no response tokens, no fields.
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
