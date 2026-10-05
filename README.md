# QKV-Steer

**From correlational attribution to causal intervention on pre-attention Q/K/V projections.**

Follow-up to *QKV-Lens: QKV Projections as Visual Fields for Hallucination
Detection*. The research plan lives in [docs/plan.md](docs/plan.md).

Two-pass inference over a frozen LLM and a frozen QKV-Lens detector:

1. **Locate** — generate once, extract the QKV feature field, run the detector,
   compute Integrated Gradients attribution over the field.
2. **Steer** — re-run generation with the pre-attention Q/K/V projections
   perturbed according to that attribution.

## Status

**Phase 1 (locate) is implemented** — this is the QKV-Lens pipeline, cleaned to
the paper's canonical method with the ablation surface removed.

**Phase 2 (steer) is not yet written.** The hooks it will need are already in
place: `src/extract/qkv_hooks.get_projection` addresses the exact module the
read path hooks, and `src/cam.attribution_field` returns attribution in the
field's own `(token, layer, segment)` coordinates.

## Repo layout

`detector.py` is the CLI for the **detector stage only** — extract, label,
train, test, inspect, cam. It is named for what it is rather than `main.py`,
because it is a supporting tool: it fits `f_theta` and produces the attribution
that the steering stage consumes. The steering entry point will be a separate
top-level script, not another subcommand here.

```
detector.py          detector-stage CLI (extract / train / test / cam / ...)
detector.slurm       batch job for that pipeline
src/extract/         generation, Q/K/V capture, feature-field construction
src/models/          the detector: backbone -> Conv1d -> BiLSTM -> head
src/models/backbones/  flat_mlp (main) | scratch_cnn
src/data/            field dataset, normalisation, QKV-Lens corpus reader
src/cam.py           Integrated Gradients attribution over the field
configs/             one config per (dataset, LLM)
docs/plan.md         the research plan
```

## The feature field

Each generated token becomes one `L x M x 3` image (QKV-Lens Algorithm 1):

```
F  ∈  R^{T × L × M × 3}
        │   │   │   └── channel: Q, K, V   (src.extract.tensor_ops.PROJECTIONS)
        │   │   └────── M pooled feature segments, S = D/M dims each
        │   └────────── L transformer layers
        └────────────── T generated tokens
```

Q/K/V are captured **pre-RoPE**, from forward hooks on each layer's
`q_proj`/`k_proj`/`v_proj`, on decode steps only. Under GQA the three have
different widths (`D_q` vs `D_kv`); each is mean-pooled to the same `M`
independently, so all three land on one channel axis.

The detector's backbone turns each token's `(L, M, 3)` field into one
embedding (`model.backbone`, default `flat_mlp`: flatten + dropout + one
Linear, no spatial structure preserved -- see
`src/models/backbones/flat_mlp.py`'s docstring for why. `scratch_cnn`, a 2D
CNN over `(L, M)`, is the "with spatial structure" ablation arm). Either way,
the per-token embeddings then go through Conv1d + BiLSTM + masked attention
pooling over the token axis, then a binary head -- see `src/models/`.

## What is fixed vs. configurable

The method's own settings are **fixed in code**, not config keys — the field
layout, mean pooling, and greedy decoding. Changing any of them would change
what a `(layer, segment, projection)` coordinate *means*, and the steering stage
would then write into a different space than the one the detector looked at.

Configs that use a removed QKV-Lens option fail loudly with the reason
(see `_REMOVED_KEYS` in `src/config.py`). Removed: `extract.source`,
`extract.extraction_type`, `extract.views`, `extract.boundary_mode`,
`extract.n_cols` (now `n_segments`), `model.channels`, `model.include`,
`model.fusion`, `model.share_backbone`, `model.fused_dim`. `extract.pool`
is NOT removed — it's the one opt-in pooling-ablation knob (see `src/config.py`).

## Usage

```bash
python detector.py --config configs/triviaqa/llama2_7b.yaml extract
python detector.py --config configs/triviaqa/llama2_7b.yaml train --run-name my_run
python detector.py --config configs/triviaqa/llama2_7b.yaml test  --checkpoint runs/my_run/best.pt
python detector.py --config configs/triviaqa/llama2_7b.yaml cam    --checkpoint runs/my_run/best.pt --idx 0
python detector.py --config configs/triviaqa/llama2_7b.yaml inspect --idx 0
```

`--set key=value` overrides any config key from the command line, on either side
of the subcommand.

Set `data_root` in `configs/default.yaml` before extracting.

**A QKV-Lens corpus is read directly — no re-extraction.** Point `data_root` at
one and it is detected from its `geometry.json` and converted on read: its
channel 0 *is* the mean-pooled activation Algorithm 1 defines, so
`old[..., 0]` with the view axis moved last recovers the field exactly. The
extra channels (DWT, or layer deltas) are derived quantities this method does
not use. A corpus pooled any other way is rejected rather than reinterpreted
(`src/data/legacy.py`). The old `{source}/{extraction_type}/` nesting is
resolved automatically; a native corpus always wins over a legacy one.

On-disk layout for a **new** extraction, one tree per `(dataset, LLM)`:

```
{data_root}/{dataset}/{llm_alias}/
    00000/tokens.npy   (T, L, M, 3) float16
    00000/meta.txt     prompt / response / gold / score / label
    manifest.jsonl     the training index
    geometry.json      the model geometry the fields were built with
```

## Datasets and models

TriviaQA, TruthfulQA, CoQA × LLaMA-2-7B, LLaMA-3.1-8B, OPT-6.7B, Qwen2.5-7B —
the 12 settings from QKV-Lens, one config each under `configs/`.

Labels follow HalluShift: max BLEURT-20-D12 against the gold references,
thresholded at 0.5. `python detector.py ... label` recomputes labels from stored
responses without re-running the LLM.

Qwen2.5-7B needs `extract.n_segments: 32` set explicitly — its `L=28` does not
divide its `D_kv=512`, so the square-field default is unavailable.

## Baselines

`scripts/experiments/run_all_baselines.sh` is the single entry point for everything the
paper compares against, across all 12 (dataset x LLM) settings, ending with
`docs/tables/main_results.md`:

```bash
bash scripts/experiments/run_all_baselines.sh
```

| Baseline | Where it comes from | Status |
|---|---|---|
| Perplexity | `scripts/reproducing_baselines/perplexity.txt` | run here |
| Lexical Similarity | `scripts/reproducing_baselines/lexical_similarity/` (Lin et al., TMLR 2024) | run here |
| SelfCheckGPT-NLI | `scripts/reproducing_baselines/selfcheckgpt/` (Manakul et al., 2023) | run here |
| Semantic Entropy | `scripts/reproducing_baselines/semantic_uncertainty/` (Kuhn et al., 2023) | run here |
| Verbalize | `scripts/reproducing_baselines/verbalize.pdf` (Lin et al., 2022) | run here |
| Self-Evaluation | `scripts/reproducing_baselines/selv-evaluation.pdf` (Kadavath et al., 2022) | run here |
| HalluShift | `scripts/run_training.py --methods hallushift` | already run |
| HaloScope, CCS | -- | out of scope; HaloScope is cited, not measured |

Protocol details for the six come from
`scripts/reproducing_baselines/main_instruction.txt`, which fixes the sampling
setting (10 generations at temperature 0.5) and the exact Verbalize /
Self-Evaluation prompts.

**They are run on our own generations, not their own.** Nothing is
regenerated: each baseline scores the greedy responses already in
`{data_root}/{dataset}/{llm_alias}/*/meta.txt`, against the BLEURT labels
already in `manifest.jsonl`, on the test indices already in
`{runs_root}/{llm_alias}_{dataset}/split.json` -- the same partition
`src/train.py` gave QKV-Steer and `scripts/run_training.py` gave HalluShift.
That is what makes the AUROC column comparable down the whole table.

Three stages, each resumable, under `scripts/baselines/`:

```
run_llm_baselines.py       loads the LLM once: perplexity, verbalized
                           confidence, P(True), and 10 sampled generations
run_sampling_baselines.py  the two NLI models: Rouge-L consistency,
                           SelfCheckGPT contradiction, semantic clustering
score_baselines.py         signs each score (high = hallucinated) and
                           evaluates it on split.json's test slice
```

Outputs land in `{data_root}/{method}/{dataset}/{llm_alias}/results.json`,
deliberately the same path shape and key spelling as HalluShift's, so
`scripts/tables/main_results.py` reads every method with one loader. These
baselines fit nothing, so only AUROC and PR-AUC are reported by default; run
with `SUBSET=all` to also get thresholded metrics (the threshold is picked on
the train split, never on test).

## Install

```bash
bash scripts/install.sh --bleurt --baselines
```

`--bleurt` pulls the TensorFlow BLEURT scorer (labeling); `--baselines` adds
`rouge_score` / `nltk` / `sentencepiece` for the baselines above.

`scripts/troubleshooting.sh` is the annotated step-by-step version of the whole
pipeline; `detector.slurm` runs it as a batch job.
