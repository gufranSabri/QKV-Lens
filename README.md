# QKV-Lens

**From correlational attribution to causal intervention on pre-attention Q/K/V projections.**

Two-pass inference over a frozen LLM and a frozen detector:

1. **Locate** — generate once, extract the QKV feature field, run the detector,
   compute Integrated Gradients attribution over the field.
2. **Steer** — re-run generation with the pre-attention Q/K/V projections
   perturbed according to that attribution.

## Status

**Phase 1 (locate) is implemented.**

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
src/models/          the detector: flat_mlp -> Conv1d -> BiLSTM -> attention pool -> head
src/data/            field dataset, normalisation
src/cam.py           Integrated Gradients attribution over the field
configs/             one config per (dataset, LLM)
docs/plan.md         the research plan
```

## The feature field

Each generated token becomes one `L x M x 3` image:

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

The hidden-states variant (`hs*`) is the same construction over each layer's
residual stream: one channel `H` instead of three, tiled to `(H, H, H)` at
load time so the detector sees the same input shape.

## The detector

```
per token:   (L, M, 3) field -> flat_mlp (flatten, dropout, Linear -> E)
over tokens: Conv1d x2 (k=3) -> BiLSTM -> masked attention pool -> head (LayerNorm, Dropout, Linear -> 1 logit)
```

Every component but the head can be switched to a parameter-free stand-in,
which is what `scripts/experiments/component_ablation.sh` sweeps:

| Key | Off means |
|---|---|
| `model.use_proj: false` | flatten only; the field itself is the token embedding |
| `model.conv1d_layers: 0` | no Conv1d |
| `model.use_lstm: false` | identity instead of the BiLSTM |
| `model.use_attn_pool: false` | masked mean over real tokens instead of attention |

## Usage

```bash
python detector.py --config configs/triviaqa/llama3.1_8b.yaml extract
python detector.py --config configs/triviaqa/llama3.1_8b.yaml extract --variants all
python detector.py --config configs/triviaqa/llama3.1_8b.yaml train --run-name my_run
python detector.py --config configs/triviaqa/llama3.1_8b.yaml test  --checkpoint runs/my_run/best.pt
python detector.py --config configs/triviaqa/llama3.1_8b.yaml cam    --checkpoint runs/my_run/best.pt --idx 0
python detector.py --config configs/triviaqa/llama3.1_8b.yaml inspect --idx 0
```

`--set key=value` overrides any config key from the command line, on either side
of the subcommand. All config keys are documented in `configs/default.yaml`.

Set `data_root` in `configs/default.yaml` before extracting. On-disk layout,
one tree per `(dataset, LLM)`, one subdirectory per field variant:

```
{data_root}/{dataset}/{llm_alias}/
    judge_cache.jsonl      llm_judge labels, shared by every variant
    default/               QKV, mean pool, the model's default M
        00000/tokens.npy   (T, L, M, C) float16, C=3 (Q, K, V) or 1 (H)
        00000/meta.txt     prompt / response / gold / score / label
        manifest.jsonl     the training index
        geometry.json      the model geometry the fields were built with
    qkv_M8/  hs/  hs_M16/  qkv_max/  ...
```

`extract` saves `default/` only; `--variants qkv_M8,hs,...` (or `--variants all`)
saves more from the same generation. A variant is named
`{qkv|hs}[_{max|min|strided}][_M{n_segments}]`, and train/test/cam pick theirs
from `extract.source`, `extract.pool` and `extract.n_segments`.

## Datasets and models

TriviaQA, TruthfulQA, CoQA × LLaMA-2-7B, LLaMA-3.1-8B, OPT-6.7B, Qwen2.5-7B,
one config each under `configs/`.

Labels default to an LLM judge (`labeling.scheme: llm_judge`, via Novita AI)
comparing the response against the gold references. `labeling.scheme: bleurt`
reproduces HalluShift's exact protocol instead (max BLEURT-20-D12 against the
gold references, thresholded at 0.5). Judge calls are cached in
`judge_cache.jsonl`, so an interrupted labeling run resumes where it stopped.
`python detector.py ... label` recomputes labels from stored responses for
every variant of a `(dataset, LLM)`, without re-running the LLM.

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
`{data_root}/{dataset}/{llm_alias}/default/*/meta.txt`, against the labels already in
`manifest.jsonl`, on the test indices already in
`{runs_root}/{llm_alias}_{dataset}/split.json` -- the same partition
`src/train.py` gave QKV-Lens and `scripts/run_training.py` gave HalluShift.
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
bash scripts/install.sh --baselines
```

The core install already covers `labeling.scheme: llm_judge` (the default);
`--bleurt` additionally pulls the TensorFlow BLEURT scorer for
`labeling.scheme: bleurt` (the HalluShift comparison); `--baselines` adds
`rouge_score` / `spacy` / `sentencepiece` and the rest the baselines above need.

`labeling.scheme: llm_judge` calls Novita AI's API and needs `NOVITA_API_KEY`
set in `scripts/.env` or the shell environment.

`scripts/troubleshooting.sh` is the annotated step-by-step version of the whole
pipeline; `detector.slurm` runs it as a batch job.

## Experiments

`scripts/experiments/all_experiments.sh` runs everything, in order, and ends
with a digest at `docs/reports/ablation_summary.md`. Every step skips output
that already exists; `RERUN=1` redoes it all.

| Step | What | Script |
|---|---|---|
| 0 | Extract TriviaQA × 4 LLMs with `--variants all`; TruthfulQA, CoQA default only | `detector.py extract` |
| 1 | Hidden states vs Q/K/V linear probes (3 LLMs) | `scripts/analysis/hidden_vs_qkv.py` |
| 2 | Component ablation (LLaMA-3.1-8B) | `component_ablation.sh` |
| 3 | Representation ablation: Q/K/V subsets + hidden states (4 LLMs) | `representation_ablation.sh` |
| 4 | Pooling ablation: M ∈ {8, 16, 32, 64, 128} for QKV and H, max/min at M=32 | `pooling_ablation.sh` |
| 5 | Collapse-axis ablation: average out M, L or channels | `collapse_axis_ablation.sh` |
| 6 | Permutation ablation: shuffle L, M or both, 3 seeds | `permutation_ablation.sh` |
| 7 | Tables for steps 2–6 | `scripts/tables/*.py` |
| 8 | Majority-vote representation → main results (4 LLMs × 3 datasets) → baselines | `run_all_baselines.sh` |
| 9 | Significance, prototype maps, IG heatmap, forecasting | `scripts/analysis/`, `scripts/figures/` |
| 10 | Digest | `scripts/analysis/ablation_summary.py` |

Sweep scripts live in `scripts/experiments/` and each runs on its own. Steps
2–6 train only: every field they need is a variant step 0 already saved.
`scripts/experiments/reproduce_all.sh` is the older driver, which also runs
cross-model transfer and token-axis compression.
