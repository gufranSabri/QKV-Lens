# Redirection plan

Three changes, then a new `all_experiments.sh`. Implement in the order of the
steps below, one step per sitting. Open decisions are collected at the end;
each step lists the ones it needs settled first.

---

## Step 1 — LLM judge is the default labeler ✅ done

Decided: BLEURT and exact_match stay selectable via `labeling.scheme`;
`llm_judge` is the default. Local `bleurt/` and `models/BLEURT-20-D12/` kept.

Done:
- Resumable judge cache: `judge_cache.jsonl` next to each corpus's `manifest.jsonl`,
  keyed by sha256(model, system prompt, response, golds). Failed calls are not
  cached, so a rerun retries them. API client is only built on a cache miss.
- BLEURT-specific wording made scheme-neutral (`run_extraction.py` comments,
  corpus_stats "Mean score", export CSV `score` column, main_results /
  run_all_baselines text).

Not done (declined): parallel calls, failing loudly (a failed call is still labeled 0).

Still to do before training: relabel every corpus under `data_root` with the
judge (`detector.py ... label`), and train into fresh run dirs (`split.json`'s
cache key has no label hash).

---

## Step 2 — `layer_grid` removed entirely ✅ done

`flat_mlp` is the only token projector, so `model.backbone` stops being a key.

**Delete**
- `src/models/backbones/layer_grid.py`
- `scripts/experiments/layer_grid_grid.sh`, `scripts/tables/layer_grid_grid.py`
- `scripts/experiments/layer_grid_component_ablation.sh`, `scripts/tables/layer_grid_component_ablation.py`
- `scripts/experiments/structure_analysis.sh`, `scripts/tables/structure_analysis.py`
  — its `gridbb` arm is gone, and `flat` vs `permL42` is exactly what
  `permutation_ablation.sh` already measures (with 3 seeds and the M axis too)

**Edit**
| File | Change |
|---|---|
| `src/models/backbones/` | package removed; `FlatMLP` now lives in `src/models/flat_mlp.py`, built directly by `classifier.py` |
| `src/config.py` | drop `VALID_BACKBONES`, `backbone`, `layer_grid_use_{gate,conv,skip}` and their validation |
| `configs/default.yaml` | same keys removed |
| `src/test.py` | `layer_grid_use_*` mismatch loop removed (Step 3 adds one for its switches) |
| `src/train.py:123,370` | backbone log line; `default_run_name` → `{alias}_{dataset}` |
| `scripts/tables/collect_runs.py:42-50` | `backbone` / `layer_grid_use_*` fields |
| `scripts/analysis/ablation_summary.py` | delete `section_layer_grid_grid`, `section_structure` |
| `scripts/experiments/permutation_ablation.sh`, `scripts/tables/permutation_ablation.py` | drop `--set model.backbone=layer_grid`; baseline is the default model |
| `scripts/experiments/all_experiments.sh` | layer_grid + structure steps cut (full restructure is Step 5) |
| `all-datasets_run.sh:59` | `--model.backbone=flat_mlp` line |
| `README.md` | architecture section |

Kept as-is: the module attribute `model.backbone` and `train.backbone_lr_scale`
(they name the role, not a choice).

---

## Step 3 — Detector components become switchable ✅ done

Decided: no head ablation (the head is always present); `no_proj` = flatten only.

### Current detector (llama3.1_8b, field 3×32×32 → 3 072 per token)

| # | Component | What it does | Params |
|---|---|---|---|
| 1 | `FlatMLP` | flatten → dropout → Linear(3072 → 2048) | 6.3 M |
| 2 | `Conv1d` ×2 | k=3 over tokens, BN, GELU | 25.2 M |
| 3 | `BiLSTM` | 2048 → 2×2048 | 67.1 M |
| 4 | `MaskedAttentionPool` | Linear(4096 → 1) score, softmax over real tokens | 4 k |
| 5 | head | LayerNorm → Dropout → Linear(4096 → 1) — always present, not ablated | 12 k |

### Arms (leave-one-out)

| Arm | Replaced by | Notes |
|---|---|---|
| `full` | — | |
| `no_proj` | flatten → dropout (no Linear) | E becomes C·L·M = 3072, so Conv1d/LSTM widths grow: 140.6 M params vs 98.6 M |
| `no_conv` | identity (`conv1d_layers=0`, already supported) | |
| `no_lstm` | identity | pool/head see the 2048-d conv output; removes 67 M of ~99 M params *and* all long-range context (conv receptive field = 5 tokens) — note this confound in the table |
| `no_attn_pool` | masked mean over real tokens | |
| `probe` *(optional floor)* | flatten → masked mean → head (9 k params) | a logistic-regression probe on token-mean features: what the architecture buys over a linear probe |

### Config

Same pattern as the old `layer_grid_use_*` switches:

```yaml
model:
  use_proj: true        # false -> flatten only
  conv1d_layers: 2      # 0 -> no conv
  use_lstm: true        # false -> identity
  use_attn_pool: true   # false -> masked mean
```

### Code

| File | Change |
|---|---|
| `src/models/flat_mlp.py` | `use_proj=False` skips the Linear; `embed_dim` reports C·L·M |
| `src/models/temporal.py` | `use_lstm`, `use_attn_pool`; new `MaskedMeanPool`; `out_dim` follows |
| `src/models/classifier.py` | temporal input width follows `backbone.embed_dim` |
| `src/config.py`, `configs/default.yaml` | the 3 switches (no validation needed: the head always has params) |
| `src/train.py:261-278` | `no_proj` leaves the backbone param group empty → fall back to one group |
| `src/test.py` | mismatch warning over `use_proj`, `conv1d_layers`, `use_lstm`, `use_attn_pool` |

Unaffected: `cam.py`, `forecasting.py`, `qualitative_maps.py` — they only call `model(images, mask)`.

---

## Step 3b — Extraction variants ✅ done

One `extract` generation writes every requested field variant to
`{data_root}/{dataset}/{llm}/<variant>/`, each a self-contained corpus
(manifest, geometry, tokens, meta). `default/` = QKV, mean pool, model's default M.

- Names: `{qkv|hs}[_{max|min|strided}][_M{n}]`, e.g. `qkv_M8`, `hs`, `hs_M16`, `qkv_max`.
  train/test/cam resolve theirs from `extract.source` / `extract.pool` / `extract.n_segments`.
- `extract` saves only the config's own variant; `--variants a,b` or `--variants all`
  adds more (`all` = qkv_M{8,16,64,128}, hs, hs_M{8,16,64,128}, qkv_max, qkv_min).
- Hidden states come from the same batched decode (HF `hidden_states[1:]`, SDPA, any batch size).
  `extract-hidden-states` and the separate `hidden_states/` tree are gone.
- New pool mode `min`. Labels computed once per run; `judge_cache.jsonl` is shared at the pair level.
- An example missing from any requested variant is regenerated and written to all of them,
  so every variant of an example comes from one decode.
- `label` relabels every variant of a pair, scoring each unique (response, gold) once.
- `all_experiments.sh` step 0 extracts `--variants all`; its HS branch uses `--variants hs`.

Old-layout corpora (`{dataset}/{llm}/` and `hidden_states/{dataset}/{llm}/`) are not
found by the new code until moved into `default/` and `hs/`.

---

## Step 4 — Component ablation script + table ✅ done

Decided: llama3.1_8b × TriviaQA × 1 seed (42) → 6 runs.

- `scripts/experiments/component_ablation.sh`: arms full, no_proj, no_conv, no_lstm,
  no_attn_pool, probe; run names `comp_<arm>_<model>_<dataset>`.
- `scripts/tables/component_ablation.py` → `docs/tables/component_ablation.{md,csv}`:
  AUROC, Δ vs full, trainable params (`train` now writes `n_params` to `results.json`).
- `ablation_summary.py`: `section_component`.

---

## Step 5 — `all_experiments.sh` ✅ done

| Step | Contents |
|---|---|
| 0 | extract TriviaQA × 4 models `--variants all`; TruthfulQA, CoQA × 4 default only |
| 1 | `hidden_vs_qkv.py --refresh` (D8: yes, after extraction) |
| 2 | component ablation |
| 3 | representation ablation (restored to TriviaQA × 4 models) |
| 4 | pooling ablation, redesigned over the extracted variants (no re-extraction) |
| 5 | collapse-axis ablation |
| 6 | permutation ablation (new in this driver) |
| 7 | tables for 2–6 |
| 8 | vote → main results 4 × 3 → baselines |
| 9 | significance, qualitative maps, IG heatmap, forecasting |
| 10 | `ablation_summary.md` |

Fixed on the way:
- Representation HS arm crashed (default `keep_channels: [Q, K]` is rejected on hidden states) → `keep_channels=null`.
- Vote: QKV winning passed no override, so main results trained on Q+K → now `keep_channels=null`.
- Pooling ablation: `pool_<variant>_<model>_<dataset>` over default, qkv_M{8,16,64,128},
  qkv_max, qkv_min, hs, hs_M{8,16,64,128} (12 runs); extracts any missing variant first.
- `ablation_summary.py`: new sections for hidden_vs_qkv, component, permutation; pooling
  section reads the new CSV; sections renumbered.

D7: `reproduce_all.sh` kept as a separate driver (only its stale pooling note fixed).

Not done: old-layout corpora on `/scratch` left untouched (user's call); step 0 re-extracts.
Open: `keep_channels: [Q, K]` and `collapse_axis: null` defaults were chosen under
layer_grid + BLEURT; revisit once the new representation table exists.

---

## Step 6 — README ✅ done

New "The detector" section (components + switches), hidden-states variant, `--variants`
usage, judge cache / relabel, baselines path, and an "Experiments" section with the
all_experiments step table.

---

## Open decisions

None.
