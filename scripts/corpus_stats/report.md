# QKV-Steer migrated corpus stats

Computed from `manifest.jsonl` under the migrated `{data_root}/{dataset}/{llm_alias}/` tree. Pairs whose BLEURT relabeling hasn't finished yet show `unlabeled` > 0 and an empty hallucination rate for the missing rows.

## Per (dataset, model)

| Dataset | Model | Total | Labeled | Unlabeled | Hallucinated | Hallucination rate | Mean BLEURT | Mean response tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coqa | llama2_7b | 7983 | 7983 | 0 | 3556 | 44.5% | 0.550 | 8.9 |
| coqa | llama3.1_8b | 7983 | 7983 | 0 | 3286 | 41.2% | 0.552 | 5.4 |
| coqa | opt_6.7b | 7983 | 7983 | 0 | 4398 | 55.1% | 0.456 | 12.3 |
| coqa | qwen2.5_7b | 7982 | 7982 | 0 | 3375 | 42.3% | 0.544 | 6.5 |
| triviaqa | llama2_7b | 9960 | 9960 | 0 | 5728 | 57.5% | 0.460 | 21.2 |
| triviaqa | llama3.1_8b | 9960 | 9960 | 0 | 4116 | 41.3% | 0.557 | 7.5 |
| triviaqa | opt_6.7b | 9960 | 0 | 9960 | 0 | — | — | 19.2 |
| triviaqa | qwen2.5_7b | 9960 | 0 | 9960 | 0 | — | — | 7.0 |
| truthfulqa | llama2_7b | 817 | 817 | 0 | 701 | 85.8% | 0.309 | 32.6 |
| truthfulqa | llama3.1_8b | 817 | 817 | 0 | 716 | 87.6% | 0.267 | 16.6 |
| truthfulqa | opt_6.7b | 817 | 817 | 0 | 743 | 90.9% | 0.284 | 39.3 |
| truthfulqa | qwen2.5_7b | 817 | 817 | 0 | 764 | 93.5% | 0.192 | 8.8 |

## Per dataset (across all models)

| Dataset | Total | Labeled | Hallucinated | Hallucination rate |
|---|---:|---:|---:|---:|
| coqa | 31931 | 31931 | 14615 | 45.8% |
| triviaqa | 39840 | 19920 | 9844 | 49.4% |
| truthfulqa | 3268 | 3268 | 2924 | 89.5% |

## Per model (across all datasets)

| Model | Total | Labeled | Hallucinated | Hallucination rate |
|---|---:|---:|---:|---:|
| llama2_7b | 18760 | 18760 | 9985 | 53.2% |
| llama3.1_8b | 18760 | 18760 | 8118 | 43.3% |
| opt_6.7b | 18760 | 8800 | 5141 | 58.4% |
| qwen2.5_7b | 18759 | 8799 | 4139 | 47.0% |

## Overall

- Total examples: **75039**
- Labeled: **55119** (unlabeled: 19920)
- Hallucinated: **27383**
- Hallucination rate (over labeled examples): **49.7%**
