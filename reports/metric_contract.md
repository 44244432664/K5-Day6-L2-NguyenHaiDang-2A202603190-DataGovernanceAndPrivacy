# T05 Metric Contract — Frozen Before Official Evaluation

> This contract records how the unchanged evaluator will be used. The instructor
> GT overlay is not installed, so no official metric is claimed here. Thresholds,
> candidate-pool rules, and seed must not change after official results are seen.

| Item | Frozen declaration |
|---|---|
| Track | T05 · Leakage & Near-Duplicate Guardian |
| Primary metric | **Leakage Detection F1 (pair)** — exact evaluator name |
| Direction | Higher is better |
| Evaluation unit | Canonical unordered pair of distinct image IDs |
| Positive GT | Both images share `dup_group_id`; evaluator expands every GT group to all within-group pairs |
| Positive prediction | Pair row is positive; when `pred_dup` exists, only `pred_dup == 1` rows are positive |
| Matching | Exact normalized unordered ID pair; no IoU/tolerance |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2PR / (P + R)`; shared helper returns 0 when precision or recall is zero/undefined |
| Candidate pool | Production exports all `C(120,2) = 7,140` public-fixture pairs with explicit 0/1; official GT pair denominator remains hidden |
| Review budget | `B = 20` pair relationships |
| Seed | `42` |
| Threshold policy | Frozen values in `configs/tracks/t05.yaml`; selected from design assumptions/generated internal examples only, never official GT |
| Eval split | T05 evaluator has no generic split filter. Use full instructor overlay exactly as installed; never tune on it |
| Primary bad outcomes | FN near-duplicate pair; FP pair/false scene merge |
| Split bad outcomes | GT duplicate or same-scene pair crossing retained train/test; image drop/retention loss |
| Required slices | Instructor `transform` slice; internal exact/JPEG/brightness/crop; same-scene nonduplicate; visually similar different-scene |
| Small-N uncertainty | Internal evidence uses scene-cluster bootstrap, not independent pair resampling. Official point metrics retain evaluator output and report N/numerator/denominator |
| Freeze record | 2026-11-15; development protocol v2 revision requested by the user before any instructor GT evaluation; human sign-off pending |

## Secondary metrics from the unchanged evaluator

| Metric | Formula | Direction |
|---|---|---|
| Pair precision | `TP / predicted positive pairs` | Higher |
| Pair recall | `TP / GT duplicate pairs` | Higher |
| False merge rate | Predicted pairs whose members have different `scene_id` / predicted pairs | Lower |
| Residual cross-split leak (near-dup) | Kept GT duplicate pairs assigned to different splits / kept GT duplicate pairs | Lower |
| Residual cross-split leak (scene) | Kept same-scene pairs assigned to different splits / kept same-scene pairs | Lower |
| % data kept | Images not missing/blank/`drop` / all GT images | Higher |

`split_manifest.csv` must sit beside each prediction file so residual leakage and
retention are evaluated. Baseline and PoC use the same fixture, seed, evaluator,
and output pair representation.

## Frozen production policy

- pHash/dHash: 8×8, Hamming limits 8/8.
- Histogram: 16 bins/channel, L1 limit 0.08; supporting evidence only.
- Thumbnail: 32×18 grayscale, normalized MAE limit 0.10.
- Duplicate score: `0.30 pHash + 0.30 dHash + 0.30 thumbnail + 0.10 histogram`.
- Original coarse gate: score at least 0.75 and at least two spatial/structural
  signals, unless exact SHA-256.
- Review band: 0.55–0.85; budget 20.
- Scene burst: at most 60 seconds per edge and 60 seconds total burst span.
- Target test fraction: 0.30; seed 42; whole split groups only.
- Release: zero unresolved reviews and zero dropped fraction by default.

### Pre-official-evaluation revision: protocol v2

The user requested mitigation of exposed synthetic-development failures. No
instructor GT or official metric was accessed. Original coarse thresholds,
review budget, seed, and split target remain unchanged. The new configuration
adds:

- `verification_enabled=1`; `0` is an explicit legacy development ablation.
- RGB views 48×32; centered crop fractions 0, .02, .04, .06, .08, .10, .12.
- Shared RGB exposure gain constrained to `[0.625,1.6]`, no per-channel recoloring.
- Symmetric relative RGB MAE maximum 0.05; minimum spatial texture std 0.02.
- Nonexact auto-positives require verification; verified crop alignment may
  recover duplicates missed by global hashes. Exact SHA-256 remains positive.
  A verified, textured registered crop is an alternative spatial gate; its global
  hashes need not still agree after the crop.
- Review selection balances accepted, withheld, bridge, and color-trap lanes,
  with component-relationship diversity at the same budget 20.

Previously exposed diagnostic examples are now explicitly development/regression
data, not a pristine holdout, regardless of their old internal partition names.
Two new source sets (seeds 2027 and 9173) were generated after fixing these
settings and were not used to retune them. They use different dimensions/layouts,
7% crops, JPEG quality 55, and brightness factor 0.9. Their results are synthetic
confirmation only, not official generalization estimates.

Historical v1 artifacts remain preserved. Use fresh `out/poc_v2/` and new reviews;
old review fingerprints must not be bypassed. Human approval and real Git commit
provenance are still pending.

## Internal versus official evidence

`tools/t05_experiments.py` may evaluate explicit generated/internal labels and
always marks them `internal_development_only_not_official`. These values must not
be copied into `reports/metrics_baseline.json` or `reports/metrics_poc.json`.

Official files are generated only after the instructor overlay is installed:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --track t05 --pred out\baseline\suspect_pairs.csv --run-name baseline --out reports\metrics_baseline.json
.\.venv\Scripts\python.exe -m src.evaluate --track t05 --pred out\poc_v2\suspect_pairs.csv --run-name poc --out reports\metrics_poc.json
.\.venv\Scripts\python.exe -m src.compare reports\metrics_baseline.json reports\metrics_poc.json --out reports\table1_baseline_vs_poc.md
```

If baseline F1 is zero, report `baseline = 0`; do not calculate a misleading
percentage improvement. Negative or unchanged results must also be retained.

## Current status

- Instructor GT overlay: **not installed**.
- `reports/metrics_baseline.json`: **not generated**.
- `reports/metrics_poc.json`: **not generated**.
- Official comparison table: **not generated**.
- Threshold changes after official evaluation: **forbidden**.
