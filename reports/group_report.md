# Group Report — T05 Leakage & Near-Duplicate Guardian

> Status: 2026-11-15. Official metric fields remain pending because the
> instructor ground-truth overlay is not installed. Observable public-run
> diagnostics are not substitutes for hidden-label metrics.

> Submission-file policy: the user selected a strict instructor-listed allowlist.
> Extra documentation/environment snapshots referenced below (`DOC/`, `PLAN/`,
> `TASK/`, `docs/T05_*.md`, `.python-version`, `requirements-lock.txt`) are retained
> locally but ignored. The versioned submission reference is this report, README,
> metric contract, implementation docstrings/tests, and the named report artifacts.

## 00 · Header

| Item | Value |
|---|---|
| Group ID | G01 |
| Challenge | T05 · Leakage & Near-Duplicate Guardian |
| Known member / role | Nguyen Hai Dang · confirmed destination owner; migration/integration validation; implementation authorship remains pending Git evidence |
| Other team members | Not supplied; no names or contributions inferred |
| Git URL | <https://github.com/44244432664/K5-Day6-L2-NguyenHaiDang-2A202603190-DataGovernanceAndPrivacy> |
| Project commit | **Pending** — T05 changes are uncommitted |
| Tag `demo-180m` | **Pending** — no repository tags exist |
| Tag `final-24h` | **Pending** — no repository tags exist |
| Deadline | Not supplied |

Current `run_info.json` files show `b0ce8bf`, the upstream HEAD present during
validation. That SHA does **not** contain the uncommitted T05 working tree and is
not presented as the implementation commit.

This working tree was locally migrated from the DangQuocThai checkout. Its old
configured owner was Dang Quoc Thai; historical validation records retain that
context and are not rewritten as Nguyen Hai Dang's authored work. The user
confirmed Nguyen Hai Dang as the owner for new destination runs. No commit or push
was requested. See `docs/T05_ENVIRONMENT_AND_MIGRATION.md` for scope and provenance.

## 01 · Problem statement

Image duplicates and related captures can cross a train/test boundary, allowing a
model to see effectively the same content during training and evaluation. This
inflates measured quality and can cause an unsafe release decision. Data owners,
model developers, reviewers, evaluators, and people represented in real images
bear the consequences of false confidence, false merges, and inappropriate reuse.

The system takes an image table, image files, a provenance manifest, and frozen
T05 configuration. It emits pair predictions, duplicate and split groups, a
group-aware split manifest, review material, audit records, diagnostics, and a
GO/HOLD/REWORK decision.

Two relations remain separate:

- **Duplicate:** effectively the same visual content; emitted as a positive pair.
- **Scene/split constraint:** content should stay in one split but is not
  necessarily duplicate; affects only split grouping.

Pre-evaluation policy:

- **GO:** observable structural checks pass with no blocking conflict, unresolved
  review, crossing, oversized group, or dropped image.
- **HOLD:** ambiguity or an indivisible/oversized unsafe component requires a
  human or policy decision; the exact-hash/random-split baseline is always HOLD.
- **REWORK:** a deterministic structural defect exists, such as a predicted/group
  crossing or feasible but unusable split.

GO is provisional structural approval, not proof of zero true leakage. A data
owner must confirm metadata semantics and acceptable false-merge/miss costs;
DPO/legal review is required before using real personal data.

## 02 · Data and provenance

| Use | N | Source | License | Synthetic? |
|---|---:|---|---|---|
| Public fixture / smoke run | 120 images | `fixtures/t05_near_duplicate/`; instructor fixture tool, seed 42 | CC0-1.0 declared in `manifest.csv` | Yes |
| Internal development examples | Depends on supplied sources | `tools/t05_experiments.py` generated variants | Inherits source constraints; operator must verify | Generated variants |
| Official eval / heldout | Pending | Instructor private overlay | Pending instructor manifest | Unknown to student repo |

Public input schema is `image_id,path,camera_id,captured_at`. The manifest covers
each image plus `images.csv` and declares source, license, and SHA-256. Strict
provenance verifies coverage, root containment, decodability, and hashes.

```powershell
.\.venv\Scripts\python.exe -m tools.verify_manifests fixtures\t05_near_duplicate
# OK fixtures\t05_near_duplicate\manifest.csv
# 1 manifests, 0 failing
```

Seed is 42. Baseline assigns images independently. PoC assigns complete split
groups and never uses `scene_id` or `dup_group_id` at runtime. No real dataset,
external model, network download, or GT file was used.

## 03 · Research and design

```text
images + table + manifest + frozen config
  → validate provenance
  → SHA-256, pHash, dHash, histogram, thumbnail
  → canonical all-pairs evidence and duplicate classification
  → conservative duplicate groups + separate scene/burst constraints
  → deterministic whole-group train/test split
  → uncertainty × consequence review queue
  → validated relation override + append-only audit
  → rebuild groups/split → validate → GO/HOLD/REWORK
```

Design choices:

1. SHA-256 detects exact byte identity without claiming visual equivalence.
2. pHash/dHash and thumbnail evidence provide complementary structural signals.
   RGB histogram contributes only 0.10 and cannot independently classify a
   duplicate because similar colors do not imply the same scene.
3. Complete-link guarded grouping prevents one weak pair from joining otherwise
   incompatible duplicate components. Pair output expands every accepted group to
   all within-group pairs, matching evaluator semantics.
4. Same-camera/time metadata can add a split constraint but never prove duplicate
   content. Burst span is bounded to prevent an unbounded transitive chain.
5. Whole-group splitting trades exact 30% balance for leakage prevention.
6. Review prioritizes uncertainty and downstream merge/split impact. Reviewed
   relations trigger a complete rebuild.

No embedding model is bundled. This avoids undocumented weights/downloads, but
limits robustness to crop, viewpoint, occlusion, and domain changes.

Frozen policy is in `configs/tracks/t05.yaml`; metric semantics are in
`reports/metric_contract.md`; schemas and APIs are in
`docs/T05_IMPLEMENTATION.md`.

## 04 · Baseline versus PoC

**Baseline:** exact SHA-256 equality over the fixed all-pairs pool, then an
independent seeded image split. It demonstrates why exact-only detection and
random splitting are insufficient and returns HOLD.

**PoC:** combined visual evidence, conservative grouping, bounded scene
constraints, whole-group split, impact-aware review, audit, and structural release
policy. Both use the same 120 public images, 7,140 candidate pairs, seed 42, and
output pair schema. Review budget is 20 for the PoC.

### Official Table 1 — pending instructor GT

| Evaluator metric | Baseline | PoC | Status |
|---|---:|---:|---|
| Leakage Detection F1 (pair) | Pending | Pending | GT overlay absent |
| Pair precision / recall | Pending | Pending | GT overlay absent |
| False merge rate | Pending | Pending | Requires instructor `scene_id` |
| Residual cross-split leak (near-dup) | Pending | Pending | Requires `dup_group_id` |
| Residual cross-split leak (scene) | Pending | Pending | Requires `scene_id` |
| % data kept | Pending | Pending | Generate with unchanged evaluator and GT |

`reports/metrics_baseline.json`, `reports/metrics_poc.json`, and
`reports/table1_baseline_vs_poc.md` are deliberately absent. Generate them by
scripts only after GT installation; no placeholder values are supplied.

### Observable diagnostics — not official metrics

**Historical v1 table.** A subsequent GT-free development revision is summarized
below; do not present these older runs as current v2 output.

These values come from generated `diagnostics.json` and `run_info.json` files.

| Diagnostic | Baseline | Automatic PoC | Reviewed mechanism rerun |
|---|---:|---:|---:|
| Images / candidate rows | 120 / 7,140 | 120 / 7,140 | 120 / 7,140 |
| Predicted positive pairs | 20 exact | 115 | 115 |
| Train / test / drop | 84 / 36 / 0 | 84 / 36 / 0 | 83 / 37 / 0 |
| Observable positive cross-split pairs | 7 exact | 0 predicted | 0 predicted |
| Duplicate / split groups | Not applicable | 62 / 41 | 62 / 40 |
| Retention fraction | 1.0 | 1.0 | 1.0 |
| Runtime on validation machine | 2.953 s | 8.313 s | 8.1178 s |
| Structural decision | HOLD | GO | GO |

The reviewed row is a mechanism test attributed to
`task14-validation-bot-not-human`. It is not genuine human review and does not
establish pair correctness.

### GT-free quality mitigation — protocol v2

At the user's request, v2 adds bounded centered-crop RGB verification with a
single constrained exposure gain, a flat-image abstention gate, and stratified
review of final group-derived matches. Original hash/score thresholds and review
budget remain unchanged; the revised development-only protocol is declared in
`reports/metric_contract.md` before any official evaluation.

On the exposed synthetic regression set, TP/FP/FN changed from 24/17/16 to
40/0/0. On two independently generated synthetic confirmation sets, each changed
from 36/0/24 to 60/0/0. Both fresh sets use different source layouts/dimensions,
7% crops, JPEG quality 55, and brightness factor 0.9. These small labelled toy
results are **not official metrics** or guarantees for unseen real scenes.
Isolating review ranking on unchanged v1 predictions, error coverage at budget 20
improved from 0/33 to 5/33—still incomplete.

Fresh public v2 runs produce 128 positive predictions, 59 duplicate groups, 33
split groups, 84/36/0, retention 1.0, and zero observable crossings. Prediction
correctness and true scene leakage are still unknown. A non-human review mechanism
test changed four final flags and appended audit evidence. Eight deterministic
artifacts matched byte-for-byte across fresh v2 runs.

Full evidence and limitations: `DOC/T05_QUALITY_GAP_MITIGATION.md`. Use fresh
`out/poc_v2/`; prior v1 reviewed exports are stale and must not be relabelled or
have their fingerprints edited.

## 05 · Evaluation protocol

Primary metric is **Leakage Detection F1 (pair)** over canonical unordered image
pairs. GT positives are all within-`dup_group_id` pairs. Predictions with explicit
`pred_dup` use only rows equal to 1. Matching is exact normalized pair identity:
no IoU or tolerance. Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and F1 is
their harmonic mean. Edge cases and denominators are frozen in
`reports/metric_contract.md` and the unchanged evaluator.

Thresholds were frozen before official evaluation: duplicate score 0.75; pHash
and dHash Hamming limits 8; histogram L1 limit 0.08; thumbnail MAE limit 0.10;
at least two structural/visual signals; review band 0.55–0.85; budget 20; target
test fraction 0.30; seed 42.

Internal uncertainty uses source/scene-cluster bootstrap instead of treating
correlated pairs as independent. Official output must report point values with N,
numerator, and denominator.

```powershell
# Run only after instructor GT overlay is installed
.\.venv\Scripts\python.exe -m src.evaluate --track t05 --pred out\baseline\suspect_pairs.csv --run-name baseline --out reports\metrics_baseline.json
.\.venv\Scripts\python.exe -m src.evaluate --track t05 --pred out\poc_v2\suspect_pairs.csv --run-name poc --out reports\metrics_poc.json
.\.venv\Scripts\python.exe -m src.compare reports\metrics_baseline.json reports\metrics_poc.json --out reports\table1_baseline_vs_poc.md
```

Thresholds must not change after viewing official results. If baseline F1 is zero,
state `baseline = 0` rather than a percentage gain.

## 06 · Slices, failure cases, and compute

Official slices are pending GT. Required slices are exact, recompression,
brightness, crop, same-scene nonduplicate, and visually similar different-scene.
Instructor `transform` labels must not be exposed to production.

Two failure cases:

1. **Histogram/color false merge.** Unrelated scenes may have near-identical color
   distributions. Mitigation: 0.10 weight, minimum two non-histogram signals,
   hard-negative slice, and review priority.
2. **Weak transitive bridge and random split leakage.** One uncertain edge can
   over-merge components, while baseline split 7 of its 20 exact positive pairs.
   Mitigation: complete-link grouping, explicit cannot-links, bounded burst span,
   whole-group split, and crossing validators.

Validation machine: Windows 11 `10.0.26300`, Python 3.12.0, Intel64 Family 6 Model
165 Stepping 2. Automatic PoC runtime was 8.313 seconds, approximately 69.275 ms
per image or 1.164 ms per candidate pair. These are single-run diagnostics, not a
benchmark with variance.

## 07 · Human review, audit, and decision

`out/poc/review_queue.csv` contains at most 20 canonical pairs and immutable
evidence. `contact_sheet.html` embeds images for offline dual-image review. Queue
priority uses uncertainty, component sizes, bridge risk, and split impact.
`review_context.json` binds review input to image bytes, config, and queue identity.

Review relations are explicit: approve means `duplicate`; reject requires
`same_scene_only` or `different_scene`; edit assigns a relation; escalate/skip
remains unresolved and causes HOLD. Accepted decisions append actor, before/after
state, reason, and time to `audit_log.jsonl`. Automatic pairs stay preserved and
all dependent groups, splits, diagnostics, and decisions are rebuilt.

Task 14 validated the mechanism with one explicit non-human rejection to
`same_scene_only`. It produced one audit event, one changed constraint, 40 split
groups, an 83/37/0 split, full retention, and structural GO. This proves plumbing
and audit behavior only. A named human reviewer must independently review/export
the queue before submission sign-off.

New destination decision owner is `Nguyen Hai Dang`; the source checkout's
historical runs used `Dang Quoc Thai`. `POLICY_CHECKS_PASSED` means declared
observable checks passed; it does not claim verified zero true leakage.

## 08 · Reproduction and tests

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m tools.verify_manifests fixtures\t05_near_duplicate

# Baseline
.\.venv\Scripts\python.exe -m src.main --track t05 --pipeline src.t05.baseline --output out\baseline --run-name baseline --offline --seed 42

# PoC; review contact_sheet.html, save review_queue_reviewed.csv, then rerun
.\.venv\Scripts\python.exe -m src.main --track t05 --output out\poc_v2 --run-name poc --offline --seed 42

# Tests
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q
```

Focused T05 Tasks 1–13 validation: **159 passed**. Task 14 full suite:
**206 passed, 39 skipped**. Quality-mitigation validation adds 11 regression and
synthetic confirmation tests (full result in its validation record). Task 14 record:
`DOC/TASK_14_DOCUMENTATION_AND_SUBMISSION_VALIDATION.md`.

The source tree was uncommitted during validation. `b0ce8bf` in `run_info.json` is
the upstream base, not implementation provenance. Repeat or clearly link the final
run after creating the real implementation commit.

## 09 · Limitations, privacy, and ethics

- Official pair quality and true-scene leakage remain unknown without GT.
- Thresholds may not transfer across cameras, compression, or domains.
- pHash/dHash and thumbnails are weak under heavy crop, perspective, occlusion,
  overlays, and adversarial manipulation.
- Camera/time metadata can be absent, wrong, or duplicated across devices.
- All-pairs evidence is quadratic; 10× images means about 100× pairs.
- No local embedding model is documented or evaluated.
- False merges reduce utility; misses can inflate evaluation through leakage.
- Public fixtures are synthetic CC0-1.0. Real images may contain faces, plates,
  locations, or timestamps and require purpose limitation, access control,
  retention policy, and verified licensing.
- Structural GO is not GDPR, Vietnamese-law, security, or ethical certification.
  DPO/legal/data-owner approval remains external.

Priorities are candidate retrieval for scale, documented crop-resistant features,
calibration on separated development data, human review, and heldout evaluation.

## 10 · Team contribution evidence

| Member | Git author/email | Role and files | Commit evidence | Presenter |
|---|---|---|---|---|
| Nguyen Hai Dang | Current local config: `44244432664 <haidang453@gmail.com>`; not a retroactive authorship claim | Confirmed destination owner; imported working-tree scope listed in `reports/team_contributions.csv` | **Pending** — no project commit by this identity; changes uncommitted | Pending |
| Other members | Not supplied | Not inferred | Not fabricated | Pending roster |

`git shortlog -sne --all` lists only upstream authors `loversky02` and `Tin Duong`;
those commits are not claimed as team T05 contributions. Working-tree paths show
scope, not authorship. Before submission, real contributors must commit their own
work with verified identities and update both contribution tables.

## 11 · References

| Reference | Accessed | Relevance |
|---|---|---|
| scikit-learn, [Common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | 2026-11-15 | Motivates strict train/test separation; T05 extends it to related-image groups. |
| Christoph Zauner, [Perceptual Image Hash Functions](https://www.phash.org/docs/pubs/thesis_zauner.pdf) | 2026-11-15 | Background for perceptual hashes and empirically selected distance thresholds. |
| Creative Commons, [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) | 2026-11-15 | License identifier declared by the synthetic fixture manifest. |

## 12 · Submission checklist

- [x] Offline baseline and PoC run with seed 42.
- [x] Public provenance and license documented and verified.
- [x] Prediction, split, evidence, group, review, diagnostic, and decision artifacts
  generated and structurally validated.
- [x] Failure cases, limitations, schemas, policy, and audit behavior documented.
- [x] Mechanism check changed a relation and rebuilt dependent artifacts.
- [ ] Genuine human review/sign-off received.
- [ ] Instructor GT installed; official metrics/comparison generated by scripts.
- [ ] Real team roster and individual Git contributions confirmed.
- [ ] T05 implementation committed with verified author identity.
- [ ] Required tags created at approved commits.

Unchecked items are explicit submission blockers.
