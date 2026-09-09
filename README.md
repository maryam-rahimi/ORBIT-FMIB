# ORBIT-FMIB: Tracing Order-Resolved Epistatic Information Through ESM-2

This repository contains the anonymized ESM-2 analysis code and lightweight reproducibility artifacts for ORBIT-FMIB. The release is intentionally scoped to the model reported in the manuscript: **ESM-2**.

## Scientific scope

ORBIT-FMIB traces experimentally aligned first- through fourth-order epistatic information through the 31 representation stages (L0--L30) of frozen `esm2_t30_150M_UR50D`. The biological analysis uses the dense four-site GB1 landscape and fixed positional subsets. Neural DV/MINE scores are used comparatively and are not interpreted as calibrated absolute mutual information in nats.

## Repository layout

```text
02 Data/                  data-access instructions; raw GB1 inputs are not redistributed here
03 Code/                  audited ESM-2 pipeline and validation scripts
04 Experiments/           lightweight validation and production outputs
05 Figure/                estimator-validation figure and provenance
05 Results/Overleaf/      paper-facing tables/figures/macros, original and replication-augmented
figures/manuscript/       manuscript-facing figures
slurm/                    generic SLURM templates
LICENSE                   MIT license
CODE_DESCRIPTION.md       script-by-script description
DATA_ACCESS.md            required external data / large-artifact notes
GITHUB_UPLOAD_STEPS.md    step-by-step GitHub instructions
CODE_HASHES.sha256        hashes of released Python sources
RESULT_HASHES.sha256      hashes of released non-code artifacts
REPLICATION_INPUT_seed_results.csv   raw per-cell independent GPU replication results
```

## Main biological execution path

1. Audit GB1 inputs and ESM-2 loading/indexing.
2. Validate the measurement pipeline on ORBIT-Synth and estimator stress tests.
3. Construct complete WT-anchored GB1 cubes.
4. Extract frozen ESM-2 site representations at L0--L30.
5. Build order-resolved Walsh samples.
6. Validate fixed-subset conditioning, null behavior, and the exact sample regime.
7. Run the authoritative fixed-subset biological dependence analysis.
8. Aggregate trajectories, compute peak-to-final retention, and generate layerwise figures.
9. **Independent replication check:** retrain the full grid on an independent
   GPU run to produce a raw per-cell CSV (see
   `REPLICATION_INPUT_seed_results.csv` for the exact schema), place it at
   the repository root under that filename, then run
   `03 Code/15_export_overleaf_results_replication.py` and
   `03 Code/16_make_gb1_layerwise_figures_replication.py` in place of their
   non-replication counterparts to regenerate the paper's tables/figures
   with the replication comparison folded in. Both scripts fall back to
   original-only output if the replication CSV is absent.

See `CODE_DESCRIPTION.md` for the exact script mapping.

## Primary released result values

The corrected-IID production run gave peak-to-final retention of
approximately 0.960, 0.971, 0.910, and 0.808 for interaction orders 1--4,
with higher-minus-lower-order contrast `-0.107`; the whole-identity
deletion analysis excluding cubes with alternative identity `41=L`
preserved the same direction (`-0.184`). **An independent replication
under matched GPU hardware and identical critic seeds substantially
reduced this contrast to `-0.017`, and its sign was not stable across
otherwise-defensible evaluation-pairing choices applied to the same
trained critics (ranging `-0.011` to `+0.015`).** CUDA training is
non-deterministic by default in this pipeline, so this replication used
independently retrained critics rather than reproducing the original
checkpoints exactly (23% of fits matched the original best-validation
step; mean absolute score difference 0.081 nats). The paper does not
treat selective higher-order information loss in ESM-2 as an established
finding; see the manuscript's Results/Discussion and
Appendix "Independent Replication and Evaluation-Pairing Sensitivity"
for full numbers and interpretation. Machine-readable outputs for both
the original production run and the replication are under
`04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910/` and
`REPLICATION_INPUT_seed_results.csv` respectively; the paper-facing
tables/figures generated from both (via `03 Code/15_export_overleaf_results_replication.py`
and `03 Code/16_make_gb1_layerwise_figures_replication.py`) are under
`05 Results/Overleaf/`.

## Data and large files

Large frozen ESM-2 layer arrays and model checkpoints are not committed. The two GB1 input CSVs are also omitted until redistribution rights are confirmed. See `DATA_ACCESS.md`.

## Environment

The audited production environment used PyTorch 2.7.1 with CUDA 12.6. Core Python dependencies are listed in `requirements.txt`. ESM-2 weights are loaded through `fair-esm` and are not stored in the repository.

## Anonymity

This release originated as a review-package build with no author names, institutional affiliations, personal emails, cluster usernames, or absolute user home paths in the tracked source files, and institution-specific cluster wording replaced with generic HPC terminology. The target venue (MoML) does not use double-blind review, so this repository is linked directly from the manuscript under the authors' names; the anonymity scan and comment-only source changes are retained here as a provenance record, not as an active review-anonymity requirement.
