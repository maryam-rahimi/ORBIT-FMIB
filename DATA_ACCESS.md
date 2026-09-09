# Data and large-artifact access

The GitHub package intentionally omits two categories of large or redistribution-sensitive files.

## GB1 input CSVs

The pipeline expects these files at the repository-relative paths:

```text
02 Data/gb1_full_standardized.csv
02 Data/gb1_flip_two_vs_rest_frozen.csv
```

These are derived from the publicly available GB1 combinatorial fitness
landscape dataset of Wu et al. (2016), *Adaptation in protein fitness
landscapes is facilitated by indirect paths*, eLife 5:e16965
(https://doi.org/10.7554/eLife.16965), as distributed via the FLIP
benchmark (Notin et al., 2023 -- ProteinGym; Dallago et al., FLIP:
Benchmark tasks in fitness landscape inference for proteins). Download
the GB1 landscape from either source and run the repository's Code 01
audit script to regenerate `gb1_full_standardized.csv`; the FLIP
two-vs-rest split used for the frozen sample regime is described in
`03 Code/07_build_gb1_strict_split.py` and
`03 Code/10_build_gb1_order_samples.py`. Copy the exact audited files
into the paths above before running Code 01.

## Frozen ESM-2 hidden-state arrays

Do **not** commit:

```text
04 Experiments/EXP02_GB1_ESM2/ESM2_Site_Reps/esm2_layer_*_sites.npy
```

They are large and reproducible. `03 Code/09_extract_esm2_representations.py` regenerates them from the label-free extraction index and the public ESM-2 checkpoint.

## Already included

The repository includes the lightweight validation artifacts needed to document the estimator checks, the corrected-IID fixed-subset result tables/manifests, the whole-identity `41=L` sensitivity outputs, the Code 15 layerwise summaries, the Code 16 figures, and the subset-conditioning validation figure/provenance files.
