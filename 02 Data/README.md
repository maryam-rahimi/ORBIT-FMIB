# GB1 input data

The biological input files are intentionally not bundled here until redistribution rights are verified.
The audited pipeline expects:

```text
02 Data/gb1_full_standardized.csv
02 Data/gb1_flip_two_vs_rest_frozen.csv
```

Running `03 Code/01_validate_gb1_inputs.py` creates the label-free protein-model index under:

```text
02 Data/PFM_Inputs/
```

If the two input CSVs are distributed publicly, preserve their exact contents and record their SHA256 hashes. Otherwise provide the public source and preprocessing instructions needed to recreate them.
