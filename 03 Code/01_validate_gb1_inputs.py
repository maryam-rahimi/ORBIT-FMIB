"""
ORBIT-FMIB Code 01
==================

Audit the existing GB1 artifacts and create a label-isolated sequence index
for downstream frozen protein-foundation-model inference.

Important biological distinction
---------------------------------
The GB1 benchmark names the four assayed residues by GB1-domain numbering:
39, 40, 41, and 54.

The FLIP-derived standardized CSV used in this project stores a longer
265-residue protein sequence. Therefore, domain positions 39/40/41/54 must
NOT be assumed to be raw string indices 38/39/40/53.

This script discovers the four variable positions in the stored 265-aa
sequence directly from the data, verifies that their WT residues are VDGV,
and verifies that extracting those positions reproduces every four-letter
variant code.

Inputs
------
02 Data/gb1_full_standardized.csv
02 Data/gb1_flip_two_vs_rest_frozen.csv

Outputs
-------
02 Data/PFM_Inputs/gb1_pfm_index.csv
02 Data/PFM_Inputs/gb1_pfm_index_manifest.json

The PFM index intentionally excludes fitness labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Frozen configuration
# ============================================================

WT_VARIANT = "VDGV"

# GB1-domain biological numbering from the experimental landscape.
GB1_DOMAIN_POSITIONS_1BASED = (39, 40, 41, 54)

# The FLIP-derived standardized artifact in this project stores a
# 265-residue sequence, not the isolated 56-aa GB1 domain.
EXPECTED_STORED_SEQUENCE_LENGTH = 265

EXPECTED_MEASURED_VARIANTS = 149_361

EXPECTED_FLIP_TRAIN = 424
EXPECTED_FLIP_TEST = 8_309
EXPECTED_FLIP_TOTAL = 8_733

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

EXPECTED_FROZEN_SHA256 = (
    "47ffb916c91a6796cbe00b9cb9be2e11"
    "f1f9f3e143606c793cc2a543a3d5c46e"
)


# ============================================================
# Paths
# ============================================================

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "02 Data"

FULL_PATH = DATA_DIR / "gb1_full_standardized.csv"
FROZEN_PATH = DATA_DIR / "gb1_flip_two_vs_rest_frozen.csv"

OUTPUT_DIR = DATA_DIR / "PFM_Inputs"
INDEX_PATH = OUTPUT_DIR / "gb1_pfm_index.csv"
MANIFEST_PATH = OUTPUT_DIR / "gb1_pfm_index_manifest.json"


# ============================================================
# Utilities
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(dataframe.columns)
    if missing:
        raise RuntimeError(
            f"{name} is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def strings_to_byte_matrix(
    values: pd.Series,
    expected_length: int,
    name: str,
) -> np.ndarray:
    values = values.astype(str)
    lengths = values.str.len().to_numpy()

    if not np.all(lengths == expected_length):
        unique, counts = np.unique(lengths, return_counts=True)
        observed = {int(k): int(v) for k, v in zip(unique, counts)}
        raise RuntimeError(
            f"{name} has unexpected string lengths.\n"
            f"Expected every row to have length {expected_length}.\n"
            f"Observed: {observed}"
        )

    try:
        blob = "".join(values.tolist()).encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"{name} contains non-ASCII characters.") from exc

    arr = np.frombuffer(blob, dtype="S1")
    expected_size = len(values) * expected_length

    if arr.size != expected_size:
        raise RuntimeError(
            f"Internal byte-matrix size mismatch for {name}: "
            f"expected {expected_size}, observed {arr.size}."
        )

    return arr.reshape(len(values), expected_length)


def validate_amino_acid_strings(
    values: pd.Series,
    name: str,
) -> None:
    observed = set("".join(values.astype(str).tolist()))
    unexpected = observed - AMINO_ACIDS

    if unexpected:
        raise RuntimeError(
            f"{name} contains non-canonical amino-acid symbols: "
            f"{sorted(unexpected)}"
        )


def decode_row(row: np.ndarray) -> str:
    return b"".join(row.tolist()).decode("ascii")


# ============================================================
# Full GB1 audit
# ============================================================

def load_full() -> pd.DataFrame:
    if not FULL_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical GB1 data:\n{FULL_PATH}"
        )

    df = pd.read_csv(FULL_PATH, low_memory=False)

    require_columns(
        df,
        {"variant", "sequence", "hd", "fitness"},
        "gb1_full_standardized.csv",
    )

    return df


def audit_full(df: pd.DataFrame) -> dict:
    # --------------------------------------------------------
    # Basic table integrity
    # --------------------------------------------------------
    if len(df) != EXPECTED_MEASURED_VARIANTS:
        raise RuntimeError(
            "Unexpected number of measured GB1 variants.\n"
            f"Expected: {EXPECTED_MEASURED_VARIANTS:,}\n"
            f"Observed: {len(df):,}"
        )

    for col in ["variant", "sequence", "hd", "fitness"]:
        if df[col].isna().any():
            raise RuntimeError(f"Missing values detected in column: {col}")

    if not np.isfinite(
        pd.to_numeric(df["fitness"], errors="coerce").to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("Non-finite/nonnumeric fitness values detected.")

    if df["variant"].duplicated().any():
        raise RuntimeError("Duplicate four-site variants detected.")

    if df["sequence"].duplicated().any():
        raise RuntimeError("Duplicate full protein sequences detected.")

    variant = df["variant"].astype(str)
    sequence = df["sequence"].astype(str)

    if not (variant.str.len() == 4).all():
        raise RuntimeError("Every GB1 variant code must have length 4.")

    if not (variant == variant.str.upper()).all():
        raise RuntimeError("GB1 variant strings must be uppercase.")

    if not (sequence == sequence.str.upper()).all():
        raise RuntimeError("Stored protein sequences must be uppercase.")

    validate_amino_acid_strings(variant, "GB1 four-site variants")
    validate_amino_acid_strings(sequence, "GB1 stored protein sequences")

    # --------------------------------------------------------
    # WT and stored sequence length
    # --------------------------------------------------------
    wt_rows = df.loc[variant == WT_VARIANT]

    if len(wt_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one WT variant {WT_VARIANT}; "
            f"found {len(wt_rows)}."
        )

    wt_sequence = str(wt_rows.iloc[0]["sequence"])

    if len(wt_sequence) != EXPECTED_STORED_SEQUENCE_LENGTH:
        raise RuntimeError(
            "Unexpected stored WT sequence length.\n"
            f"Expected: {EXPECTED_STORED_SEQUENCE_LENGTH}\n"
            f"Observed: {len(wt_sequence)}"
        )

    sequence_matrix = strings_to_byte_matrix(
        sequence,
        EXPECTED_STORED_SEQUENCE_LENGTH,
        "GB1 stored protein sequences",
    )

    variant_matrix = strings_to_byte_matrix(
        variant,
        4,
        "GB1 four-site variants",
    )

    wt_bytes = np.frombuffer(
        wt_sequence.encode("ascii"),
        dtype="S1",
    )

    # --------------------------------------------------------
    # Discover variable positions in the STORED sequence.
    # Do not assume domain numbering equals string indexing.
    # --------------------------------------------------------
    differs_from_wt = sequence_matrix != wt_bytes[None, :]
    variable_positions_0based = tuple(
        int(i)
        for i in np.flatnonzero(np.any(differs_from_wt, axis=0))
    )

    if len(variable_positions_0based) != 4:
        raise RuntimeError(
            "Expected exactly four variable positions in the stored "
            "protein sequence.\n"
            f"Observed zero-based positions: {variable_positions_0based}\n"
            f"Observed one-based positions: "
            f"{tuple(i + 1 for i in variable_positions_0based)}"
        )

    stored_variable_positions_1based = tuple(
        i + 1 for i in variable_positions_0based
    )

    wt_at_variable_positions = "".join(
        wt_sequence[i] for i in variable_positions_0based
    )

    if wt_at_variable_positions != WT_VARIANT:
        raise RuntimeError(
            "WT residues at the four discovered stored-sequence variable "
            "positions do not equal VDGV.\n"
            f"Discovered positions (1-based): "
            f"{stored_variable_positions_1based}\n"
            f"Observed WT residues: {wt_at_variable_positions}"
        )

    # --------------------------------------------------------
    # Every stored sequence must reproduce its four-aa variant
    # at the four discovered variable positions.
    # --------------------------------------------------------
    extracted = sequence_matrix[:, list(variable_positions_0based)]

    if not np.array_equal(extracted, variant_matrix):
        mismatch_rows = np.where(
            np.any(extracted != variant_matrix, axis=1)
        )[0]

        examples = []
        for idx in mismatch_rows[:10]:
            examples.append(
                {
                    "row": int(idx),
                    "variant": variant.iloc[idx],
                    "stored_sequence_extracted": decode_row(extracted[idx]),
                }
            )

        raise RuntimeError(
            "Stored-sequence / four-site variant mapping FAILED.\n"
            f"Discovered stored positions (1-based): "
            f"{stored_variable_positions_1based}\n"
            f"First mismatches: {examples}"
        )

    # --------------------------------------------------------
    # Hamming-distance cross-check in both representations.
    # --------------------------------------------------------
    wt_variant_bytes = np.frombuffer(
        WT_VARIANT.encode("ascii"),
        dtype="S1",
    )

    variant_hd = np.sum(
        variant_matrix != wt_variant_bytes[None, :],
        axis=1,
    )

    sequence_hd = np.sum(
        differs_from_wt,
        axis=1,
    )

    observed_hd = pd.to_numeric(
        df["hd"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(observed_hd).all():
        raise RuntimeError("GB1 hd contains missing/nonnumeric values.")

    if not np.array_equal(
        observed_hd.astype(np.int64),
        variant_hd.astype(np.int64),
    ):
        raise RuntimeError(
            "Stored hd does not equal Hamming distance of the "
            "four-site variant from VDGV."
        )

    if not np.array_equal(sequence_hd, variant_hd):
        raise RuntimeError(
            "Stored full-sequence Hamming distance from WT does not equal "
            "four-site Hamming distance. This indicates variation outside "
            "the four discovered assay positions."
        )

    hd_counts = {
        int(k): int(v)
        for k, v in zip(*np.unique(variant_hd, return_counts=True))
    }

    domain_to_stored_position_1based = {
        str(domain_pos): int(stored_pos)
        for domain_pos, stored_pos in zip(
            GB1_DOMAIN_POSITIONS_1BASED,
            stored_variable_positions_1based,
        )
    }

    domain_to_stored_index_0based = {
        str(domain_pos): int(stored_pos - 1)
        for domain_pos, stored_pos in zip(
            GB1_DOMAIN_POSITIONS_1BASED,
            stored_variable_positions_1based,
        )
    }

    return {
        "wt_sequence": wt_sequence,
        "stored_sequence_length": EXPECTED_STORED_SEQUENCE_LENGTH,
        "stored_variable_positions_1based": stored_variable_positions_1based,
        "stored_variable_positions_0based": variable_positions_0based,
        "domain_to_stored_position_1based": domain_to_stored_position_1based,
        "domain_to_stored_index_0based": domain_to_stored_index_0based,
        "hd_counts": hd_counts,
    }


# ============================================================
# Frozen FLIP audit
# ============================================================

def load_frozen() -> pd.DataFrame:
    if not FROZEN_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen FLIP split:\n{FROZEN_PATH}"
        )

    observed_hash = sha256_file(FROZEN_PATH)

    if observed_hash != EXPECTED_FROZEN_SHA256:
        raise RuntimeError(
            "Frozen FLIP split hash mismatch.\n"
            f"Expected: {EXPECTED_FROZEN_SHA256}\n"
            f"Observed: {observed_hash}"
        )

    df = pd.read_csv(FROZEN_PATH, low_memory=False)

    require_columns(
        df,
        {"sequence", "variant", "hd", "fitness", "set"},
        "gb1_flip_two_vs_rest_frozen.csv",
    )

    return df


def audit_frozen(
    frozen: pd.DataFrame,
    full: pd.DataFrame,
) -> dict:
    if len(frozen) != EXPECTED_FLIP_TOTAL:
        raise RuntimeError(
            "Unexpected FLIP split size.\n"
            f"Expected: {EXPECTED_FLIP_TOTAL:,}\n"
            f"Observed: {len(frozen):,}"
        )

    if frozen["sequence"].duplicated().any():
        raise RuntimeError("Frozen FLIP split contains duplicate sequences.")

    if frozen["variant"].duplicated().any():
        raise RuntimeError("Frozen FLIP split contains duplicate variants.")

    normalized_set = (
        frozen["set"].astype(str).str.strip().str.lower()
    )

    observed_sets = set(normalized_set.unique())

    if observed_sets != {"train", "test"}:
        raise RuntimeError(
            f"Unexpected FLIP set labels: {sorted(observed_sets)}"
        )

    train = frozen.loc[normalized_set == "train"]
    test = frozen.loc[normalized_set == "test"]

    if len(train) != EXPECTED_FLIP_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_FLIP_TRAIN} FLIP train rows; "
            f"found {len(train)}."
        )

    if len(test) != EXPECTED_FLIP_TEST:
        raise RuntimeError(
            f"Expected {EXPECTED_FLIP_TEST:,} FLIP test rows; "
            f"found {len(test):,}."
        )

    if not (pd.to_numeric(train["hd"]) <= 2).all():
        raise RuntimeError("FLIP train contains HD > 2 variants.")

    if not (pd.to_numeric(test["hd"]) > 2).all():
        raise RuntimeError("FLIP test contains HD <= 2 variants.")

    concordance = frozen[
        ["sequence", "variant", "hd", "fitness", "set"]
    ].merge(
        full[["sequence", "variant", "hd", "fitness"]],
        on=["sequence", "variant"],
        how="left",
        validate="one_to_one",
        suffixes=("_frozen", "_full"),
        indicator=True,
    )

    if not (concordance["_merge"] == "both").all():
        bad = concordance.loc[
            concordance["_merge"] != "both",
            ["sequence", "variant", "_merge"],
        ].head(10)

        raise RuntimeError(
            "Frozen FLIP rows do not map one-to-one to full GB1 data.\n"
            f"Examples:\n{bad.to_string(index=False)}"
        )

    if not np.allclose(
        concordance["fitness_frozen"].to_numpy(dtype=np.float64),
        concordance["fitness_full"].to_numpy(dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Frozen FLIP fitness values differ from full GB1 fitness."
        )

    if not np.array_equal(
        concordance["hd_frozen"].to_numpy(dtype=np.int64),
        concordance["hd_full"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError(
            "Frozen FLIP Hamming distances differ from full GB1 data."
        )

    return {
        "train": int(len(train)),
        "test": int(len(test)),
        "total": int(len(frozen)),
    }


# ============================================================
# Label-isolated PFM index
# ============================================================

def build_pfm_index(
    full: pd.DataFrame,
    frozen: pd.DataFrame,
) -> pd.DataFrame:
    # Fitness is intentionally excluded from the downstream PFM index.
    result = full[
        ["variant", "sequence", "hd"]
    ].copy()

    split_lookup = frozen[
        ["sequence", "variant", "set"]
    ].copy()

    result = result.merge(
        split_lookup,
        on=["sequence", "variant"],
        how="left",
        validate="one_to_one",
    )

    result["flip_set"] = result["set"].fillna("other")
    result = result.drop(columns=["set"])

    result.insert(
        0,
        "pfm_row_id",
        np.arange(len(result), dtype=np.int64),
    )

    result["hd"] = pd.to_numeric(
        result["hd"],
        errors="raise",
    ).astype(np.int8)

    counts = result["flip_set"].value_counts().to_dict()

    expected_counts = {
        "train": EXPECTED_FLIP_TRAIN,
        "test": EXPECTED_FLIP_TEST,
        "other": (
            EXPECTED_MEASURED_VARIANTS
            - EXPECTED_FLIP_TRAIN
            - EXPECTED_FLIP_TEST
        ),
    }

    if counts != expected_counts:
        raise RuntimeError(
            "Unexpected PFM-index FLIP membership counts.\n"
            f"Expected: {expected_counts}\n"
            f"Observed: {counts}"
        )

    return result


# ============================================================
# Output
# ============================================================

def write_outputs(
    pfm_index: pd.DataFrame,
    full_audit: dict,
    frozen_audit: dict,
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pfm_index.to_csv(INDEX_PATH, index=False)

    manifest = {
        "status": "PASS",
        "purpose": (
            "Label-isolated GB1 sequence index for frozen PFM inference"
        ),
        "source_full_path": str(FULL_PATH),
        "source_full_sha256": sha256_file(FULL_PATH),
        "source_frozen_path": str(FROZEN_PATH),
        "source_frozen_sha256": sha256_file(FROZEN_PATH),
        "output_index_path": str(INDEX_PATH),
        "output_index_sha256": sha256_file(INDEX_PATH),
        "n_variants": int(len(pfm_index)),
        "stored_sequence_length": int(
            full_audit["stored_sequence_length"]
        ),
        "gb1_domain_positions_1based": list(
            GB1_DOMAIN_POSITIONS_1BASED
        ),
        "stored_variable_positions_1based": list(
            full_audit["stored_variable_positions_1based"]
        ),
        "stored_variable_positions_0based": list(
            full_audit["stored_variable_positions_0based"]
        ),
        "domain_to_stored_position_1based": (
            full_audit["domain_to_stored_position_1based"]
        ),
        "domain_to_stored_index_0based": (
            full_audit["domain_to_stored_index_0based"]
        ),
        "wt_variant": WT_VARIANT,
        "wt_sequence": full_audit["wt_sequence"],
        "hamming_distance_counts": {
            str(k): int(v)
            for k, v in full_audit["hd_counts"].items()
        },
        "flip_counts": frozen_audit,
        "invariants": [
            "FLIP-derived standardized sequence is 265 aa",
            "exactly four stored-sequence positions vary across the landscape",
            "WT residues at the four discovered positions are VDGV",
            "the four discovered positions reproduce every 4-aa variant code",
            "all other stored-sequence positions are invariant relative to WT",
            "stored HD equals independently recomputed 4-site and full-sequence HD",
            "frozen FLIP split hash matches the pinned artifact",
            "PFM input index excludes fitness labels",
            "no model training occurs",
        ],
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

    return INDEX_PATH, MANIFEST_PATH


# ============================================================
# Self-test
# ============================================================

def run_self_test() -> None:
    # Synthetic stored sequence length matches the real artifact.
    length = EXPECTED_STORED_SEQUENCE_LENGTH

    # Deliberately choose full-sequence storage positions that are NOT
    # equal to GB1 domain numbers. This tests the discovery logic.
    synthetic_variable_0based = (100, 101, 102, 115)

    chars = ["A"] * length
    for idx, aa in zip(synthetic_variable_0based, WT_VARIANT):
        chars[idx] = aa

    wt_sequence = "".join(chars)

    variants = [
        "VDGV",
        "ADGV",
        "VNGV",
        "VDAV",
        "VDGI",
        "ACWY",
    ]

    sequences = []

    for var in variants:
        seq_chars = list(wt_sequence)
        for idx, aa in zip(synthetic_variable_0based, var):
            seq_chars[idx] = aa
        sequences.append("".join(seq_chars))

    matrix = strings_to_byte_matrix(
        pd.Series(sequences),
        length,
        "self-test sequences",
    )

    wt_bytes = np.frombuffer(
        sequences[0].encode("ascii"),
        dtype="S1",
    )

    discovered = tuple(
        int(i)
        for i in np.flatnonzero(
            np.any(matrix != wt_bytes[None, :], axis=0)
        )
    )

    assert discovered == synthetic_variable_0based

    extracted = matrix[:, list(discovered)]
    decoded = [decode_row(row) for row in extracted]

    assert decoded == variants
    assert "".join(sequences[0][i] for i in discovered) == WT_VARIANT

    print("SELF-TEST: PASS")
    print(
        "Verified: stored-sequence mutation positions are discovered "
        "from the data rather than assumed from GB1-domain numbering."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal mapping/discovery tests and exit.",
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run all real-data audits without writing outputs.",
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    print("=" * 76)
    print("ORBIT-FMIB CODE 01 — GB1 INPUT AUDIT")
    print("=" * 76)
    print()

    full = load_full()
    full_audit = audit_full(full)

    print("Full GB1 audit: PASS")
    print(f"  variants: {len(full):,}")
    print(
        f"  stored sequence length: "
        f"{full_audit['stored_sequence_length']}"
    )
    print(
        "  GB1-domain -> stored-sequence mapping:"
    )
    for domain_pos in GB1_DOMAIN_POSITIONS_1BASED:
        stored_pos = full_audit[
            "domain_to_stored_position_1based"
        ][str(domain_pos)]
        stored_idx = full_audit[
            "domain_to_stored_index_0based"
        ][str(domain_pos)]

        print(
            f"    domain {domain_pos} -> "
            f"stored position {stored_pos} "
            f"(Python index {stored_idx})"
        )

    print("  four-site variant mapping: PASS")
    print("  invariant non-assayed residues: PASS")
    print("  Hamming-distance cross-check: PASS")
    print()

    frozen = load_frozen()
    frozen_audit = audit_frozen(frozen, full)

    print("Frozen FLIP provenance audit: PASS")
    print(f"  train: {frozen_audit['train']:,}")
    print(f"  test: {frozen_audit['test']:,}")
    print(f"  SHA256: {sha256_file(FROZEN_PATH)}")
    print()

    pfm_index = build_pfm_index(full, frozen)

    print("Label-isolated PFM sequence-index construction: PASS")
    print(
        pfm_index["flip_set"]
        .value_counts()
        .to_string()
    )
    print("  fitness column present in PFM index: NO")
    print()

    if args.no_write:
        print("NO-WRITE MODE: audits completed; no files written.")
    else:
        index_path, manifest_path = write_outputs(
            pfm_index,
            full_audit,
            frozen_audit,
        )

        print("Saved:")
        print(f"  {index_path}")
        print(f"  {manifest_path}")

    print()
    print("ORBIT-FMIB CODE 01: PASS")


if __name__ == "__main__":
    main()
