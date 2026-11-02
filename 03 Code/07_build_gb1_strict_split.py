#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 07 — Build the AA-identity-strict GB1 estimator split.

Purpose
-------
Construct the leakage-safe train/validation/test cube partition used later by
ORBIT-FMIB mutual-information critics. The split is defined only from the
non-WT amino-acid identities chosen at GB1 positions 39, 40, 41, and 54.
Experimental fitness and ESM-2 representations are NOT used to create it.

Frozen split rule
-----------------
At each of the four assayed positions, the 19 non-WT amino acids are
reproducibly partitioned with NumPy RNG seed 42 into:

    10 identities -> estimator_train
     4 identities -> estimator_val
     5 identities -> estimator_test

A complete WT-anchored cube belongs to one strict partition only when all four
of its selected alternative amino acids belong to that same partition.
Otherwise it is labelled "mixed" and excluded from strict critic fitting and
evaluation.

This reproduces the previously audited ORBIT AA-identity-strict construction,
while adapting it to the Code-06 ORBIT-FMIB artifacts and provenance chain.

Leakage guarantee
-----------------
Because the non-WT amino-acid identities are disjoint at every assayed
position, estimator_train / estimator_val / estimator_test must share no
non-WT GB1 variants. WT = VDGV is necessarily shared because every cube is
WT-anchored. Code 07 verifies this directly both in four-site variant space
and with the Code-06 label-free PFM row indices.

Important boundary
------------------
This is the split for ORBIT-FMIB critic estimation. It does NOT replace the
official FLIP two_vs_rest model split and does NOT imply that test variants
were absent from ESM pretraining or from any upstream benchmark training set.

Default input
-------------
04 Experiments/EXP02_GB1_ESM2/gb1_cube_manifest.json

Default outputs
---------------
04 Experiments/EXP02_GB1_ESM2/
    gb1_strict_aa_assignment.csv
    gb1_strict_cube_split.csv
    gb1_strict_overlap_check.csv
    gb1_strict_variant_summary.csv
    gb1_strict_split_indices.npz
    gb1_strict_split_manifest.json

Usage
-----
    python "03 Code\\07_build_gb1_strict_split.py" --self-test
    python "03 Code\\07_build_gb1_strict_split.py"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Frozen Phase-1 contract
# =============================================================================

CODE_NAME = "ORBIT-FMIB CODE 07"
SCHEMA_VERSION = 1
EXPECTED_CODE06 = "ORBIT-FMIB CODE 06"

WT = "VDGV"
POSITIONS = (39, 40, 41, 54)
ALT_COLUMNS = ("alt_39", "alt_40", "alt_41", "alt_54")
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_SET = set(AMINO_ACIDS)

RANDOM_SEED = 42
N_TRAIN_AA = 10
N_VAL_AA = 4
N_TEST_AA = 5
EXPECTED_ALTERNATIVES_PER_POSITION = 19

EXPECTED_COMPLETE_CUBES = 109_235
EXPECTED_SPLIT_COUNTS = {
    "estimator_train": 7_352,
    "estimator_val": 245,
    "estimator_test": 613,
    "mixed": 101_025,
}
EXPECTED_STRICT_CUBES = 8_210

EXPECTED_VARIANT_SHAPE = (EXPECTED_COMPLETE_CUBES, 16)


# =============================================================================
# Generic helpers
# =============================================================================


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def variant_to_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, np.bytes_):
        return value.decode("ascii")
    return str(value)


def hamming_from_wt(variant: str) -> int:
    variant = variant_to_text(variant)
    if len(variant) != len(WT):
        raise RuntimeError(f"Invalid GB1 variant length: {variant!r}")
    return sum(a != b for a, b in zip(variant, WT))


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Path: {path}\nExpected: {expected}\nObserved: {observed}"
        )


# =============================================================================
# Code-06 provenance and artifact loading
# =============================================================================


def load_and_verify_code06(manifest_path: Path) -> Tuple[Dict, str, Dict[str, Path]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Code-06 manifest not found: {manifest_path}\n"
            "Run Code 06 successfully before Code 07."
        )

    manifest = load_json(manifest_path)
    if manifest.get("code_name") != EXPECTED_CODE06:
        raise RuntimeError(
            f"Unexpected Code-06 manifest code_name: {manifest.get('code_name')!r}"
        )
    if not bool(manifest.get("overall_pass", False)):
        raise RuntimeError("Code-06 manifest does not record overall_pass=true.")

    audit = manifest.get("audit", {})
    if int(audit.get("complete_cubes", -1)) != EXPECTED_COMPLETE_CUBES:
        raise RuntimeError(
            "Code-06 manifest complete-cube count is not the frozen GB1 value.\n"
            f"Expected: {EXPECTED_COMPLETE_CUBES:,}\n"
            f"Observed: {audit.get('complete_cubes')!r}"
        )

    outputs = manifest.get("outputs", {})
    hashes = manifest.get("file_sha256", {})
    required_keys = ("metadata", "variants", "fitness", "source_rows", "pfm_rows")

    paths: Dict[str, Path] = {}
    for key in required_keys:
        name = outputs.get(key)
        if not name:
            raise RuntimeError(f"Code-06 manifest is missing outputs.{key}.")
        path = manifest_path.parent / str(name)
        expected_hash = hashes.get(path.name)
        if not expected_hash:
            raise RuntimeError(f"Code-06 manifest is missing SHA256 for {path.name}.")
        verify_file_hash(path, str(expected_hash), f"Code-06 {key} artifact")
        paths[key] = path

    return manifest, sha256_file(manifest_path), paths


def load_code06_artifacts(paths: Mapping[str, Path]) -> Dict[str, object]:
    metadata = pd.read_csv(paths["metadata"], low_memory=False)
    variants = np.load(paths["variants"], allow_pickle=False)
    fitness = np.load(paths["fitness"], allow_pickle=False)
    source_rows = np.load(paths["source_rows"], allow_pickle=False)
    pfm_rows = np.load(paths["pfm_rows"], allow_pickle=False)

    n = EXPECTED_COMPLETE_CUBES
    if len(metadata) != n:
        raise RuntimeError(
            f"Unexpected metadata row count: expected {n:,}, observed {len(metadata):,}."
        )

    required_columns = {"cube_id", "wt", "far_corner", *ALT_COLUMNS}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise RuntimeError(f"Code-06 metadata missing columns: {sorted(missing)}")

    for name, array in (
        ("variants", variants),
        ("fitness", fitness),
        ("source_rows", source_rows),
        ("pfm_rows", pfm_rows),
    ):
        if array.shape != EXPECTED_VARIANT_SHAPE:
            raise RuntimeError(
                f"{name} shape mismatch: expected {EXPECTED_VARIANT_SHAPE}, observed {array.shape}."
            )

    if not np.array_equal(metadata["cube_id"].to_numpy(dtype=np.int64), np.arange(n)):
        raise RuntimeError("Code-06 cube_id ordering is not contiguous 0..N-1.")

    if metadata[list(ALT_COLUMNS)].duplicated().any():
        raise RuntimeError("Duplicate complete-cube alternative definitions detected.")

    if not np.all(variants[:, 0] == WT.encode("ascii")):
        raise RuntimeError("Code-06 cube vertex 0 is not WT in every cube.")

    far_corner = np.asarray([variant_to_text(v) for v in variants[:, 15]], dtype="U4")
    expected_far = metadata[list(ALT_COLUMNS)].astype(str).agg("".join, axis=1).to_numpy()
    if not np.array_equal(far_corner, expected_far):
        raise RuntimeError("Code-06 metadata and variant-array rows are misaligned.")

    if not np.isfinite(fitness).all():
        raise RuntimeError("Non-finite Code-06 cube fitness detected.")
    if np.any(source_rows < 0) or np.any(pfm_rows < 0):
        raise RuntimeError("Negative source/PFM row indices detected.")

    return {
        "metadata": metadata,
        "variants": variants,
        "fitness": fitness,
        "source_rows": source_rows,
        "pfm_rows": pfm_rows,
    }


# =============================================================================
# AA-identity-strict split
# =============================================================================


def create_amino_acid_split() -> Tuple[pd.DataFrame, Dict[int, Dict[str, str]]]:
    if N_TRAIN_AA + N_VAL_AA + N_TEST_AA != EXPECTED_ALTERNATIVES_PER_POSITION:
        raise RuntimeError("Frozen AA split sizes must sum to 19.")

    rng = np.random.default_rng(RANDOM_SEED)
    rows: List[Dict] = []
    maps: Dict[int, Dict[str, str]] = {}

    for position_index, (biological_position, wt_aa) in enumerate(zip(POSITIONS, WT), start=1):
        alternatives = [aa for aa in AMINO_ACIDS if aa != wt_aa]
        if len(alternatives) != EXPECTED_ALTERNATIVES_PER_POSITION:
            raise RuntimeError(
                f"Position {biological_position}: expected 19 non-WT amino acids."
            )

        shuffled = rng.permutation(alternatives).tolist()
        groups = {
            "estimator_train": shuffled[:N_TRAIN_AA],
            "estimator_val": shuffled[N_TRAIN_AA:N_TRAIN_AA + N_VAL_AA],
            "estimator_test": shuffled[N_TRAIN_AA + N_VAL_AA:],
        }

        all_assigned = set().union(*(set(v) for v in groups.values()))
        if all_assigned != set(alternatives):
            raise RuntimeError(f"Position {biological_position}: AA partition is incomplete.")
        if any(set(groups[a]) & set(groups[b]) for a, b in (
            ("estimator_train", "estimator_val"),
            ("estimator_train", "estimator_test"),
            ("estimator_val", "estimator_test"),
        )):
            raise RuntimeError(f"Position {biological_position}: AA split groups overlap.")

        split_map: Dict[str, str] = {}
        for split_name, amino_acids in groups.items():
            for aa in amino_acids:
                split_map[aa] = split_name
                rows.append(
                    {
                        "position_index": position_index,
                        "biological_position": biological_position,
                        "wt_aa": wt_aa,
                        "alternative_aa": aa,
                        "estimator_split": split_name,
                    }
                )
        maps[biological_position] = split_map

    assignment = pd.DataFrame(rows)
    return assignment, maps


def audit_assignment(assignment: pd.DataFrame, maps: Mapping[int, Mapping[str, str]]) -> Dict:
    if len(assignment) != 4 * EXPECTED_ALTERNATIVES_PER_POSITION:
        raise RuntimeError("AA assignment table has the wrong number of rows.")

    expected_group_sizes = {
        "estimator_train": N_TRAIN_AA,
        "estimator_val": N_VAL_AA,
        "estimator_test": N_TEST_AA,
    }

    per_position: Dict[str, Dict[str, object]] = {}
    for biological_position, wt_aa in zip(POSITIONS, WT):
        sub = assignment[assignment["biological_position"] == biological_position]
        if len(sub) != 19:
            raise RuntimeError(f"Position {biological_position}: assignment count != 19.")
        if set(sub["alternative_aa"]) != AMINO_ACID_SET - {wt_aa}:
            raise RuntimeError(f"Position {biological_position}: wrong non-WT AA set.")

        counts = sub["estimator_split"].value_counts().to_dict()
        if counts != expected_group_sizes:
            raise RuntimeError(
                f"Position {biological_position}: wrong split sizes. "
                f"Expected {expected_group_sizes}; observed {counts}."
            )

        per_position[str(biological_position)] = {
            "wt_aa": wt_aa,
            "train": sub.loc[sub["estimator_split"] == "estimator_train", "alternative_aa"].tolist(),
            "val": sub.loc[sub["estimator_split"] == "estimator_val", "alternative_aa"].tolist(),
            "test": sub.loc[sub["estimator_split"] == "estimator_test", "alternative_aa"].tolist(),
        }

    return {
        "random_seed": RANDOM_SEED,
        "train_identities_per_position": N_TRAIN_AA,
        "val_identities_per_position": N_VAL_AA,
        "test_identities_per_position": N_TEST_AA,
        "per_position": per_position,
    }


def assign_cubes(metadata: pd.DataFrame, maps: Mapping[int, Mapping[str, str]]) -> pd.DataFrame:
    mapped_columns: List[np.ndarray] = []

    for biological_position, column in zip(POSITIONS, ALT_COLUMNS):
        mapped = metadata[column].astype(str).map(maps[biological_position])
        if mapped.isna().any():
            bad = sorted(metadata.loc[mapped.isna(), column].astype(str).unique().tolist())
            raise RuntimeError(
                f"Unrecognized alternatives at position {biological_position}: {bad}"
            )
        mapped_columns.append(mapped.to_numpy(dtype=object))

    split_matrix = np.column_stack(mapped_columns)
    same_split = np.all(split_matrix == split_matrix[:, [0]], axis=1)
    labels = np.full(len(metadata), "mixed", dtype=object)
    labels[same_split] = split_matrix[same_split, 0]

    out = metadata.copy()
    out["estimator_split"] = labels
    return out


# =============================================================================
# Direct leakage audits
# =============================================================================


def unique_values_for_rows(array: np.ndarray, row_indices: np.ndarray) -> set:
    if len(row_indices) == 0:
        return set()
    return set(np.unique(array[row_indices].reshape(-1)).tolist())


def validate_cross_split_overlap(
    variants: np.ndarray,
    pfm_rows: np.ndarray,
    split_df: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, set], Dict[str, set]]:
    split_names = ("estimator_train", "estimator_val", "estimator_test")
    labels = split_df["estimator_split"].to_numpy(dtype=object)

    index_sets: Dict[str, np.ndarray] = {}
    variant_sets: Dict[str, set] = {}
    pfm_sets: Dict[str, set] = {}

    for split_name in split_names:
        indices = np.flatnonzero(labels == split_name).astype(np.int64)
        if indices.size == 0:
            raise RuntimeError(f"Strict split {split_name} contains zero cubes.")
        index_sets[split_name] = indices
        variant_sets[split_name] = {
            variant_to_text(v) for v in unique_values_for_rows(variants, indices)
        }
        pfm_sets[split_name] = {int(v) for v in unique_values_for_rows(pfm_rows, indices)}

    wt_pfm_rows = {
        int(pfm_rows[index_sets[split_name][0], 0])
        for split_name in split_names
    }
    if len(wt_pfm_rows) != 1:
        raise RuntimeError("WT does not map to one common PFM row across strict partitions.")
    wt_pfm_row = next(iter(wt_pfm_rows))

    rows: List[Dict] = []
    pairs = (
        ("estimator_train", "estimator_val"),
        ("estimator_train", "estimator_test"),
        ("estimator_val", "estimator_test"),
    )

    for left, right in pairs:
        shared_variants = variant_sets[left] & variant_sets[right]
        shared_nonwt_variants = shared_variants - {WT}
        shared_pfm_rows = pfm_sets[left] & pfm_sets[right]
        shared_nonwt_pfm_rows = shared_pfm_rows - {wt_pfm_row}

        if shared_nonwt_variants:
            raise RuntimeError(
                "AA-identity leakage detected in variant space.\n"
                f"{left} vs {right}: {sorted(shared_nonwt_variants)[:10]}"
            )
        if shared_variants != {WT}:
            raise RuntimeError(
                f"Expected only WT={WT} to be shared between {left} and {right}; "
                f"observed {sorted(shared_variants)}."
            )
        if shared_nonwt_pfm_rows:
            raise RuntimeError(
                "AA-identity leakage detected in PFM-row space.\n"
                f"{left} vs {right}: {sorted(shared_nonwt_pfm_rows)[:10]}"
            )
        if shared_pfm_rows != {wt_pfm_row}:
            raise RuntimeError(
                f"Expected only WT PFM row {wt_pfm_row} to be shared between {left} and {right}; "
                f"observed {sorted(shared_pfm_rows)}."
            )

        rows.append(
            {
                "split_1": left,
                "split_2": right,
                "shared_variants_total": len(shared_variants),
                "shared_nonwt_variants": len(shared_nonwt_variants),
                "shared_pfm_rows_total": len(shared_pfm_rows),
                "shared_nonwt_pfm_rows": len(shared_nonwt_pfm_rows),
                "shared_variant_values": ";".join(sorted(shared_variants)),
                "shared_pfm_rows": ";".join(str(x) for x in sorted(shared_pfm_rows)),
            }
        )

    return index_sets, pd.DataFrame(rows), variant_sets, pfm_sets


def make_variant_summary(variant_sets: Mapping[str, set]) -> pd.DataFrame:
    interpretation = {
        0: "WT",
        1: "single mutant",
        2: "double mutant",
        3: "triple mutant",
        4: "quadruple mutant",
    }
    rows: List[Dict] = []
    for split_name in ("estimator_train", "estimator_val", "estimator_test"):
        for hd in range(5):
            count = sum(hamming_from_wt(v) == hd for v in variant_sets[split_name])
            rows.append(
                {
                    "estimator_split": split_name,
                    "hd": hd,
                    "interpretation": interpretation[hd],
                    "unique_variants": int(count),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Full audit and saving
# =============================================================================


def audit_real_split(
    split_df: pd.DataFrame,
    assignment: pd.DataFrame,
    assignment_audit: Dict,
    variants: np.ndarray,
    pfm_rows: np.ndarray,
) -> Tuple[Dict, Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    counts_series = split_df["estimator_split"].value_counts()
    observed_counts = {
        name: int(counts_series.get(name, 0)) for name in EXPECTED_SPLIT_COUNTS
    }
    if observed_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "Unexpected AA-identity-strict complete-cube counts.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Observed: {observed_counts}"
        )

    strict_total = sum(observed_counts[k] for k in (
        "estimator_train", "estimator_val", "estimator_test"
    ))
    if strict_total != EXPECTED_STRICT_CUBES:
        raise RuntimeError(
            f"Expected {EXPECTED_STRICT_CUBES:,} strict cubes; observed {strict_total:,}."
        )
    if sum(observed_counts.values()) != EXPECTED_COMPLETE_CUBES:
        raise RuntimeError("Strict + mixed cube counts do not sum to complete cube population.")

    # Every strict cube must obey its identity label at every position.
    lookup = {
        (int(row.biological_position), str(row.alternative_aa)): str(row.estimator_split)
        for row in assignment.itertuples(index=False)
    }
    for split_name in ("estimator_train", "estimator_val", "estimator_test"):
        sub = split_df[split_df["estimator_split"] == split_name]
        for biological_position, column in zip(POSITIONS, ALT_COLUMNS):
            observed = {
                lookup[(biological_position, str(aa))]
                for aa in sub[column].astype(str).unique()
            }
            if observed != {split_name}:
                raise RuntimeError(
                    f"Cube assignment rule violated for {split_name}, position {biological_position}."
                )

    index_sets, overlap_df, variant_sets, pfm_sets = validate_cross_split_overlap(
        variants, pfm_rows, split_df
    )
    variant_summary = make_variant_summary(variant_sets)

    audit = {
        "complete_cubes": EXPECTED_COMPLETE_CUBES,
        "split_counts": observed_counts,
        "strict_cubes": strict_total,
        "mixed_cubes": observed_counts["mixed"],
        "assignment": assignment_audit,
        "cube_assignment_rule": "PASS",
        "cross_split_nonwt_variant_overlap": 0,
        "cross_split_nonwt_pfm_row_overlap": 0,
        "wt_is_only_shared_sequence": True,
        "phenotype_used_to_create_split": False,
        "esm_representation_used_to_create_split": False,
        "overall_pass": True,
    }
    return audit, index_sets, overlap_df, variant_summary


def save_outputs(
    output_dir: Path,
    assignment: pd.DataFrame,
    split_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    variant_summary: pd.DataFrame,
    index_sets: Mapping[str, np.ndarray],
    audit: Dict,
    code06_manifest_path: Path,
    code06_manifest_sha: str,
) -> Tuple[List[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    assignment_path = output_dir / "gb1_strict_aa_assignment.csv"
    split_path = output_dir / "gb1_strict_cube_split.csv"
    overlap_path = output_dir / "gb1_strict_overlap_check.csv"
    summary_path = output_dir / "gb1_strict_variant_summary.csv"
    indices_path = output_dir / "gb1_strict_split_indices.npz"
    manifest_path = output_dir / "gb1_strict_split_manifest.json"

    assignment.to_csv(assignment_path, index=False, lineterminator="\n")
    split_df.to_csv(split_path, index=False, lineterminator="\n")
    overlap_df.to_csv(overlap_path, index=False, lineterminator="\n")
    variant_summary.to_csv(summary_path, index=False, lineterminator="\n")
    np.savez_compressed(
        indices_path,
        estimator_train_indices=index_sets["estimator_train"].astype(np.int64),
        estimator_val_indices=index_sets["estimator_val"].astype(np.int64),
        estimator_test_indices=index_sets["estimator_test"].astype(np.int64),
    )

    output_paths = [assignment_path, split_path, overlap_path, summary_path, indices_path]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": "Build the AA-identity-strict GB1 cube split for held-out ORBIT-FMIB critic estimation.",
        "upstream": {
            "code06_manifest_path": str(code06_manifest_path.resolve()),
            "code06_manifest_sha256": code06_manifest_sha,
            "code06_overall_pass": True,
        },
        "split_contract": {
            "wt_variant": WT,
            "biological_positions": list(POSITIONS),
            "random_seed": RANDOM_SEED,
            "nonwt_identities_per_position": 19,
            "train_identities_per_position": N_TRAIN_AA,
            "val_identities_per_position": N_VAL_AA,
            "test_identities_per_position": N_TEST_AA,
            "rule": "A cube is strict train/val/test only when all four selected alternative identities belong to that same identity partition; otherwise mixed.",
        },
        "frozen_expected_counts": EXPECTED_SPLIT_COUNTS,
        "audit": audit,
        "scientific_boundary": {
            "uses_fitness_to_assign_split": False,
            "uses_esm_representations_to_assign_split": False,
            "replaces_flip_split": False,
            "implies_esm_pretraining_novelty": False,
        },
        "outputs": {
            "aa_assignment": assignment_path.name,
            "cube_split": split_path.name,
            "overlap_check": overlap_path.name,
            "variant_summary": summary_path.name,
            "split_indices": indices_path.name,
        },
        "overall_pass": True,
    }
    manifest["file_sha256"] = {path.name: sha256_file(path) for path in output_paths}
    write_json(manifest_path, manifest)
    return output_paths, manifest_path


# =============================================================================
# Deterministic self-test
# =============================================================================


def run_self_test() -> None:
    assignment1, maps1 = create_amino_acid_split()
    assignment2, maps2 = create_amino_acid_split()

    pd.testing.assert_frame_equal(assignment1, assignment2)
    assert maps1 == maps2
    assignment_audit = audit_assignment(assignment1, maps1)
    assert assignment_audit["random_seed"] == RANDOM_SEED

    # Candidate-space arithmetic implied by the 10/4/5 identity partition.
    assert N_TRAIN_AA ** 4 == 10_000
    assert N_VAL_AA ** 4 == 256
    assert N_TEST_AA ** 4 == 625
    assert 19 ** 4 - 10_000 - 256 - 625 == 119_440
    assert 19 ** 4 == 130_321

    # Toy cube definitions chosen directly from deterministic maps.
    def first_aa(position: int, split_name: str) -> str:
        return next(aa for aa, label in maps1[position].items() if label == split_name)

    rows = []
    for cube_id, split_name in enumerate(("estimator_train", "estimator_val", "estimator_test")):
        alts = [first_aa(position, split_name) for position in POSITIONS]
        rows.append(
            {
                "cube_id": cube_id,
                "wt": WT,
                "alt_39": alts[0],
                "alt_40": alts[1],
                "alt_41": alts[2],
                "alt_54": alts[3],
                "far_corner": "".join(alts),
            }
        )

    mixed_alts = [
        first_aa(POSITIONS[0], "estimator_train"),
        first_aa(POSITIONS[1], "estimator_val"),
        first_aa(POSITIONS[2], "estimator_test"),
        first_aa(POSITIONS[3], "estimator_train"),
    ]
    rows.append(
        {
            "cube_id": 3,
            "wt": WT,
            "alt_39": mixed_alts[0],
            "alt_40": mixed_alts[1],
            "alt_41": mixed_alts[2],
            "alt_54": mixed_alts[3],
            "far_corner": "".join(mixed_alts),
        }
    )
    toy = pd.DataFrame(rows)
    assigned = assign_cubes(toy, maps1)
    assert assigned["estimator_split"].tolist() == [
        "estimator_train", "estimator_val", "estimator_test", "mixed"
    ]

    print("SELF-TEST: PASS")
    print(
        "Verified: deterministic 10/4/5 AA identity partition, 19^4 candidate arithmetic, "
        "strict cube-assignment rule, and mixed-cube exclusion."
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run deterministic split tests and exit.")
    parser.add_argument("--code06-manifest", type=Path, default=None, help="Override Code-06 cube manifest path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override Code-07 output directory.")
    parser.add_argument("--no-write", action="store_true", help="Run all audits without writing output files.")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    root = project_root()
    output_dir = args.output_dir or (root / "04 Experiments" / "EXP02_GB1_ESM2")
    code06_manifest_path = args.code06_manifest or (output_dir / "gb1_cube_manifest.json")

    print("=" * 76)
    print(f"{CODE_NAME} — AA-IDENTITY-STRICT GB1 ESTIMATOR SPLIT")
    print("=" * 76)
    print()

    code06_manifest, code06_sha, paths = load_and_verify_code06(code06_manifest_path)
    print("Code-06 provenance gate: PASS")
    print(f"  manifest SHA256: {code06_sha}")
    print(f"  complete cubes: {EXPECTED_COMPLETE_CUBES:,}")
    print("  all Code-06 artifact hashes: PASS")
    print()

    artifacts = load_code06_artifacts(paths)
    metadata = artifacts["metadata"]
    variants = artifacts["variants"]
    pfm_rows = artifacts["pfm_rows"]
    print("Code-06 cube artifact audit: PASS")
    print(f"  metadata rows: {len(metadata):,}")
    print(f"  cube arrays: {variants.shape[0]:,} x {variants.shape[1]}")
    print("  WT vertex / far-corner / row alignment: PASS")
    print()

    assignment, maps = create_amino_acid_split()
    assignment_audit = audit_assignment(assignment, maps)
    print("Amino-acid identity partition: PASS")
    print(f"  RNG seed: {RANDOM_SEED}")
    print(f"  per position: {N_TRAIN_AA} train / {N_VAL_AA} val / {N_TEST_AA} test non-WT identities")
    for position in POSITIONS:
        info = assignment_audit["per_position"][str(position)]
        print(f"  position {position} (WT={info['wt_aa']}):")
        print("    train: " + " ".join(info["train"]))
        print("    val:   " + " ".join(info["val"]))
        print("    test:  " + " ".join(info["test"]))
    print()

    split_df = assign_cubes(metadata, maps)
    audit, index_sets, overlap_df, variant_summary = audit_real_split(
        split_df, assignment, assignment_audit, variants, pfm_rows
    )

    print("AA-identity-strict complete-cube split: PASS")
    for split_name in ("estimator_train", "estimator_val", "estimator_test", "mixed"):
        print(f"  {split_name:15s}: {audit['split_counts'][split_name]:,}")
    print(f"  strict retained: {audit['strict_cubes']:,}")
    print()

    print("Cross-partition leakage audit: PASS")
    print("  shared non-WT variants: 0")
    print("  shared non-WT PFM rows: 0")
    print(f"  only shared sequence: WT = {WT}")
    print()

    print("Scientific split boundary: PASS")
    print("  fitness used to create split: NO")
    print("  ESM representations used to create split: NO")
    print("  replaces FLIP split: NO")
    print()

    if args.no_write:
        print("NO-WRITE MODE: all real-data strict-split audits passed; no files written.")
    else:
        output_paths, manifest_path = save_outputs(
            output_dir=output_dir,
            assignment=assignment,
            split_df=split_df,
            overlap_df=overlap_df,
            variant_summary=variant_summary,
            index_sets=index_sets,
            audit=audit,
            code06_manifest_path=code06_manifest_path,
            code06_manifest_sha=code06_sha,
        )
        print("Saved:")
        for path in output_paths:
            print(f"  {path}")
        print(f"  {manifest_path}")

    print()
    print(f"{CODE_NAME}: PASS")
    print("Leakage-safe GB1 estimator partitions are frozen for downstream ESM-2 extraction indexing.")


if __name__ == "__main__":
    main()
