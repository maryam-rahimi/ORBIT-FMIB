#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 08 — Build the exact frozen ESM-2 extraction index.

Purpose
-------
Construct the smallest label-free sequence index needed for Phase-1 ESM-2
representation extraction after the AA-identity-strict GB1 estimator split
has been frozen by Code 07.

Only sequences that occur in estimator_train / estimator_val / estimator_test
complete cubes are retained. Mixed cubes are deliberately excluded.

Scientific boundary
-------------------
Code 08 does NOT:
  - run ESM-2,
  - use experimental fitness,
  - compute Walsh coefficients,
  - fit a mutual-information critic.

It only converts the strict cube partitions into a deterministic union of
PFM rows / protein sequences and freezes the cube-vertex -> extraction-row
mapping used by Code 09 and later stages.

Default inputs
--------------
04 Experiments/EXP02_GB1_ESM2/gb1_strict_split_manifest.json
04 Experiments/EXP02_GB1_ESM2/gb1_cube_manifest.json
02 Data/PFM_Inputs/esm2_loading_validation_manifest.json

Default outputs
---------------
04 Experiments/EXP02_GB1_ESM2/
    gb1_esm2_extraction_index.csv
    gb1_strict_cube_extract_rows.npz
    gb1_esm2_extraction_split_summary.csv
    gb1_esm2_extraction_manifest.json

Usage
-----
    python "03 Code\\08_build_esm2_extraction_index.py" --self-test
    python "03 Code\\08_build_esm2_extraction_index.py"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Frozen Phase-1 contract
# =============================================================================

CODE_NAME = "ORBIT-FMIB CODE 08"
SCHEMA_VERSION = 1

EXPECTED_CODE07 = "ORBIT-FMIB CODE 07"
EXPECTED_CODE06 = "ORBIT-FMIB CODE 06"
EXPECTED_CODE02_FILE = "02_validate_esm2_loading.py"

MODEL_NAME = "esm2_t30_150M_UR50D"
EXPECTED_NUM_LAYERS = 30
EXPECTED_EMBED_DIM = 640
EXPECTED_MODEL_REPR_LAYERS = tuple(range(EXPECTED_NUM_LAYERS + 1))  # 0..30

WT = "VDGV"
POSITIONS = (39, 40, 41, 54)
EXPECTED_TARGET_TOKEN_INDICES = (39, 40, 41, 54)

EXPECTED_PFM_ROWS = 149_361
EXPECTED_COMPLETE_CUBES = 109_235
EXPECTED_STRICT_COUNTS = {
    "estimator_train": 7_352,
    "estimator_val": 245,
    "estimator_test": 613,
}
STRICT_SPLIT_KEYS = {
    "estimator_train": "estimator_train_indices",
    "estimator_val": "estimator_val_indices",
    "estimator_test": "estimator_test_indices",
}

EXPECTED_VERTICES_PER_CUBE = 16
EXPECTED_SEQUENCE_LENGTH = 265


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


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Path: {path}\nExpected: {expected}\nObserved: {observed}"
        )


def locate_unique_existing(paths: Sequence[Path], label: str, override_flag: str) -> Path:
    existing = [p for p in paths if p.exists()]
    if not existing:
        formatted = "\n".join(f"  {p}" for p in paths)
        raise FileNotFoundError(f"Could not find {label}. Checked:\n{formatted}")
    if len(existing) > 1:
        formatted = "\n".join(f"  {p}" for p in existing)
        raise RuntimeError(
            f"Multiple {label} candidates exist; refusing to guess.\n"
            f"Existing candidates:\n{formatted}\n"
            f"Pass the intended file explicitly with {override_flag}."
        )
    return existing[0]


# =============================================================================
# Code-07 provenance
# =============================================================================


def load_and_verify_code07(manifest_path: Path) -> Tuple[Dict, str, Dict[str, Path]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Code-07 manifest not found: {manifest_path}\n"
            "Run Code 07 successfully before Code 08."
        )

    manifest = load_json(manifest_path)
    if manifest.get("code_name") != EXPECTED_CODE07:
        raise RuntimeError(
            f"Unexpected Code-07 manifest code_name: {manifest.get('code_name')!r}"
        )
    if not bool(manifest.get("overall_pass", False)):
        raise RuntimeError("Code-07 manifest does not record overall_pass=true.")

    audit = manifest.get("audit", {})
    if not bool(audit.get("overall_pass", False)):
        raise RuntimeError("Code-07 audit.overall_pass is not true.")

    observed_counts = audit.get("split_counts", {})
    for split_name, expected in EXPECTED_STRICT_COUNTS.items():
        if int(observed_counts.get(split_name, -1)) != expected:
            raise RuntimeError(
                f"Code-07 strict count mismatch for {split_name}: "
                f"expected {expected:,}, observed {observed_counts.get(split_name)!r}."
            )

    outputs = manifest.get("outputs", {})
    hashes = manifest.get("file_sha256", {})
    required_output_keys = (
        "aa_assignment",
        "cube_split",
        "overlap_check",
        "variant_summary",
        "split_indices",
    )

    paths: Dict[str, Path] = {}
    for key in required_output_keys:
        name = outputs.get(key)
        if not name:
            raise RuntimeError(f"Code-07 manifest is missing outputs.{key}.")
        path = manifest_path.parent / str(name)
        expected_hash = hashes.get(path.name)
        if not expected_hash:
            raise RuntimeError(f"Code-07 manifest is missing SHA256 for {path.name}.")
        verify_file_hash(path, str(expected_hash), f"Code-07 {key} artifact")
        paths[key] = path

    upstream = manifest.get("upstream", {})
    code06_path_raw = upstream.get("code06_manifest_path")
    code06_sha = upstream.get("code06_manifest_sha256")
    if not code06_path_raw or not code06_sha:
        raise RuntimeError("Code-07 manifest does not pin the Code-06 manifest.")

    code06_path = Path(str(code06_path_raw))
    if not code06_path.exists():
        # Safe relocation fallback: Code 06 and Code 07 normally share EXP02.
        candidate = manifest_path.parent / "gb1_cube_manifest.json"
        if candidate.exists():
            code06_path = candidate
        else:
            raise FileNotFoundError(
                f"Pinned Code-06 manifest is unavailable: {code06_path_raw}"
            )
    verify_file_hash(code06_path, str(code06_sha), "Code-06 manifest pinned by Code 07")
    paths["code06_manifest"] = code06_path

    return manifest, sha256_file(manifest_path), paths


# =============================================================================
# Code-06 / PFM provenance
# =============================================================================


def load_and_verify_code06_for_extraction(
    manifest_path: Path,
) -> Tuple[Dict, str, Path, Path]:
    manifest = load_json(manifest_path)
    if manifest.get("code_name") != EXPECTED_CODE06:
        raise RuntimeError(
            f"Unexpected Code-06 manifest code_name: {manifest.get('code_name')!r}"
        )
    if not bool(manifest.get("overall_pass", False)):
        raise RuntimeError("Code-06 manifest does not record overall_pass=true.")

    audit = manifest.get("audit", {})
    if int(audit.get("complete_cubes", -1)) != EXPECTED_COMPLETE_CUBES:
        raise RuntimeError("Code-06 complete-cube count is not 109,235.")

    outputs = manifest.get("outputs", {})
    hashes = manifest.get("file_sha256", {})

    pfm_rows_name = outputs.get("pfm_rows")
    if not pfm_rows_name:
        raise RuntimeError("Code-06 manifest is missing outputs.pfm_rows.")
    pfm_rows_path = manifest_path.parent / str(pfm_rows_name)
    pfm_rows_hash = hashes.get(pfm_rows_path.name)
    if not pfm_rows_hash:
        raise RuntimeError("Code-06 manifest is missing the PFM-row artifact hash.")
    verify_file_hash(pfm_rows_path, str(pfm_rows_hash), "Code-06 PFM-row artifact")

    inputs = manifest.get("inputs", {})
    pfm_path_raw = inputs.get("pfm_index_path")
    pfm_sha = inputs.get("pfm_index_sha256")
    if not pfm_path_raw or not pfm_sha:
        raise RuntimeError("Code-06 manifest does not pin its Code-01 PFM index.")

    pfm_path = Path(str(pfm_path_raw))
    if not pfm_path.exists():
        root = project_root()
        pfm_path = locate_unique_existing(
            [
                root / "02 Data" / "PFM_Inputs" / "gb1_pfm_index.csv",
                root / "02 Data" / "03_PFM_Inputs" / "GB1" / "gb1_pfm_index.csv",
                root / "02 Data" / "03_PFM_Inputs" / "gb1_pfm_index.csv",
            ],
            "Code-01 PFM index",
            "--pfm-index",
        )
    verify_file_hash(pfm_path, str(pfm_sha), "Code-01 label-free PFM index")

    return manifest, sha256_file(manifest_path), pfm_rows_path, pfm_path


def load_and_audit_pfm_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    forbidden = [c for c in df.columns if "fitness" in str(c).lower()]
    if forbidden:
        raise RuntimeError(
            "LABEL BOUNDARY FAILURE: the PFM index contains fitness-bearing columns: "
            f"{forbidden}"
        )

    required = {"pfm_row_id", "variant", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"PFM index missing required columns: {sorted(missing)}")

    if len(df) != EXPECTED_PFM_ROWS:
        raise RuntimeError(
            f"Unexpected PFM row count: expected {EXPECTED_PFM_ROWS:,}, observed {len(df):,}."
        )

    row_ids = pd.to_numeric(df["pfm_row_id"], errors="raise").to_numpy(dtype=np.int64)
    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(row_ids, expected):
        raise RuntimeError("PFM row IDs are not contiguous deterministic IDs 0..N-1.")

    if df["variant"].isna().any() or df["variant"].astype(str).duplicated().any():
        raise RuntimeError("PFM variants must be present and unique.")
    if df["sequence"].isna().any() or df["sequence"].astype(str).duplicated().any():
        raise RuntimeError("PFM sequences must be present and unique.")

    variants = df["variant"].astype(str).str.strip().str.upper()
    if not (variants.str.len() == 4).all():
        raise RuntimeError("PFM four-site variants must all have length 4.")
    df = df.copy()
    df["variant"] = variants

    lengths = df["sequence"].astype(str).str.len().to_numpy(dtype=np.int64)
    if not np.all(lengths == EXPECTED_SEQUENCE_LENGTH):
        values, counts = np.unique(lengths, return_counts=True)
        observed = {int(v): int(c) for v, c in zip(values, counts)}
        raise RuntimeError(
            f"Unexpected full-sequence lengths in PFM index. "
            f"Expected all {EXPECTED_SEQUENCE_LENGTH}; observed {observed}."
        )

    wt = df.loc[df["variant"] == WT]
    if len(wt) != 1:
        raise RuntimeError(f"Expected exactly one WT PFM row {WT}; found {len(wt)}.")

    return df.reset_index(drop=True)


# =============================================================================
# Code-02 ESM-2 contract
# =============================================================================


def load_and_verify_code02(manifest_path: Path, pfm_sha: str) -> Tuple[Dict, str]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Code-02 ESM-2 validation manifest not found: {manifest_path}\n"
            "Run Code 02 successfully before Code 08."
        )

    manifest = load_json(manifest_path)

    if manifest.get("status") != "PASS":
        raise RuntimeError("Code-02 manifest status is not PASS.")
    if manifest.get("code") != EXPECTED_CODE02_FILE:
        raise RuntimeError(
            f"Unexpected Code-02 manifest code field: {manifest.get('code')!r}"
        )
    if manifest.get("model_name") != MODEL_NAME:
        raise RuntimeError(
            f"Unexpected frozen model: {manifest.get('model_name')!r}"
        )

    arch = manifest.get("architecture", {})
    if int(arch.get("num_layers", -1)) != EXPECTED_NUM_LAYERS:
        raise RuntimeError("Code-02 transformer depth is not 30.")
    if int(arch.get("embed_dim", -1)) != EXPECTED_EMBED_DIM:
        raise RuntimeError("Code-02 embedding dimension is not 640.")
    if bool(arch.get("training_flag", True)):
        raise RuntimeError("Code-02 manifest indicates ESM-2 was in training mode.")
    if not bool(arch.get("all_parameters_frozen", False)):
        raise RuntimeError("Code-02 manifest does not confirm all parameters frozen.")

    tokenization = manifest.get("tokenization", {})
    if int(tokenization.get("sequence_length", -1)) != EXPECTED_SEQUENCE_LENGTH:
        raise RuntimeError("Code-02 sequence-length contract is not 265.")

    positions = tuple(int(x) for x in manifest.get("gb1_domain_positions_1based", []))
    if positions != POSITIONS:
        raise RuntimeError(
            f"Code-02 GB1 positions mismatch: expected {POSITIONS}, observed {positions}."
        )

    forward = manifest.get("forward_pass", {})
    target_indices = tuple(int(x) for x in forward.get("target_token_indices", []))
    if target_indices != EXPECTED_TARGET_TOKEN_INDICES:
        raise RuntimeError(
            "Code-02 target token indices mismatch.\n"
            f"Expected: {EXPECTED_TARGET_TOKEN_INDICES}\nObserved: {target_indices}"
        )

    source_pfm_sha = str(manifest.get("source_pfm_index_sha256", ""))
    if source_pfm_sha.lower() != pfm_sha.lower():
        raise RuntimeError(
            "Code-02 / current PFM-index provenance mismatch.\n"
            f"Code-02 pinned: {source_pfm_sha}\nCurrent: {pfm_sha}"
        )

    return manifest, sha256_file(manifest_path)


# =============================================================================
# Strict cube loading and extraction-index construction
# =============================================================================


def load_strict_indices(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays: Dict[str, np.ndarray] = {}
        for split_name, key in STRICT_SPLIT_KEYS.items():
            if key not in data.files:
                raise RuntimeError(f"Code-07 split NPZ is missing key {key!r}.")
            idx = np.asarray(data[key], dtype=np.int64).reshape(-1)
            expected_n = EXPECTED_STRICT_COUNTS[split_name]
            if len(idx) != expected_n:
                raise RuntimeError(
                    f"{split_name} cube-index count mismatch: "
                    f"expected {expected_n:,}, observed {len(idx):,}."
                )
            if len(np.unique(idx)) != len(idx):
                raise RuntimeError(f"{split_name} contains duplicate cube indices.")
            if idx.size and (idx.min() < 0 or idx.max() >= EXPECTED_COMPLETE_CUBES):
                raise RuntimeError(f"{split_name} contains out-of-range cube indices.")
            arrays[split_name] = idx

    names = tuple(STRICT_SPLIT_KEYS)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = np.intersect1d(arrays[left], arrays[right], assume_unique=False)
            if shared.size:
                raise RuntimeError(
                    f"Strict cube-index overlap detected between {left} and {right}."
                )

    return arrays


def load_pfm_cube_rows(path: Path) -> np.ndarray:
    rows = np.load(path, allow_pickle=False)
    expected_shape = (EXPECTED_COMPLETE_CUBES, EXPECTED_VERTICES_PER_CUBE)
    if rows.shape != expected_shape:
        raise RuntimeError(
            f"Code-06 PFM-row matrix has shape {rows.shape}; expected {expected_shape}."
        )
    if not np.issubdtype(rows.dtype, np.integer):
        raise RuntimeError("Code-06 PFM-row matrix must have integer dtype.")
    rows = rows.astype(np.int64, copy=False)
    if rows.min() < 0 or rows.max() >= EXPECTED_PFM_ROWS:
        raise RuntimeError("Code-06 PFM-row matrix contains out-of-range row IDs.")
    return rows


def build_extraction_artifacts(
    pfm: pd.DataFrame,
    cube_pfm_rows: np.ndarray,
    strict_indices: Mapping[str, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame, Dict]:
    split_pfm_sets: Dict[str, set[int]] = {}
    strict_cube_pfm: Dict[str, np.ndarray] = {}

    for split_name, cube_ids in strict_indices.items():
        matrix = cube_pfm_rows[cube_ids]
        strict_cube_pfm[split_name] = matrix
        split_pfm_sets[split_name] = {
            int(x) for x in np.unique(matrix.reshape(-1)).tolist()
        }

    wt_rows = pfm.loc[pfm["variant"] == WT, "pfm_row_id"].astype(np.int64).to_numpy()
    if len(wt_rows) != 1:
        raise RuntimeError("Could not resolve a unique WT PFM row.")
    wt_pfm_row = int(wt_rows[0])

    overlap_rows = []
    split_names = tuple(STRICT_SPLIT_KEYS)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            shared = split_pfm_sets[left] & split_pfm_sets[right]
            shared_nonwt = shared - {wt_pfm_row}
            if shared_nonwt:
                raise RuntimeError(
                    f"Non-WT PFM-row leakage between {left} and {right}: "
                    f"{sorted(shared_nonwt)[:10]}"
                )
            if shared != {wt_pfm_row}:
                raise RuntimeError(
                    f"Expected only WT PFM row {wt_pfm_row} shared by {left}/{right}; "
                    f"observed {sorted(shared)}."
                )
            overlap_rows.append(
                {
                    "split_1": left,
                    "split_2": right,
                    "shared_pfm_rows_total": len(shared),
                    "shared_nonwt_pfm_rows": len(shared_nonwt),
                    "shared_pfm_rows": ";".join(str(x) for x in sorted(shared)),
                }
            )

    union_rows = np.asarray(
        sorted(set().union(*split_pfm_sets.values())),
        dtype=np.int64,
    )
    if union_rows.size == 0:
        raise RuntimeError("Strict extraction union is empty.")
    if wt_pfm_row not in set(union_rows.tolist()):
        raise RuntimeError("WT PFM row is absent from the strict extraction union.")

    extraction = pfm.iloc[union_rows].copy().reset_index(drop=True)
    extraction.insert(0, "extract_row_id", np.arange(len(extraction), dtype=np.int64))

    for split_name in split_names:
        extraction[f"in_{split_name}"] = extraction["pfm_row_id"].isin(
            split_pfm_sets[split_name]
        )

    membership_labels = []
    for row in extraction.itertuples(index=False):
        memberships = [
            split_name
            for split_name in split_names
            if bool(getattr(row, f"in_{split_name}"))
        ]
        if int(row.pfm_row_id) == wt_pfm_row:
            if set(memberships) != set(split_names):
                raise RuntimeError("WT does not belong to all three strict extraction sets.")
            membership_labels.append("shared_wt")
        else:
            if len(memberships) != 1:
                raise RuntimeError(
                    f"Non-WT PFM row {row.pfm_row_id} has unexpected memberships: {memberships}"
                )
            membership_labels.append(memberships[0])

    extraction["strict_membership"] = membership_labels
    extraction["sequence_length"] = extraction["sequence"].astype(str).str.len().astype(np.int64)

    # Dense deterministic PFM-row -> extraction-row map.
    dense = np.full(len(pfm), -1, dtype=np.int64)
    dense[union_rows] = np.arange(len(union_rows), dtype=np.int64)

    cube_extract_rows: Dict[str, np.ndarray] = {}
    for split_name, pfm_matrix in strict_cube_pfm.items():
        mapped = dense[pfm_matrix]
        if np.any(mapped < 0):
            raise RuntimeError(f"{split_name} contains PFM rows absent from extraction union.")
        cube_extract_rows[split_name] = mapped.astype(np.int64, copy=False)

    # Exact round-trip: extraction index must reproduce every strict cube PFM row.
    extraction_pfm = extraction["pfm_row_id"].to_numpy(dtype=np.int64)
    for split_name in split_names:
        reconstructed = extraction_pfm[cube_extract_rows[split_name]]
        if not np.array_equal(reconstructed, strict_cube_pfm[split_name]):
            raise RuntimeError(
                f"{split_name} extraction-row mapping does not round-trip to PFM rows."
            )

    summary_rows = []
    for split_name in split_names:
        rows = split_pfm_sets[split_name]
        nonwt = rows - {wt_pfm_row}
        summary_rows.append(
            {
                "partition": split_name,
                "strict_cubes": len(strict_indices[split_name]),
                "unique_pfm_rows_including_wt": len(rows),
                "unique_nonwt_pfm_rows": len(nonwt),
                "wt_present": wt_pfm_row in rows,
            }
        )
    summary_rows.append(
        {
            "partition": "union",
            "strict_cubes": sum(len(strict_indices[k]) for k in split_names),
            "unique_pfm_rows_including_wt": len(union_rows),
            "unique_nonwt_pfm_rows": len(union_rows) - 1,
            "wt_present": True,
        }
    )
    summary = pd.DataFrame(summary_rows)

    audit = {
        "strict_cube_counts": {
            k: int(len(strict_indices[k])) for k in split_names
        },
        "unique_sequence_counts": {
            k: int(len(split_pfm_sets[k])) for k in split_names
        },
        "union_unique_sequences": int(len(union_rows)),
        "union_unique_nonwt_sequences": int(len(union_rows) - 1),
        "wt_pfm_row": wt_pfm_row,
        "cross_split_nonwt_pfm_row_overlap": 0,
        "wt_is_only_shared_pfm_row": True,
        "cube_to_extraction_roundtrip": "PASS",
        "all_extraction_sequences_unique": bool(
            not extraction["sequence"].astype(str).duplicated().any()
        ),
        "all_extraction_variants_unique": bool(
            not extraction["variant"].astype(str).duplicated().any()
        ),
        "fitness_present_in_extraction_index": False,
        "mixed_cubes_included": False,
        "overall_pass": True,
    }

    return extraction, cube_extract_rows, summary, audit


# =============================================================================
# Saving
# =============================================================================


def save_outputs(
    output_dir: Path,
    extraction: pd.DataFrame,
    cube_extract_rows: Mapping[str, np.ndarray],
    summary: pd.DataFrame,
    audit: Dict,
    code07_manifest_path: Path,
    code07_sha: str,
    code06_manifest_path: Path,
    code06_sha: str,
    code02_manifest_path: Path,
    code02_sha: str,
    pfm_index_path: Path,
    pfm_sha: str,
) -> Tuple[list[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "gb1_esm2_extraction_index.csv"
    map_path = output_dir / "gb1_strict_cube_extract_rows.npz"
    summary_path = output_dir / "gb1_esm2_extraction_split_summary.csv"
    manifest_path = output_dir / "gb1_esm2_extraction_manifest.json"

    extraction.to_csv(index_path, index=False, lineterminator="\n")
    np.savez_compressed(
        map_path,
        estimator_train_extract_rows=cube_extract_rows["estimator_train"],
        estimator_val_extract_rows=cube_extract_rows["estimator_val"],
        estimator_test_extract_rows=cube_extract_rows["estimator_test"],
    )
    summary.to_csv(summary_path, index=False, lineterminator="\n")

    output_paths = [index_path, map_path, summary_path]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": "Freeze the exact label-free GB1 sequence union and cube-vertex mapping required for ESM-2 extraction.",
        "upstream": {
            "code07_manifest_path": str(code07_manifest_path.resolve()),
            "code07_manifest_sha256": code07_sha,
            "code07_overall_pass": True,
            "code06_manifest_path": str(code06_manifest_path.resolve()),
            "code06_manifest_sha256": code06_sha,
            "code06_overall_pass": True,
            "code02_manifest_path": str(code02_manifest_path.resolve()),
            "code02_manifest_sha256": code02_sha,
            "code02_status": "PASS",
            "pfm_index_path": str(pfm_index_path.resolve()),
            "pfm_index_sha256": pfm_sha,
        },
        "esm2_contract": {
            "model_name": MODEL_NAME,
            "transformer_layers": EXPECTED_NUM_LAYERS,
            "representation_layers": list(EXPECTED_MODEL_REPR_LAYERS),
            "embedding_dim": EXPECTED_EMBED_DIM,
            "biological_positions_1based": list(POSITIONS),
            "esm_token_indices": list(EXPECTED_TARGET_TOKEN_INDICES),
            "full_sequence_length": EXPECTED_SEQUENCE_LENGTH,
            "model_frozen": True,
            "model_eval_mode": True,
        },
        "extraction_contract": {
            "index_order": "ascending pfm_row_id",
            "extract_row_id_contiguous": True,
            "include_partitions": list(STRICT_SPLIT_KEYS),
            "exclude_mixed_cubes": True,
            "fitness_columns_allowed": False,
            "code09_expected_saved_site_tensor_shape_per_layer": [
                "N_extract",
                len(POSITIONS),
                EXPECTED_EMBED_DIM,
            ],
            "recommended_disk_dtype_for_code09": "float16",
            "required_compute_dtype_for_walsh_mi_downstream": "float32_or_float64",
        },
        "audit": audit,
        "outputs": {
            "extraction_index": index_path.name,
            "strict_cube_extract_rows": map_path.name,
            "split_summary": summary_path.name,
        },
        "overall_pass": True,
    }
    manifest["file_sha256"] = {p.name: sha256_file(p) for p in output_paths}
    write_json(manifest_path, manifest)
    return output_paths, manifest_path


# =============================================================================
# Deterministic self-test
# =============================================================================


def run_self_test() -> None:
    # Seven toy PFM rows. Row 0 is WT and is the only shared row.
    pfm = pd.DataFrame(
        {
            "pfm_row_id": np.arange(7, dtype=np.int64),
            "variant": ["VDGV", "ADGV", "VAGV", "CDGV", "VCGV", "EDGV", "VEGV"],
            "sequence": ["A" * EXPECTED_SEQUENCE_LENGTH for _ in range(7)],
        }
    )
    # Make sequences unique while preserving length 265.
    pfm["sequence"] = [
        ("A" * (EXPECTED_SEQUENCE_LENGTH - 1)) + chr(ord("A") + i)
        for i in range(7)
    ]

    # Three toy cubes, one per split. Vertex matrices are synthetic for mapping test.
    # Repeated row IDs are fine inside this mapping self-test; real Code-06 cubes have
    # 16 biological vertices and are audited upstream.
    cube_rows = np.asarray(
        [
            [0, 1, 2] * 5 + [0],
            [0, 3, 4] * 5 + [0],
            [0, 5, 6] * 5 + [0],
        ],
        dtype=np.int64,
    )
    assert cube_rows.shape == (3, 16)

    strict = {
        "estimator_train": np.asarray([0], dtype=np.int64),
        "estimator_val": np.asarray([1], dtype=np.int64),
        "estimator_test": np.asarray([2], dtype=np.int64),
    }

    extraction, maps, summary, audit = build_extraction_artifacts(
        pfm=pfm,
        cube_pfm_rows=cube_rows,
        strict_indices=strict,
    )

    assert extraction["pfm_row_id"].tolist() == list(range(7))
    assert extraction.loc[0, "strict_membership"] == "shared_wt"
    assert audit["union_unique_sequences"] == 7
    assert audit["cross_split_nonwt_pfm_row_overlap"] == 0
    assert maps["estimator_train"].shape == (1, 16)
    assert len(summary) == 4

    print("SELF-TEST: PASS")
    print(
        "Verified: strict-union construction, WT-only cross-partition overlap, "
        "deterministic PFM->extraction indexing, and exact cube-row round trip."
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run deterministic extraction-index tests and exit.")
    parser.add_argument("--code07-manifest", type=Path, default=None, help="Override Code-07 strict-split manifest path.")
    parser.add_argument("--code02-manifest", type=Path, default=None, help="Override Code-02 ESM-2 validation manifest path.")
    parser.add_argument("--pfm-index", type=Path, default=None, help="Override the label-free PFM index path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override Code-08 output directory.")
    parser.add_argument("--no-write", action="store_true", help="Run all audits without writing outputs.")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    root = project_root()
    output_dir = args.output_dir or (root / "04 Experiments" / "EXP02_GB1_ESM2")
    code07_manifest_path = args.code07_manifest or (output_dir / "gb1_strict_split_manifest.json")

    if args.code02_manifest is not None:
        code02_manifest_path = args.code02_manifest
    else:
        code02_manifest_path = locate_unique_existing(
            [
                root / "02 Data" / "PFM_Inputs" / "esm2_loading_validation_manifest.json",
                root / "02 Data" / "03_PFM_Inputs" / "GB1" / "esm2_loading_validation_manifest.json",
                root / "02 Data" / "03_PFM_Inputs" / "esm2_loading_validation_manifest.json",
            ],
            "Code-02 ESM-2 validation manifest",
            "--code02-manifest",
        )

    print("=" * 76)
    print(f"{CODE_NAME} — FROZEN ESM-2 EXTRACTION INDEX")
    print("=" * 76)
    print()

    code07_manifest, code07_sha, code07_paths = load_and_verify_code07(
        code07_manifest_path
    )
    print("Code-07 strict-split provenance gate: PASS")
    print(f"  manifest SHA256: {code07_sha}")
    print("  all Code-07 output hashes: PASS")
    for split_name in STRICT_SPLIT_KEYS:
        print(f"  {split_name:15s}: {EXPECTED_STRICT_COUNTS[split_name]:,} cubes")
    print()

    code06_manifest_path = code07_paths["code06_manifest"]
    code06_manifest, code06_sha, cube_pfm_path, manifest_pfm_path = (
        load_and_verify_code06_for_extraction(code06_manifest_path)
    )
    print("Code-06 / PFM provenance gate: PASS")
    print(f"  Code-06 manifest SHA256: {code06_sha}")
    print(f"  complete cubes: {EXPECTED_COMPLETE_CUBES:,}")
    print("  cube PFM-row artifact hash: PASS")
    print()

    if args.pfm_index is not None:
        pfm_path = args.pfm_index
        pinned_sha = str(code06_manifest.get("inputs", {}).get("pfm_index_sha256", ""))
        verify_file_hash(pfm_path, pinned_sha, "explicit PFM index")
    else:
        pfm_path = manifest_pfm_path

    pfm_sha = sha256_file(pfm_path)
    pfm = load_and_audit_pfm_index(pfm_path)
    print("Label-free Code-01 PFM input audit: PASS")
    print(f"  PFM rows: {len(pfm):,}")
    print(f"  sequence length: {EXPECTED_SEQUENCE_LENGTH}")
    print("  fitness-bearing columns present: NO")
    print()

    code02_manifest, code02_sha = load_and_verify_code02(
        code02_manifest_path, pfm_sha
    )
    print("Code-02 frozen ESM-2 contract: PASS")
    print(f"  model: {MODEL_NAME}")
    print(f"  transformer layers: {EXPECTED_NUM_LAYERS}")
    print(f"  representation layers requested downstream: 0..{EXPECTED_NUM_LAYERS}")
    print(f"  embedding dimension: {EXPECTED_EMBED_DIM}")
    print(f"  target ESM token indices: {EXPECTED_TARGET_TOKEN_INDICES}")
    print()

    strict_indices = load_strict_indices(code07_paths["split_indices"])
    cube_pfm_rows = load_pfm_cube_rows(cube_pfm_path)

    extraction, cube_extract_rows, summary, audit = build_extraction_artifacts(
        pfm=pfm,
        cube_pfm_rows=cube_pfm_rows,
        strict_indices=strict_indices,
    )

    print("Strict sequence-union audit: PASS")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.partition:15s}: "
            f"{int(row.unique_pfm_rows_including_wt):,} unique sequences "
            f"({int(row.unique_nonwt_pfm_rows):,} non-WT)"
        )
    print("  cross-partition shared non-WT PFM rows: 0")
    print(f"  only shared PFM row: WT = {WT} (row {audit['wt_pfm_row']})")
    print()

    print("Extraction-index contract: PASS")
    print(f"  exact unique sequences to run through ESM-2: {len(extraction):,}")
    print("  index order: ascending pfm_row_id")
    print("  extract_row_id contiguous: PASS")
    print("  cube -> extraction-row round trip: PASS")
    print("  mixed cubes included: NO")
    print("  fitness present: NO")
    print()

    if args.no_write:
        print("NO-WRITE MODE: all extraction-index audits passed; no files written.")
    else:
        output_paths, manifest_path = save_outputs(
            output_dir=output_dir,
            extraction=extraction,
            cube_extract_rows=cube_extract_rows,
            summary=summary,
            audit=audit,
            code07_manifest_path=code07_manifest_path,
            code07_sha=code07_sha,
            code06_manifest_path=code06_manifest_path,
            code06_sha=code06_sha,
            code02_manifest_path=code02_manifest_path,
            code02_sha=code02_sha,
            pfm_index_path=pfm_path,
            pfm_sha=pfm_sha,
        )
        print("Saved:")
        for path in output_paths:
            print(f"  {path}")
        print(f"  {manifest_path}")

    print()
    print(f"{CODE_NAME}: PASS")
    print("Exact frozen sequence set is ready for Code 09 ESM-2 extraction.")


if __name__ == "__main__":
    main()
