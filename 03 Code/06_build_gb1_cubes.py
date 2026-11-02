#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 06 — Build and audit complete WT-anchored GB1 cubes.

Purpose
-------
Construct the real biological units used by the Phase-1 ORBIT-FMIB analysis
only after the synthetic measurement gate (Code 05) has passed.

The four assayed GB1 coordinates are represented by the variant code

    V39 D40 G41 V54  ->  "VDGV"  (wild type)

For one chosen non-WT amino acid at each coordinate, a local binary cube has
2^4 = 16 vertices.  In every coordinate:

    bit 0 = WT amino acid
    bit 1 = the selected alternative amino acid

Vertex masks are ordered 0..15, with bit i controlling coordinate i in the
ordered coordinate list (39, 40, 41, 54).  This is the same convention used
by the previous ORBIT Walsh implementation: WT/0 -> -1, mutation/1 -> +1.

There are 19 alternatives at each of four positions, hence 19^4 = 130,321
candidate WT-anchored cubes.  A cube is retained only when all 16 variants
have experimentally measured GB1 fitness.

Scientific role
---------------
Code 06 does NOT run ESM-2 and does NOT train a mutual-information critic.
It constructs the exact experimental cube population from which later codes
will compute

    E_{C,S} = 2^-4 sum_x f(x) chi_S(x)

and pair those experimental coefficients with representation-level Walsh
components from frozen ESM-2.

Leakage / label boundary
------------------------
Experimental fitness is intentionally present in the cube artifact because it
is required later to construct E_{C,S}.  It must never be passed to ESM-2.
The separate Code-01 PFM index is required and re-audited to be label-free;
later extraction must use that label-free index.

Default inputs
--------------
The full GB1 table is auto-located from supported project layouts, including:
    02 Data/gb1_full_standardized.csv
    02 Data/02_Processed/GB1/gb1_full_standardized.csv
    02 Data/02_Processed/gb1_full_standardized.csv

The Code-01 PFM index is auto-located from supported layouts, including:
    02 Data/PFM_Inputs/gb1_pfm_index.csv
    02 Data/03_PFM_Inputs/GB1/gb1_pfm_index.csv
    02 Data/03_PFM_Inputs/gb1_pfm_index.csv

04 Experiments/EXP01_ORBIT_Synth/fmib_code05_validation_manifest.json

If more than one candidate for an input exists, the script refuses to guess
and requires the intended file explicitly with --full-data or --pfm-index.

Default outputs
---------------
04 Experiments/EXP02_GB1_ESM2/
    gb1_complete_cube_metadata.csv
    gb1_complete_cube_variants.npy
    gb1_complete_cube_fitness.npy
    gb1_complete_cube_source_rows.npy
    gb1_complete_cube_pfm_rows.npy
    gb1_cube_manifest.json

Usage
-----
    python "03 Code\\06_build_gb1_cubes.py" --self-test
    python "03 Code\\06_build_gb1_cubes.py"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Frozen Phase-1 cube contract
# =============================================================================

CODE_NAME = "ORBIT-FMIB CODE 06"
SCHEMA_VERSION = 1
EXPECTED_CODE05 = "ORBIT-FMIB CODE 05"

WT_VARIANT = "VDGV"
VARIABLE_POSITIONS = (39, 40, 41, 54)
D = 4
AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_AA_SET = set(AA_ALPHABET)
EXPECTED_MEASURED_VARIANTS = 149_361
EXPECTED_CANDIDATE_CUBES = 19 ** 4  # 130,321
# This is a frozen, previously reproduced ORBIT count for the exact GB1 table.
EXPECTED_COMPLETE_CUBES = 109_235
EXPECTED_VERTICES_PER_CUBE = 16
EXPECTED_HD_PROFILE = {0: 1, 1: 4, 2: 6, 3: 4, 4: 1}


# =============================================================================
# Paths / provenance helpers
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


def locate_unique_existing(
    paths: Sequence[Path],
    label: str,
    override_flag: str,
) -> Path:
    """Locate exactly one existing artifact; never silently choose among duplicates."""
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


def verify_manifest_file_hashes(manifest_path: Path, manifest: Dict) -> None:
    """Verify files pinned by a manifest when a file_sha256 mapping is present."""
    hashes = manifest.get("file_sha256", {})
    if not isinstance(hashes, dict):
        raise RuntimeError("Upstream manifest file_sha256 field is not a mapping.")
    for name, expected in hashes.items():
        path = manifest_path.parent / str(name)
        if not path.exists():
            raise FileNotFoundError(f"Upstream manifest-pinned file is missing: {path}")
        observed = sha256_file(path)
        if observed.lower() != str(expected).lower():
            raise RuntimeError(
                "Upstream artifact SHA256 mismatch.\n"
                f"File: {path}\nExpected: {expected}\nObserved: {observed}"
            )


def verify_code05_gate(path: Path) -> Tuple[Dict, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Code-05 validation manifest not found: {path}\n"
            "Run Code 05 successfully before constructing biological GB1 cubes."
        )
    manifest = load_json(path)
    if manifest.get("code_name") != EXPECTED_CODE05:
        raise RuntimeError(
            f"Unexpected Code-05 manifest code_name: {manifest.get('code_name')!r}"
        )
    if not bool(manifest.get("overall_pass", False)):
        raise RuntimeError(
            "Code-05 synthetic measurement gate is not recorded as PASS. "
            "GB1 construction is blocked."
        )
    acceptance = manifest.get("acceptance_results", {})
    if acceptance and not bool(acceptance.get("overall_pass", False)):
        raise RuntimeError("Code-05 acceptance_results.overall_pass is not true.")
    verify_manifest_file_hashes(path, manifest)
    return manifest, sha256_file(path)


# =============================================================================
# Input audits
# =============================================================================


def normalize_variant_series(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise RuntimeError(f"{label}: missing variant values detected.")
    out = series.astype(str).str.strip().str.upper()
    if not (out.str.len() == D).all():
        observed = sorted(set(out.str.len().tolist()))
        raise RuntimeError(f"{label}: variants must all have length 4; observed {observed}.")
    observed_alphabet = set("".join(out.tolist()))
    if not observed_alphabet.issubset(EXPECTED_AA_SET):
        raise RuntimeError(
            f"{label}: noncanonical amino-acid symbols detected: "
            f"{sorted(observed_alphabet - EXPECTED_AA_SET)}"
        )
    return out


def load_and_audit_full(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GB1 full standardized table not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {"variant", "fitness"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"GB1 full table missing required columns: {sorted(missing)}")

    df = df.copy()
    df["variant"] = normalize_variant_series(df["variant"], "GB1 full table")
    df["fitness"] = pd.to_numeric(df["fitness"], errors="raise").astype(np.float64)

    if len(df) != EXPECTED_MEASURED_VARIANTS:
        raise RuntimeError(
            "Unexpected measured GB1 variant count.\n"
            f"Expected: {EXPECTED_MEASURED_VARIANTS:,}\nObserved: {len(df):,}"
        )
    if df["variant"].duplicated().any():
        raise RuntimeError("Duplicate four-site variants detected in GB1 full table.")
    if not np.isfinite(df["fitness"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Missing or non-finite GB1 fitness values detected.")

    observed_alphabet = set("".join(df["variant"].tolist()))
    if observed_alphabet != EXPECTED_AA_SET:
        raise RuntimeError(
            "Unexpected GB1 amino-acid alphabet.\n"
            f"Expected all 20 canonical amino acids; observed {sorted(observed_alphabet)}"
        )
    wt_count = int((df["variant"] == WT_VARIANT).sum())
    if wt_count != 1:
        raise RuntimeError(f"Expected exactly one WT {WT_VARIANT}; found {wt_count}.")

    return df.reset_index(drop=True)


def load_and_audit_pfm_index(path: Path, full: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Audit the label-free Code-01 PFM index and one-to-one variant compatibility."""
    if not path.exists():
        raise FileNotFoundError(f"Code-01 PFM index not found: {path}")
    pfm = pd.read_csv(path, low_memory=False)

    if "fitness" in pfm.columns:
        raise RuntimeError(
            "LABEL BOUNDARY FAILURE: Code-01 PFM index contains a fitness column. "
            "The frozen foundation-model input must remain label-free."
        )
    required = {"variant", "sequence"}
    missing = required - set(pfm.columns)
    if missing:
        raise RuntimeError(f"PFM index missing required columns: {sorted(missing)}")

    pfm = pfm.copy()
    pfm["variant"] = normalize_variant_series(pfm["variant"], "PFM index")
    if len(pfm) != EXPECTED_MEASURED_VARIANTS:
        raise RuntimeError(
            f"Unexpected PFM index row count: {len(pfm):,}; "
            f"expected {EXPECTED_MEASURED_VARIANTS:,}."
        )
    if pfm["variant"].duplicated().any():
        raise RuntimeError("Duplicate variants detected in PFM index.")
    if pfm["sequence"].isna().any() or pfm["sequence"].astype(str).duplicated().any():
        raise RuntimeError("PFM index sequences must be present and unique.")

    full_variants = set(full["variant"].tolist())
    pfm_variants = set(pfm["variant"].tolist())
    if full_variants != pfm_variants:
        raise RuntimeError(
            "PFM/full-table variant sets differ; biological cubes cannot be safely mapped "
            "to frozen model inputs."
        )

    if "pfm_row_id" in pfm.columns:
        row_ids = pd.to_numeric(pfm["pfm_row_id"], errors="raise").to_numpy(dtype=np.int64)
        expected = np.arange(len(pfm), dtype=np.int64)
        if not np.array_equal(row_ids, expected):
            raise RuntimeError("PFM row IDs are not the expected contiguous deterministic IDs.")
    else:
        # Compatibility fallback: the physical row order becomes the deterministic PFM row ID.
        pfm.insert(0, "pfm_row_id", np.arange(len(pfm), dtype=np.int64))

    return pfm.reset_index(drop=True), sha256_file(path)


# =============================================================================
# Cube construction
# =============================================================================


def get_alternatives() -> List[List[str]]:
    all_alternatives: List[List[str]] = []
    for coordinate, wt_aa in zip(VARIABLE_POSITIONS, WT_VARIANT):
        alternatives = [aa for aa in AA_ALPHABET if aa != wt_aa]
        if len(alternatives) != 19 or wt_aa in alternatives:
            raise RuntimeError(
                f"Coordinate {coordinate}: alternative-amino-acid construction failed."
            )
        all_alternatives.append(alternatives)
    return all_alternatives


def make_cube_vertices(alternatives: Sequence[str]) -> List[str]:
    if len(alternatives) != D:
        raise ValueError("Need exactly four selected alternative amino acids.")
    for i, aa in enumerate(alternatives):
        if aa not in EXPECTED_AA_SET:
            raise ValueError(f"Noncanonical alternative amino acid: {aa!r}")
        if aa == WT_VARIANT[i]:
            raise ValueError(
                f"Alternative at coordinate {VARIABLE_POSITIONS[i]} equals WT amino acid {aa}."
            )

    vertices: List[str] = []
    for mask in range(1 << D):
        chars = list(WT_VARIANT)
        for position in range(D):
            if (mask >> position) & 1:
                chars[position] = alternatives[position]
        vertices.append("".join(chars))
    return vertices


def vertex_hamming_distances() -> np.ndarray:
    return np.asarray([int(mask).bit_count() for mask in range(1 << D)], dtype=np.int8)


def enumerate_complete_cubes(
    full: pd.DataFrame,
    pfm: pd.DataFrame,
    candidate_alternatives: Optional[Iterable[Tuple[str, str, str, str]]] = None,
) -> Dict:
    """Enumerate candidate cubes and retain only fully measured 16-vertex cubes."""
    source_row_lookup = {v: i for i, v in enumerate(full["variant"].tolist())}
    fitness_lookup = dict(zip(full["variant"].tolist(), full["fitness"].tolist()))
    pfm_row_lookup = dict(
        zip(pfm["variant"].tolist(), pfm["pfm_row_id"].astype(np.int64).tolist())
    )

    if len(source_row_lookup) != len(full) or len(pfm_row_lookup) != len(pfm):
        raise RuntimeError("Variant lookup construction is not one-to-one.")

    if candidate_alternatives is None:
        candidate_iter = product(*get_alternatives())
    else:
        candidate_iter = candidate_alternatives

    metadata_rows: List[Dict] = []
    variant_rows: List[List[str]] = []
    fitness_rows: List[List[float]] = []
    source_index_rows: List[List[int]] = []
    pfm_index_rows: List[List[int]] = []
    incomplete_examples: List[Dict] = []
    candidate_count = 0

    for alternatives_raw in candidate_iter:
        alternatives = tuple(str(x).upper() for x in alternatives_raw)
        candidate_count += 1
        vertices = make_cube_vertices(alternatives)
        missing = [v for v in vertices if v not in source_row_lookup]
        if missing:
            if len(incomplete_examples) < 5:
                incomplete_examples.append(
                    {"alternatives": list(alternatives), "missing_variants": missing}
                )
            continue

        # PFM mapping must exist independently for every measured vertex.
        missing_pfm = [v for v in vertices if v not in pfm_row_lookup]
        if missing_pfm:
            raise RuntimeError(
                "A complete experimental cube contains variants absent from the label-free PFM index: "
                f"{missing_pfm[:5]}"
            )

        cube_id = len(metadata_rows)
        metadata_rows.append(
            {
                "cube_id": cube_id,
                "wt": WT_VARIANT,
                "alt_39": alternatives[0],
                "alt_40": alternatives[1],
                "alt_41": alternatives[2],
                "alt_54": alternatives[3],
                "far_corner": vertices[15],
            }
        )
        variant_rows.append(vertices)
        fitness_rows.append([float(fitness_lookup[v]) for v in vertices])
        source_index_rows.append([int(source_row_lookup[v]) for v in vertices])
        pfm_index_rows.append([int(pfm_row_lookup[v]) for v in vertices])

    n_complete = len(metadata_rows)
    return {
        "candidate_count": candidate_count,
        "complete_count": n_complete,
        "metadata": pd.DataFrame(metadata_rows),
        "variants": np.asarray(variant_rows, dtype="S4").reshape(n_complete, 16),
        "fitness": np.asarray(fitness_rows, dtype=np.float64).reshape(n_complete, 16),
        "source_rows": np.asarray(source_index_rows, dtype=np.int32).reshape(n_complete, 16),
        "pfm_rows": np.asarray(pfm_index_rows, dtype=np.int32).reshape(n_complete, 16),
        "incomplete_examples": incomplete_examples,
    }


# =============================================================================
# Result audit
# =============================================================================


def audit_cube_result(result: Dict, full: pd.DataFrame, pfm: pd.DataFrame, real_run: bool) -> Dict:
    n = int(result["complete_count"])
    variants = result["variants"]
    fitness = result["fitness"]
    source_rows = result["source_rows"]
    pfm_rows = result["pfm_rows"]
    metadata = result["metadata"]

    if real_run and int(result["candidate_count"]) != EXPECTED_CANDIDATE_CUBES:
        raise RuntimeError(
            f"Wrong candidate-cube count: expected {EXPECTED_CANDIDATE_CUBES:,}, "
            f"observed {result['candidate_count']:,}."
        )
    if real_run and n != EXPECTED_COMPLETE_CUBES:
        raise RuntimeError(
            "Unexpected number of complete GB1 cubes for the frozen landscape.\n"
            f"Expected previously reproduced ORBIT count: {EXPECTED_COMPLETE_CUBES:,}\n"
            f"Observed: {n:,}"
        )

    expected_shape = (n, EXPECTED_VERTICES_PER_CUBE)
    for name, array in (
        ("variants", variants),
        ("fitness", fitness),
        ("source_rows", source_rows),
        ("pfm_rows", pfm_rows),
    ):
        if array.shape != expected_shape:
            raise RuntimeError(f"{name} has wrong shape {array.shape}; expected {expected_shape}.")

    if len(metadata) != n:
        raise RuntimeError("Cube metadata row count does not match complete cube count.")
    if n and not np.array_equal(metadata["cube_id"].to_numpy(), np.arange(n)):
        raise RuntimeError("cube_id is not contiguous and deterministic.")
    if not np.isfinite(fitness).all():
        raise RuntimeError("Non-finite fitness values in complete-cube matrix.")
    if n and not np.all(variants[:, 0] == WT_VARIANT.encode("ascii")):
        raise RuntimeError("Vertex mask 0000 is not WT in every complete cube.")

    # Every cube must contain 16 unique vertices with the exact 1/4/6/4/1 HD pattern.
    hd = vertex_hamming_distances()
    hd_profile = {k: int(np.sum(hd == k)) for k in range(D + 1)}
    if hd_profile != EXPECTED_HD_PROFILE:
        raise RuntimeError(f"Internal vertex Hamming profile mismatch: {hd_profile}")

    # Audit exact variant/row/fitness consistency in vectorized form.
    full_variants = full["variant"].astype(str).to_numpy(dtype="U4")
    full_fitness = full["fitness"].to_numpy(dtype=np.float64)
    pfm_variants = pfm["variant"].astype(str).to_numpy(dtype="U4")

    if n:
        decoded = variants.astype("U4")
        if not np.array_equal(full_variants[source_rows], decoded):
            raise RuntimeError("Stored source-row indices do not reproduce cube variants.")
        if not np.allclose(full_fitness[source_rows], fitness, atol=0.0, rtol=0.0):
            raise RuntimeError("Stored cube fitness does not exactly reproduce the source table.")
        if not np.array_equal(pfm_variants[pfm_rows], decoded):
            raise RuntimeError("Stored PFM-row indices do not reproduce cube variants.")

        # Decode each vertex against the metadata-selected alternatives.
        # Chunked to keep memory bounded while still auditing every cube.
        masks = np.arange(16, dtype=np.uint8)
        for start in range(0, n, 10_000):
            stop = min(n, start + 10_000)
            chunk_meta = metadata.iloc[start:stop]
            observed = decoded[start:stop]
            expected = np.empty_like(observed)
            for local_i, row in enumerate(chunk_meta.itertuples(index=False)):
                alts = (row.alt_39, row.alt_40, row.alt_41, row.alt_54)
                expected[local_i] = np.asarray(make_cube_vertices(alts), dtype="U4")
            if not np.array_equal(observed, expected):
                raise RuntimeError(
                    f"Cube vertex ordering/content mismatch in rows {start}:{stop}."
                )

    return {
        "candidate_cubes": int(result["candidate_count"]),
        "complete_cubes": n,
        "incomplete_cubes": int(result["candidate_count"] - n),
        "vertices_per_cube": EXPECTED_VERTICES_PER_CUBE,
        "vertex_hd_profile": {str(k): v for k, v in hd_profile.items()},
        "all_fitness_finite": True,
        "source_row_mapping": "PASS",
        "pfm_row_mapping": "PASS",
        "vertex_ordering": "PASS",
    }


# =============================================================================
# Saving
# =============================================================================


def save_outputs(
    output_dir: Path,
    result: Dict,
    audit: Dict,
    full_path: Path,
    full_sha: str,
    pfm_path: Path,
    pfm_sha: str,
    code05_path: Path,
    code05_sha: str,
) -> Tuple[List[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "gb1_complete_cube_metadata.csv"
    variants_path = output_dir / "gb1_complete_cube_variants.npy"
    fitness_path = output_dir / "gb1_complete_cube_fitness.npy"
    source_rows_path = output_dir / "gb1_complete_cube_source_rows.npy"
    pfm_rows_path = output_dir / "gb1_complete_cube_pfm_rows.npy"
    manifest_path = output_dir / "gb1_cube_manifest.json"

    result["metadata"].to_csv(metadata_path, index=False, lineterminator="\n")
    np.save(variants_path, result["variants"], allow_pickle=False)
    np.save(fitness_path, result["fitness"], allow_pickle=False)
    np.save(source_rows_path, result["source_rows"], allow_pickle=False)
    np.save(pfm_rows_path, result["pfm_rows"], allow_pickle=False)

    output_paths = [
        metadata_path,
        variants_path,
        fitness_path,
        source_rows_path,
        pfm_rows_path,
    ]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": "Construct and audit complete WT-anchored 4D GB1 cubes for Phase-1 ORBIT-FMIB.",
        "upstream_gate": {
            "code05_manifest_path": str(code05_path.resolve()),
            "code05_manifest_sha256": code05_sha,
            "code05_overall_pass": True,
        },
        "inputs": {
            "gb1_full_path": str(full_path.resolve()),
            "gb1_full_sha256": full_sha,
            "pfm_index_path": str(pfm_path.resolve()),
            "pfm_index_sha256": pfm_sha,
            "pfm_index_fitness_column_present": False,
        },
        "cube_definition": {
            "wt_variant": WT_VARIANT,
            "biological_positions": list(VARIABLE_POSITIONS),
            "dimensions": D,
            "bit_convention": "0=WT, 1=selected alternative",
            "walsh_sign_convention": "bit 0 / WT -> -1; bit 1 / mutation -> +1",
            "vertex_mask_order": list(range(16)),
            "vertex_hamming_distance": vertex_hamming_distances().astype(int).tolist(),
            "alternatives_per_position": 19,
            "candidate_cubes_expected": EXPECTED_CANDIDATE_CUBES,
            "complete_cubes_expected_frozen_reference": EXPECTED_COMPLETE_CUBES,
        },
        "audit": audit,
        "label_boundary": {
            "experimental_cube_artifact_contains_fitness": True,
            "pfm_input_contains_fitness": False,
            "rule": "Fitness is used only for experimental Walsh targets E_C,S; it must not be an ESM-2 input.",
        },
        "incomplete_examples": result["incomplete_examples"],
        "outputs": {
            "metadata": metadata_path.name,
            "variants": variants_path.name,
            "fitness": fitness_path.name,
            "source_rows": source_rows_path.name,
            "pfm_rows": pfm_rows_path.name,
        },
        "overall_pass": True,
    }
    manifest["file_sha256"] = {p.name: sha256_file(p) for p in output_paths}
    write_json(manifest_path, manifest)
    return output_paths, manifest_path


# =============================================================================
# Self-test
# =============================================================================


def run_self_test() -> None:
    alternatives = ("A", "A", "A", "A")
    vertices = make_cube_vertices(alternatives)
    assert len(vertices) == 16
    assert len(set(vertices)) == 16
    assert vertices[0] == "VDGV"
    assert vertices[1] == "ADGV"   # bit 0 -> position 39
    assert vertices[2] == "VAGV"   # bit 1 -> position 40
    assert vertices[4] == "VDAV"   # bit 2 -> position 41
    assert vertices[8] == "VDGA"   # bit 3 -> position 54
    assert vertices[15] == "AAAA"

    hd = vertex_hamming_distances()
    profile = {k: int(np.sum(hd == k)) for k in range(5)}
    assert profile == EXPECTED_HD_PROFILE

    # Build a tiny source/PFM universe containing one complete cube plus one
    # extra candidate that must be incomplete.  This exercises membership,
    # row-index mapping, label isolation, and exact fitness transfer.
    tiny_variants = vertices.copy()
    tiny_full = pd.DataFrame(
        {
            "variant": tiny_variants,
            "fitness": np.linspace(-1.0, 1.0, 16, dtype=np.float64),
        }
    )
    tiny_pfm = pd.DataFrame(
        {
            "pfm_row_id": np.arange(16, dtype=np.int64),
            "variant": tiny_variants,
            "sequence": [f"SEQ_{v}" for v in tiny_variants],
        }
    )
    candidates = [alternatives, ("C", "A", "A", "A")]
    result = enumerate_complete_cubes(tiny_full, tiny_pfm, candidates)
    assert result["candidate_count"] == 2
    assert result["complete_count"] == 1
    assert result["variants"].shape == (1, 16)
    assert result["fitness"].shape == (1, 16)
    assert np.array_equal(result["source_rows"][0], np.arange(16, dtype=np.int32))
    assert np.array_equal(result["pfm_rows"][0], np.arange(16, dtype=np.int32))
    audit_cube_result(result, tiny_full, tiny_pfm, real_run=False)

    print("SELF-TEST: PASS")
    print("Verified: 4-bit vertex ordering, 1/4/6/4/1 Hamming profile, complete/incomplete cube logic, exact source/PFM row mapping, and fitness transfer.")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run deterministic internal cube tests and exit.")
    parser.add_argument("--full-data", type=Path, default=None, help="Override gb1_full_standardized.csv path.")
    parser.add_argument("--pfm-index", type=Path, default=None, help="Override Code-01 label-free PFM index path.")
    parser.add_argument("--code05-manifest", type=Path, default=None, help="Override Code-05 validation manifest path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override Code-06 output directory.")
    parser.add_argument("--no-write", action="store_true", help="Run all real-data audits without writing outputs.")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    root = project_root()

    if args.full_data is not None:
        full_path = args.full_data
    else:
        full_path = locate_unique_existing(
            [
                root / "02 Data" / "gb1_full_standardized.csv",
                root / "02 Data" / "02_Processed" / "GB1" / "gb1_full_standardized.csv",
                root / "02 Data" / "02_Processed" / "gb1_full_standardized.csv",
            ],
            "GB1 full standardized table",
            "--full-data",
        )

    if args.pfm_index is not None:
        pfm_path = args.pfm_index
    else:
        pfm_path = locate_unique_existing(
            [
                root / "02 Data" / "PFM_Inputs" / "gb1_pfm_index.csv",
                root / "02 Data" / "03_PFM_Inputs" / "GB1" / "gb1_pfm_index.csv",
                root / "02 Data" / "03_PFM_Inputs" / "gb1_pfm_index.csv",
            ],
            "Code-01 PFM index",
            "--pfm-index",
        )

    code05_path = args.code05_manifest or (
        root / "04 Experiments" / "EXP01_ORBIT_Synth" / "fmib_code05_validation_manifest.json"
    )
    output_dir = args.output_dir or (root / "04 Experiments" / "EXP02_GB1_ESM2")

    print("=" * 76)
    print(f"{CODE_NAME} — COMPLETE WT-ANCHORED GB1 CUBES")
    print("=" * 76)
    print()

    code05_manifest, code05_sha = verify_code05_gate(code05_path)
    print("Synthetic measurement gate: PASS")
    print(f"  Code-05 manifest SHA256: {code05_sha}")
    print("  Code-05 overall_pass: PASS")
    print()

    full = load_and_audit_full(full_path)
    full_sha = sha256_file(full_path)
    print("GB1 experimental input audit: PASS")
    print(f"  measured variants: {len(full):,}")
    print(f"  WT: {WT_VARIANT}")
    print(f"  full-data SHA256: {full_sha}")
    print()

    pfm, pfm_sha = load_and_audit_pfm_index(pfm_path, full)
    print("Code-01 label-free PFM compatibility audit: PASS")
    print(f"  PFM rows: {len(pfm):,}")
    print("  fitness column present: NO")
    print("  full-table / PFM variant-set identity: PASS")
    print(f"  PFM-index SHA256: {pfm_sha}")
    print()

    print("Enumerating 19^4 WT-anchored candidate cubes ...")
    result = enumerate_complete_cubes(full, pfm)
    audit = audit_cube_result(result, full, pfm, real_run=True)

    print("GB1 complete-cube audit: PASS")
    print(f"  candidate cubes: {audit['candidate_cubes']:,}")
    print(f"  complete cubes: {audit['complete_cubes']:,}")
    print(f"  incomplete cubes: {audit['incomplete_cubes']:,}")
    print(f"  vertices per complete cube: {audit['vertices_per_cube']}")
    print("  vertex HD profile: 1 WT / 4 singles / 6 doubles / 4 triples / 1 quadruple")
    print("  source-row mapping: PASS")
    print("  PFM-row mapping: PASS")
    print("  exact fitness transfer: PASS")
    print("  label boundary (PFM index has no fitness): PASS")
    print()

    if args.no_write:
        print("NO-WRITE MODE: all real-data cube audits passed; no files written.")
    else:
        output_paths, manifest_path = save_outputs(
            output_dir=output_dir,
            result=result,
            audit=audit,
            full_path=full_path,
            full_sha=full_sha,
            pfm_path=pfm_path,
            pfm_sha=pfm_sha,
            code05_path=code05_path,
            code05_sha=code05_sha,
        )
        print("Saved:")
        for path in output_paths:
            print(f"  {path}")
        print(f"  {manifest_path}")

    print()
    print(f"{CODE_NAME}: PASS")
    print("Real complete GB1 cube population is frozen for downstream strict splitting and ESM-2 mapping.")


if __name__ == "__main__":
    main()
