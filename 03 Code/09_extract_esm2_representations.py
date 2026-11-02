#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 09 — Frozen ESM-2 site-representation extraction.

Purpose
-------
Run the frozen Phase-1 model `esm2_t30_150M_UR50D` on the exact label-free
sequence union frozen by Code 08 and save layerwise hidden states only at the
four GB1 assayed residues (39, 40, 41, 54).

For each representation layer l = 0..30, the saved array is

    H_l.shape = [N_extract, 4, 640]

with site order (39, 40, 41, 54).  Disk dtype is float16; ESM-2 computation is
performed in its normal float32 model dtype (no autocast).

Scientific boundary
-------------------
Code 09 NEVER reads experimental fitness.  It does not compute Walsh
coefficients or mutual information.  It is a frozen-model representation
extraction stage only.

Resume behavior
---------------
The extraction is restartable.  A state file records the next committed row.
At each checkpoint, all layer memmaps are flushed before the state advances.
A restart therefore safely rewrites at most the uncommitted tail.

Default input
-------------
04 Experiments/EXP02_GB1_ESM2/gb1_esm2_extraction_manifest.json

Default output
--------------
04 Experiments/EXP02_GB1_ESM2/ESM2_Site_Reps/
    esm2_layer_00_sites.npy
    ...
    esm2_layer_30_sites.npy
    esm2_extraction_state.json
    esm2_extraction_manifest.json

Typical HPC cluster run
------------------
    python "03 Code/09_extract_esm2_representations.py" --device cuda --batch-size 16
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Frozen Phase-1 contract
# =============================================================================

CODE_NAME = "ORBIT-FMIB CODE 09"
SCHEMA_VERSION = 1

EXPECTED_CODE08 = "ORBIT-FMIB CODE 08"
EXPECTED_CODE02_FILE = "02_validate_esm2_loading.py"

MODEL_NAME = "esm2_t30_150M_UR50D"
EXPECTED_NUM_LAYERS = 30
REPR_LAYERS = tuple(range(EXPECTED_NUM_LAYERS + 1))  # 0..30
EXPECTED_EMBED_DIM = 640

POSITIONS = (39, 40, 41, 54)
TARGET_TOKEN_INDICES = (39, 40, 41, 54)
EXPECTED_SEQUENCE_LENGTH = 265
EXPECTED_TOKEN_LENGTH = 267
EXPECTED_N_EXTRACT = 13_150

OUTPUT_DTYPE = np.float16
OUTPUT_DTYPE_NAME = "float16"
SITE_SHAPE = (len(POSITIONS), EXPECTED_EMBED_DIM)

EXPECTED_PARAMETER_COUNT = 148_140_154


# =============================================================================
# Helpers
# =============================================================================


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Path: {path}\nExpected: {expected}\nObserved: {observed}"
        )


def layer_filename(layer: int) -> str:
    if layer not in REPR_LAYERS:
        raise ValueError(f"Invalid ESM-2 representation layer: {layer}")
    return f"esm2_layer_{layer:02d}_sites.npy"


def import_esm_stack():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Code 09.") from exc
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "The official fair-esm package is required.\n"
            "Install with: pip install fair-esm"
        ) from exc
    return torch, esm


def choose_device(torch_module: Any, requested: str):
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch_module.device("cuda")
    if requested == "auto":
        return torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
    raise ValueError(requested)


# =============================================================================
# Code-08 / Code-02 provenance
# =============================================================================


def load_and_verify_code08(
    manifest_path: Path,
) -> Tuple[Dict, str, Path, str, Path, str]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Code-08 manifest not found: {manifest_path}\n"
            "Code 08 must pass before Code 09."
        )

    manifest = load_json(manifest_path)
    if manifest.get("code_name") != EXPECTED_CODE08:
        raise RuntimeError(
            f"Unexpected Code-08 manifest code_name: {manifest.get('code_name')!r}"
        )
    if not bool(manifest.get("overall_pass", False)):
        raise RuntimeError("Code-08 manifest does not record overall_pass=true.")

    audit = manifest.get("audit", {})
    if not bool(audit.get("overall_pass", False)):
        raise RuntimeError("Code-08 audit.overall_pass is not true.")
    n_extract = int(audit.get("union_unique_sequences", -1))
    if n_extract != EXPECTED_N_EXTRACT:
        raise RuntimeError(
            "Frozen extraction-union size changed.\n"
            f"Expected: {EXPECTED_N_EXTRACT:,}\nObserved: {n_extract:,}"
        )
    if int(audit.get("cross_split_nonwt_pfm_row_overlap", -1)) != 0:
        raise RuntimeError("Code-08 manifest does not certify zero non-WT overlap.")
    if bool(audit.get("fitness_present_in_extraction_index", True)):
        raise RuntimeError("Code-08 manifest indicates fitness in extraction index.")
    if bool(audit.get("mixed_cubes_included", True)):
        raise RuntimeError("Code-08 manifest indicates mixed cubes were included.")

    outputs = manifest.get("outputs", {})
    hashes = manifest.get("file_sha256", {})
    index_name = outputs.get("extraction_index")
    if not index_name:
        raise RuntimeError("Code-08 manifest is missing outputs.extraction_index.")
    index_path = manifest_path.parent / str(index_name)
    index_sha = hashes.get(index_path.name)
    if not index_sha:
        raise RuntimeError("Code-08 manifest is missing extraction-index SHA256.")
    verify_file_hash(index_path, str(index_sha), "Code-08 extraction index")

    esm_contract = manifest.get("esm2_contract", {})
    if esm_contract.get("model_name") != MODEL_NAME:
        raise RuntimeError("Code-08 ESM-2 model contract mismatch.")
    if int(esm_contract.get("transformer_layers", -1)) != EXPECTED_NUM_LAYERS:
        raise RuntimeError("Code-08 ESM-2 depth contract mismatch.")
    if int(esm_contract.get("embedding_dim", -1)) != EXPECTED_EMBED_DIM:
        raise RuntimeError("Code-08 ESM-2 width contract mismatch.")
    if tuple(int(x) for x in esm_contract.get("biological_positions_1based", [])) != POSITIONS:
        raise RuntimeError("Code-08 biological-position contract mismatch.")
    if tuple(int(x) for x in esm_contract.get("esm_token_indices", [])) != TARGET_TOKEN_INDICES:
        raise RuntimeError("Code-08 target-token contract mismatch.")
    if int(esm_contract.get("full_sequence_length", -1)) != EXPECTED_SEQUENCE_LENGTH:
        raise RuntimeError("Code-08 sequence-length contract mismatch.")
    if not bool(esm_contract.get("model_frozen", False)):
        raise RuntimeError("Code-08 does not certify a frozen ESM-2 model.")
    if not bool(esm_contract.get("model_eval_mode", False)):
        raise RuntimeError("Code-08 does not certify ESM-2 eval mode.")

    upstream = manifest.get("upstream", {})
    code02_path_raw = upstream.get("code02_manifest_path")
    code02_sha = str(upstream.get("code02_manifest_sha256", ""))
    if not code02_path_raw or not code02_sha:
        raise RuntimeError("Code-08 manifest does not pin Code-02 provenance.")

    code02_path = Path(str(code02_path_raw))
    if not code02_path.exists():
        root = project_root()
        candidates = [
            root / "02 Data" / "PFM_Inputs" / "esm2_loading_validation_manifest.json",
            root / "02 Data" / "03_PFM_Inputs" / "GB1" / "esm2_loading_validation_manifest.json",
            root / "02 Data" / "03_PFM_Inputs" / "esm2_loading_validation_manifest.json",
        ]
        existing = [p for p in candidates if p.exists()]
        if len(existing) != 1:
            raise FileNotFoundError(
                "Pinned Code-02 manifest is unavailable and relocation was ambiguous."
            )
        code02_path = existing[0]
    verify_file_hash(code02_path, code02_sha, "Code-02 manifest pinned by Code 08")

    return (
        manifest,
        sha256_file(manifest_path),
        index_path,
        str(index_sha),
        code02_path,
        code02_sha,
    )


def verify_code02_contract(manifest_path: Path, expected_sha: str) -> Dict:
    verify_file_hash(manifest_path, expected_sha, "Code-02 manifest")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise RuntimeError("Code-02 status is not PASS.")
    if manifest.get("code") != EXPECTED_CODE02_FILE:
        raise RuntimeError("Unexpected Code-02 code field.")
    if manifest.get("model_name") != MODEL_NAME:
        raise RuntimeError("Code-02 model mismatch.")

    arch = manifest.get("architecture", {})
    if int(arch.get("num_layers", -1)) != EXPECTED_NUM_LAYERS:
        raise RuntimeError("Code-02 transformer-depth mismatch.")
    if int(arch.get("embed_dim", -1)) != EXPECTED_EMBED_DIM:
        raise RuntimeError("Code-02 embedding-width mismatch.")
    if bool(arch.get("training_flag", True)):
        raise RuntimeError("Code-02 says model was in training mode.")
    if not bool(arch.get("all_parameters_frozen", False)):
        raise RuntimeError("Code-02 does not certify frozen parameters.")

    forward = manifest.get("forward_pass", {})
    tokens = tuple(int(x) for x in forward.get("target_token_indices", []))
    if tokens != TARGET_TOKEN_INDICES:
        raise RuntimeError("Code-02 target-token mapping mismatch.")

    return manifest


def load_extraction_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    forbidden = [c for c in df.columns if "fitness" in str(c).lower()]
    if forbidden:
        raise RuntimeError(
            "LABEL BOUNDARY FAILURE: extraction index contains fitness-bearing "
            f"columns: {forbidden}"
        )

    required = {
        "extract_row_id",
        "pfm_row_id",
        "variant",
        "sequence",
        "strict_membership",
        "sequence_length",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Code-08 extraction index missing columns: {sorted(missing)}"
        )

    if len(df) != EXPECTED_N_EXTRACT:
        raise RuntimeError(
            f"Expected {EXPECTED_N_EXTRACT:,} extraction rows; observed {len(df):,}."
        )

    row_id = pd.to_numeric(df["extract_row_id"], errors="raise").to_numpy(np.int64)
    if not np.array_equal(row_id, np.arange(len(df), dtype=np.int64)):
        raise RuntimeError("extract_row_id is not contiguous 0..N-1.")

    if df["sequence"].astype(str).duplicated().any():
        raise RuntimeError("Extraction index contains duplicate full protein sequences.")

    lengths = df["sequence"].astype(str).str.len().to_numpy(np.int64)
    if not np.all(lengths == EXPECTED_SEQUENCE_LENGTH):
        raise RuntimeError("Extraction index contains unexpected sequence lengths.")

    recorded_lengths = pd.to_numeric(
        df["sequence_length"], errors="raise"
    ).to_numpy(np.int64)
    if not np.array_equal(recorded_lengths, lengths):
        raise RuntimeError("sequence_length column disagrees with actual sequence text.")

    if (df["strict_membership"].astype(str) == "shared_wt").sum() != 1:
        raise RuntimeError("Expected exactly one shared_wt extraction row.")

    return df.reset_index(drop=True)


# =============================================================================
# Model audit
# =============================================================================


def load_and_audit_model(esm_module: Any, device: Any):
    if not hasattr(esm_module.pretrained, MODEL_NAME):
        raise RuntimeError(
            f"Installed fair-esm does not expose pretrained.{MODEL_NAME}."
        )

    loader = getattr(esm_module.pretrained, MODEL_NAME)
    model, alphabet = loader()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layers = getattr(model, "layers", None)
    if layers is None or len(layers) != EXPECTED_NUM_LAYERS:
        raise RuntimeError("Loaded ESM-2 transformer depth is not 30.")
    if int(getattr(model, "embed_dim", -1)) != EXPECTED_EMBED_DIM:
        raise RuntimeError("Loaded ESM-2 embedding width is not 640.")
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Unexpected ESM-2 parameter count: {parameter_count:,}."
        )
    if model.training:
        raise RuntimeError("ESM-2 remains in training mode.")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("At least one ESM-2 parameter still requires gradients.")

    model = model.to(device)
    return model, alphabet, parameter_count


# =============================================================================
# Restartable storage
# =============================================================================


def run_fingerprint(
    code08_sha: str,
    code02_sha: str,
    extraction_index_sha: str,
    batch_size: int,
    device_type: str,
) -> str:
    payload = {
        "code08_manifest_sha256": code08_sha,
        "code02_manifest_sha256": code02_sha,
        "extraction_index_sha256": extraction_index_sha,
        "model_name": MODEL_NAME,
        "representation_layers": list(REPR_LAYERS),
        "positions": list(POSITIONS),
        "token_indices": list(TARGET_TOKEN_INDICES),
        "n_extract": EXPECTED_N_EXTRACT,
        "site_shape": list(SITE_SHAPE),
        "output_dtype": OUTPUT_DTYPE_NAME,
        "batch_size": int(batch_size),
        "compute_device_type": str(device_type),
        "autocast": False,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def expected_array_shape() -> Tuple[int, int, int]:
    return (EXPECTED_N_EXTRACT, len(POSITIONS), EXPECTED_EMBED_DIM)


def remove_existing_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for layer in REPR_LAYERS:
        p = output_dir / layer_filename(layer)
        if p.exists():
            p.unlink()
    for name in ("esm2_extraction_state.json", "esm2_extraction_manifest.json"):
        p = output_dir / name
        if p.exists():
            p.unlink()


def open_layer_arrays(
    output_dir: Path,
    resume: bool,
) -> Dict[int, np.memmap]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: Dict[int, np.memmap] = {}
    shape = expected_array_shape()

    for layer in REPR_LAYERS:
        path = output_dir / layer_filename(layer)
        if resume:
            if not path.exists():
                raise FileNotFoundError(
                    f"Resume state exists but layer file is missing: {path}"
                )
            arr = np.lib.format.open_memmap(path, mode="r+")
            if arr.shape != shape:
                raise RuntimeError(
                    f"Resume layer {layer} shape mismatch: {arr.shape} vs {shape}."
                )
            if arr.dtype != OUTPUT_DTYPE:
                raise RuntimeError(
                    f"Resume layer {layer} dtype mismatch: {arr.dtype} vs {OUTPUT_DTYPE_NAME}."
                )
        else:
            if path.exists():
                raise RuntimeError(
                    f"Output already exists without an accepted resume state: {path}\n"
                    "Use --overwrite to start a fresh extraction."
                )
            arr = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=OUTPUT_DTYPE,
                shape=shape,
            )
        arrays[layer] = arr

    return arrays


def flush_all(arrays: Mapping[int, np.memmap]) -> None:
    for arr in arrays.values():
        arr.flush()


def validate_existing_complete(
    output_dir: Path,
    manifest_path: Path,
    expected_fingerprint: str,
) -> None:
    manifest = load_json(manifest_path)
    if manifest.get("code_name") != CODE_NAME:
        raise RuntimeError("Existing Code-09 manifest has unexpected code_name.")
    if not bool(manifest.get("overall_pass", False)):
        raise RuntimeError("Existing Code-09 manifest is not a PASS.")
    if manifest.get("run_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "Existing Code-09 output fingerprint differs from requested run."
        )

    hashes = manifest.get("file_sha256", {})
    shape = expected_array_shape()
    for layer in REPR_LAYERS:
        path = output_dir / layer_filename(layer)
        expected_sha = hashes.get(path.name)
        if not expected_sha:
            raise RuntimeError(f"Existing manifest lacks SHA256 for {path.name}.")
        verify_file_hash(path, expected_sha, f"existing layer {layer}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.shape != shape or arr.dtype != OUTPUT_DTYPE:
            raise RuntimeError(f"Existing layer {layer} shape/dtype mismatch.")
        # Streaming finiteness check in row blocks.
        for start in range(0, len(arr), 1024):
            block = np.asarray(arr[start : start + 1024], dtype=np.float32)
            if not np.isfinite(block).all():
                raise RuntimeError(f"Non-finite values in existing layer {layer}.")

    print(f"{CODE_NAME}: PASS")
    print("Existing complete extraction revalidated; no inference rerun was needed.")


# =============================================================================
# Batch token / representation audit
# =============================================================================


def validate_batch_tokens(
    batch_tokens: Any,
    sequences: Sequence[str],
    alphabet: Any,
    torch_module: Any,
) -> None:
    if tuple(batch_tokens.shape) != (len(sequences), EXPECTED_TOKEN_LENGTH):
        raise RuntimeError(
            f"Unexpected token tensor shape: {tuple(batch_tokens.shape)}"
        )

    if not torch_module.all(batch_tokens[:, 0] == int(alphabet.cls_idx)).item():
        raise RuntimeError("BOS/CLS token audit failed.")
    if not torch_module.all(
        batch_tokens[:, EXPECTED_SEQUENCE_LENGTH + 1] == int(alphabet.eos_idx)
    ).item():
        raise RuntimeError("EOS token audit failed.")

    # Exact residue identity audit at all four extracted token positions.
    expected = []
    for sequence in sequences:
        expected.append(
            [
                int(alphabet.get_idx(sequence[token_idx - 1]))
                for token_idx in TARGET_TOKEN_INDICES
            ]
        )
    expected_tensor = torch_module.tensor(
        expected,
        dtype=batch_tokens.dtype,
        device=batch_tokens.device,
    )
    observed = batch_tokens[:, list(TARGET_TOKEN_INDICES)]
    if not torch_module.equal(observed, expected_tensor):
        raise RuntimeError("Target-site residue-token identity audit failed.")


# =============================================================================
# Extraction
# =============================================================================


def run_extraction(
    df: pd.DataFrame,
    model: Any,
    alphabet: Any,
    torch_module: Any,
    device: Any,
    arrays: Mapping[int, np.memmap],
    start_row: int,
    batch_size: int,
    checkpoint_every: int,
    state_path: Path,
    fingerprint: str,
) -> Dict:
    batch_converter = alphabet.get_batch_converter()
    n = len(df)
    if start_row < 0 or start_row > n:
        raise RuntimeError(f"Invalid resume row {start_row} for N={n}.")

    start_time = time.time()
    batches_done = 0
    rows_this_run = 0

    with torch_module.inference_mode():
        for batch_start in range(start_row, n, batch_size):
            batch_stop = min(n, batch_start + batch_size)
            sub = df.iloc[batch_start:batch_stop]
            sequences = sub["sequence"].astype(str).tolist()
            labels = [f"extract_{int(x)}" for x in sub["extract_row_id"].tolist()]
            data = list(zip(labels, sequences))

            _, _, batch_tokens = batch_converter(data)
            validate_batch_tokens(
                batch_tokens=batch_tokens,
                sequences=sequences,
                alphabet=alphabet,
                torch_module=torch_module,
            )

            batch_tokens = batch_tokens.to(device, non_blocking=True)
            result = model(
                batch_tokens,
                repr_layers=list(REPR_LAYERS),
                return_contacts=False,
            )
            reps = result.get("representations", {})
            if set(int(k) for k in reps.keys()) != set(REPR_LAYERS):
                raise RuntimeError(
                    "ESM-2 did not return the complete requested layer set 0..30."
                )

            for layer in REPR_LAYERS:
                tensor = reps[layer]
                expected_shape = (
                    batch_stop - batch_start,
                    EXPECTED_TOKEN_LENGTH,
                    EXPECTED_EMBED_DIM,
                )
                if tuple(tensor.shape) != expected_shape:
                    raise RuntimeError(
                        f"Layer {layer} tensor shape mismatch: "
                        f"{tuple(tensor.shape)} vs {expected_shape}."
                    )
                sites = tensor[:, list(TARGET_TOKEN_INDICES), :]
                if tuple(sites.shape[1:]) != SITE_SHAPE:
                    raise RuntimeError(f"Layer {layer} target-site shape mismatch.")
                if not torch_module.isfinite(sites).all().item():
                    raise RuntimeError(f"Non-finite ESM-2 values at layer {layer}.")

                arrays[layer][batch_start:batch_stop] = (
                    sites.detach().to("cpu", dtype=torch_module.float32).numpy()
                    .astype(OUTPUT_DTYPE, copy=False)
                )

            batches_done += 1
            rows_this_run += batch_stop - batch_start

            should_checkpoint = (
                batches_done % checkpoint_every == 0 or batch_stop == n
            )
            if should_checkpoint:
                flush_all(arrays)
                write_json_atomic(
                    state_path,
                    {
                        "code_name": CODE_NAME,
                        "status": "in_progress" if batch_stop < n else "inference_complete",
                        "run_fingerprint": fingerprint,
                        "next_row": int(batch_stop),
                        "n_extract": n,
                        "batch_size": int(batch_size),
                        "device": str(device),
                        "representation_layers": list(REPR_LAYERS),
                        "output_dtype": OUTPUT_DTYPE_NAME,
                    },
                )
                elapsed = time.time() - start_time
                rate = rows_this_run / max(elapsed, 1e-9)
                print(
                    f"  committed rows {batch_stop:>6,}/{n:,} "
                    f"({100.0 * batch_stop / n:5.1f}%) | "
                    f"{rate:6.2f} seq/s",
                    flush=True,
                )

            # Release GPU references promptly.
            del result, reps, batch_tokens
            if device.type == "cuda":
                # No empty_cache call per batch: allocator reuse is desirable.
                pass

    return {
        "start_row": int(start_row),
        "rows_processed_this_run": int(rows_this_run),
        "batches_this_run": int(batches_done),
        "elapsed_seconds_this_run": float(time.time() - start_time),
    }


# =============================================================================
# Final validation and manifest
# =============================================================================


def final_validate_and_hash(
    output_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, object]]]:
    shape = expected_array_shape()
    hashes: Dict[str, str] = {}
    layer_stats: Dict[str, Dict[str, object]] = {}

    print()
    print("Final layer-file validation / SHA256 pass ...")
    for layer in REPR_LAYERS:
        path = output_dir / layer_filename(layer)
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.shape != shape:
            raise RuntimeError(
                f"Final layer {layer} shape mismatch: {arr.shape} vs {shape}."
            )
        if arr.dtype != OUTPUT_DTYPE:
            raise RuntimeError(
                f"Final layer {layer} dtype mismatch: {arr.dtype}."
            )

        min_value = np.inf
        max_value = -np.inf
        for start in range(0, len(arr), 1024):
            block = np.asarray(arr[start : start + 1024], dtype=np.float32)
            if not np.isfinite(block).all():
                raise RuntimeError(f"Non-finite values in final layer {layer}.")
            min_value = min(min_value, float(block.min()))
            max_value = max(max_value, float(block.max()))

        hashes[path.name] = sha256_file(path)
        layer_stats[str(layer)] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min_float32_scan": float(min_value),
            "max_float32_scan": float(max_value),
            "file_bytes": int(path.stat().st_size),
        }
        print(f"  layer {layer:02d}: PASS")

    return hashes, layer_stats


def save_final_manifest(
    output_dir: Path,
    state_path: Path,
    code08_manifest_path: Path,
    code08_sha: str,
    extraction_index_path: Path,
    extraction_index_sha: str,
    code02_manifest_path: Path,
    code02_sha: str,
    run_fingerprint_value: str,
    device: Any,
    batch_size: int,
    parameter_count: int,
    torch_module: Any,
    esm_module: Any,
    run_stats: Dict,
    file_hashes: Dict[str, str],
    layer_stats: Dict[str, Dict[str, object]],
) -> Path:
    manifest_path = output_dir / "esm2_extraction_manifest.json"

    total_layer_bytes = int(
        sum(int(v["file_bytes"]) for v in layer_stats.values())
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": "Frozen ESM-2 layerwise site-representation extraction for the exact Code-08 strict GB1 sequence union.",
        "run_fingerprint": run_fingerprint_value,
        "upstream": {
            "code08_manifest_path": str(code08_manifest_path.resolve()),
            "code08_manifest_sha256": code08_sha,
            "code08_overall_pass": True,
            "extraction_index_path": str(extraction_index_path.resolve()),
            "extraction_index_sha256": extraction_index_sha,
            "code02_manifest_path": str(code02_manifest_path.resolve()),
            "code02_manifest_sha256": code02_sha,
            "code02_status": "PASS",
        },
        "model": {
            "name": MODEL_NAME,
            "parameter_count": int(parameter_count),
            "transformer_layers": EXPECTED_NUM_LAYERS,
            "representation_layers": list(REPR_LAYERS),
            "embedding_dim": EXPECTED_EMBED_DIM,
            "eval_mode": True,
            "all_parameters_frozen": True,
            "autocast_used": False,
            "forward_compute_model_dtype": "float32",
        },
        "extraction": {
            "n_sequences": EXPECTED_N_EXTRACT,
            "sequence_length": EXPECTED_SEQUENCE_LENGTH,
            "token_length_with_bos_eos": EXPECTED_TOKEN_LENGTH,
            "biological_positions_1based": list(POSITIONS),
            "esm_token_indices": list(TARGET_TOKEN_INDICES),
            "sites_per_sequence": len(POSITIONS),
            "saved_shape_per_layer": list(expected_array_shape()),
            "saved_dtype": OUTPUT_DTYPE_NAME,
            "batch_size": int(batch_size),
            "device": str(device),
            "total_layer_file_bytes": total_layer_bytes,
            "fitness_read_or_used": False,
            "mixed_cubes_included": False,
        },
        "run_stats": run_stats,
        "layer_validation": layer_stats,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": str(torch_module.__version__),
            "fair_esm_distribution": package_version("fair-esm"),
            "esm_module_file": str(getattr(esm_module, "__file__", "unknown")),
        },
        "outputs": {
            "representation_directory": str(output_dir.resolve()),
            "layer_files": {
                str(layer): layer_filename(layer) for layer in REPR_LAYERS
            },
            "state_file": state_path.name,
        },
        "file_sha256": file_hashes,
        "overall_pass": True,
        "interpretation_boundary": [
            "These are frozen ESM-2 hidden states, not fitness predictions.",
            "Only the four GB1 assayed residue positions are stored.",
            "Float16 is a disk-storage representation; downstream Walsh/MI computation must cast to float32 or float64.",
            "Code 09 does not use experimental fitness and cannot by itself support biological MI conclusions.",
        ],
    }
    write_json_atomic(manifest_path, manifest)

    write_json_atomic(
        state_path,
        {
            "code_name": CODE_NAME,
            "status": "complete_pass",
            "run_fingerprint": run_fingerprint_value,
            "next_row": EXPECTED_N_EXTRACT,
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return manifest_path


# =============================================================================
# Self-test
# =============================================================================


def run_self_test() -> None:
    assert REPR_LAYERS == tuple(range(31))
    assert layer_filename(0) == "esm2_layer_00_sites.npy"
    assert layer_filename(30) == "esm2_layer_30_sites.npy"
    assert expected_array_shape() == (13_150, 4, 640)

    # Fingerprint must be deterministic and sensitive to a material run setting.
    a = run_fingerprint("a", "b", "c", 16, "cuda")
    b = run_fingerprint("a", "b", "c", 16, "cuda")
    c = run_fingerprint("a", "b", "c", 8, "cuda")
    assert a == b
    assert a != c

    # Storage-size arithmetic: 31 layers * N * 4 * 640 * 2 bytes.
    expected_bytes = 31 * EXPECTED_N_EXTRACT * 4 * EXPECTED_EMBED_DIM * 2
    assert expected_bytes == 2_087_168_000

    print("SELF-TEST: PASS")
    print(
        "Verified: layer naming 0..30, frozen [13,150,4,640] output contract, "
        "restart fingerprint determinism, and ~2.087 GB float16 storage arithmetic."
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run deterministic non-model tests and exit.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Flush all layer files and advance restart state every N batches.")
    parser.add_argument("--code08-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing Code-09 extraction and start from row 0.")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1.")

    root = project_root()
    exp_dir = root / "04 Experiments" / "EXP02_GB1_ESM2"
    code08_manifest_path = args.code08_manifest or (
        exp_dir / "gb1_esm2_extraction_manifest.json"
    )
    output_dir = args.output_dir or (exp_dir / "ESM2_Site_Reps")
    state_path = output_dir / "esm2_extraction_state.json"
    final_manifest_path = output_dir / "esm2_extraction_manifest.json"

    print("=" * 76)
    print(f"{CODE_NAME} — FROZEN ESM-2 SITE REPRESENTATIONS")
    print("=" * 76)
    print()

    (
        code08_manifest,
        code08_sha,
        extraction_index_path,
        extraction_index_sha,
        code02_manifest_path,
        code02_sha,
    ) = load_and_verify_code08(code08_manifest_path)
    verify_code02_contract(code02_manifest_path, code02_sha)
    df = load_extraction_index(extraction_index_path)

    print("Upstream provenance / label-boundary gate: PASS")
    print(f"  Code-08 manifest SHA256: {code08_sha}")
    print(f"  extraction-index SHA256: {extraction_index_sha}")
    print(f"  exact sequences: {len(df):,}")
    print("  fitness-bearing columns: NO")
    print("  mixed cubes: NO")
    print()

    torch, esm = import_esm_stack()
    device = choose_device(torch, args.device)

    fingerprint = run_fingerprint(
        code08_sha=code08_sha,
        code02_sha=code02_sha,
        extraction_index_sha=extraction_index_sha,
        batch_size=args.batch_size,
        device_type=device.type,
    )

    if args.overwrite:
        remove_existing_outputs(output_dir)

    # If already complete with the same fingerprint, revalidate instead of rerun.
    if final_manifest_path.exists() and state_path.exists() and not args.overwrite:
        state = load_json(state_path)
        if (
            state.get("status") == "complete_pass"
            and state.get("run_fingerprint") == fingerprint
        ):
            print("Matching complete Code-09 extraction found; revalidating hashes.")
            validate_existing_complete(
                output_dir=output_dir,
                manifest_path=final_manifest_path,
                expected_fingerprint=fingerprint,
            )
            return
        raise RuntimeError(
            "Existing Code-09 outputs do not match the requested completed run.\n"
            "Use --overwrite only if you intentionally want a fresh extraction."
        )

    resume = False
    start_row = 0
    if state_path.exists() and not args.overwrite:
        state = load_json(state_path)
        if state.get("status") not in ("in_progress", "inference_complete"):
            raise RuntimeError(
                f"Unrecognized/incompatible Code-09 state: {state.get('status')!r}"
            )
        if state.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                "Existing restart state fingerprint differs from requested run.\n"
                "Use the same batch/device contract or pass --overwrite."
            )
        start_row = int(state.get("next_row", -1))
        if not (0 <= start_row <= EXPECTED_N_EXTRACT):
            raise RuntimeError("Restart state next_row is invalid.")
        resume = True

    print(f"Requested device: {args.device}")
    print(f"Resolved device:  {device}")
    if device.type != "cuda":
        print("WARNING: Code 09 is GPU-preferred; CPU extraction will be slow.")
    print(f"Batch size:       {args.batch_size}")
    print(f"Resume:           {'YES' if resume else 'NO'}")
    print(f"Start row:        {start_row:,}")
    print("Compute autocast: NO (float32 model computation)")
    print("Disk dtype:       float16")
    print("Estimated layer-file storage: 2.087 GB")
    print()

    print(f"Loading official pretrained {MODEL_NAME} ...")
    model, alphabet, parameter_count = load_and_audit_model(
        esm_module=esm,
        device=device,
    )
    print("ESM-2 architecture/freeze audit: PASS")
    print(f"  parameters: {parameter_count:,}")
    print(f"  transformer layers: {EXPECTED_NUM_LAYERS}")
    print(f"  embedding dimension: {EXPECTED_EMBED_DIM}")
    print("  eval mode: PASS")
    print("  frozen parameters: PASS")
    print()

    arrays = open_layer_arrays(output_dir=output_dir, resume=resume)

    # If inference was fully committed before a crash during final hashing,
    # do not rerun the model.  Proceed directly to final validation.
    if start_row < EXPECTED_N_EXTRACT:
        write_json_atomic(
            state_path,
            {
                "code_name": CODE_NAME,
                "status": "in_progress",
                "run_fingerprint": fingerprint,
                "next_row": start_row,
                "n_extract": EXPECTED_N_EXTRACT,
                "batch_size": int(args.batch_size),
                "device": str(device),
            },
        )
        run_stats = run_extraction(
            df=df,
            model=model,
            alphabet=alphabet,
            torch_module=torch,
            device=device,
            arrays=arrays,
            start_row=start_row,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            state_path=state_path,
            fingerprint=fingerprint,
        )
    else:
        run_stats = {
            "start_row": int(start_row),
            "rows_processed_this_run": 0,
            "batches_this_run": 0,
            "elapsed_seconds_this_run": 0.0,
            "note": "Inference had already been fully committed; resumed at final validation.",
        }

    flush_all(arrays)
    del arrays

    file_hashes, layer_stats = final_validate_and_hash(output_dir)
    manifest_path = save_final_manifest(
        output_dir=output_dir,
        state_path=state_path,
        code08_manifest_path=code08_manifest_path,
        code08_sha=code08_sha,
        extraction_index_path=extraction_index_path,
        extraction_index_sha=extraction_index_sha,
        code02_manifest_path=code02_manifest_path,
        code02_sha=code02_sha,
        run_fingerprint_value=fingerprint,
        device=device,
        batch_size=args.batch_size,
        parameter_count=parameter_count,
        torch_module=torch,
        esm_module=esm,
        run_stats=run_stats,
        file_hashes=file_hashes,
        layer_stats=layer_stats,
    )

    print()
    print("Extraction audit: PASS")
    print(f"  sequences: {EXPECTED_N_EXTRACT:,}")
    print("  layers: 0..30")
    print("  saved tensor per layer: [13,150, 4, 640]")
    print("  saved dtype: float16")
    print("  all values finite: PASS")
    print("  all layer SHA256 hashes recorded: PASS")
    print("  experimental fitness read/used: NO")
    print()
    print("Saved:")
    print(f"  {output_dir}")
    print(f"  {manifest_path}")
    print(f"  {state_path}")
    print()
    print(f"{CODE_NAME}: PASS")
    print("Frozen ESM-2 representations are ready for Code 11 after Code 10 targets are available.")


if __name__ == "__main__":
    main()
