"""
ORBIT-FMIB Code 02
==================

Validate loading, tokenization, residue indexing, and hidden-state shapes for
the frozen ESM-2 150M model used in Phase 1.

Primary model
-------------
esm2_t30_150M_UR50D

This script is a validation gate only. It does NOT perform bulk representation
extraction and does NOT use fitness labels.

Inputs
------
02 Data/PFM_Inputs/gb1_pfm_index.csv
02 Data/PFM_Inputs/gb1_pfm_index_manifest.json

Output
------
02 Data/PFM_Inputs/esm2_loading_validation_manifest.json

Default device
--------------
CPU. This validation is intentionally small and CPU-safe.

The first model load may download the pretrained ESM-2 checkpoint if it is not
already cached locally.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Frozen Phase-1 configuration
# ============================================================

MODEL_NAME = "esm2_t30_150M_UR50D"

EXPECTED_NUM_LAYERS = 30
EXPECTED_EMBED_DIM = 640

WT_VARIANT = "VDGV"
GB1_DOMAIN_POSITIONS_1BASED = (39, 40, 41, 54)

# We validate representative hidden states without materializing all layers.
VALIDATION_REPR_LAYERS = (0, 1, 15, 30)

EXPECTED_PFM_COLUMNS = {
    "pfm_row_id",
    "variant",
    "sequence",
    "hd",
    "flip_set",
}


# ============================================================
# Paths
# ============================================================

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

PFM_DIR = PROJECT_ROOT / "02 Data" / "PFM_Inputs"

PFM_INDEX_PATH = PFM_DIR / "gb1_pfm_index.csv"
GB1_MANIFEST_PATH = PFM_DIR / "gb1_pfm_index_manifest.json"

OUTPUT_MANIFEST_PATH = (
    PFM_DIR / "esm2_loading_validation_manifest.json"
)


# ============================================================
# Utilities
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}:\n{path}\n\n"
            "Code 01 must pass before Code 02 is run."
        )


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


def choose_device(torch_module: Any, requested: str):
    if requested == "cpu":
        return torch_module.device("cpu")

    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is not available."
            )
        return torch_module.device("cuda")

    if requested == "auto":
        return torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )

    raise ValueError(f"Unsupported device request: {requested}")


# ============================================================
# Input audit
# ============================================================

def load_inputs() -> tuple[pd.DataFrame, dict]:
    require_file(PFM_INDEX_PATH, "Code-01 PFM sequence index")
    require_file(GB1_MANIFEST_PATH, "Code-01 GB1 manifest")

    index = pd.read_csv(PFM_INDEX_PATH, low_memory=False)

    require_columns(
        index,
        EXPECTED_PFM_COLUMNS,
        "gb1_pfm_index.csv",
    )

    # Critical leakage gate:
    # the PFM input index must not carry the fitness target.
    forbidden = {"fitness", "label", "target", "y"}
    present_forbidden = forbidden & set(index.columns)

    if present_forbidden:
        raise RuntimeError(
            "Label-isolation failure: downstream PFM index contains "
            f"forbidden target-like columns: {sorted(present_forbidden)}"
        )

    with GB1_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("status") != "PASS":
        raise RuntimeError(
            "Code-01 GB1 manifest does not report PASS."
        )

    if int(manifest.get("n_variants", -1)) != len(index):
        raise RuntimeError(
            "PFM index row count does not match Code-01 manifest."
        )

    if manifest.get("wt_variant") != WT_VARIANT:
        raise RuntimeError(
            "Unexpected WT variant in Code-01 manifest."
        )

    stored_mapping = manifest.get(
        "domain_to_stored_position_1based"
    )

    if not isinstance(stored_mapping, dict):
        raise RuntimeError(
            "Code-01 manifest lacks the domain-to-stored-position mapping."
        )

    observed_domain_positions = tuple(
        int(k) for k in stored_mapping.keys()
    )

    if set(observed_domain_positions) != set(
        GB1_DOMAIN_POSITIONS_1BASED
    ):
        raise RuntimeError(
            "Unexpected GB1 domain positions in Code-01 manifest.\n"
            f"Expected: {GB1_DOMAIN_POSITIONS_1BASED}\n"
            f"Observed: {sorted(observed_domain_positions)}"
        )

    # Verify index hash against the hash recorded by Code 01.
    expected_hash = manifest.get("output_index_sha256")
    observed_hash = sha256_file(PFM_INDEX_PATH)

    if expected_hash != observed_hash:
        raise RuntimeError(
            "PFM index hash mismatch relative to Code-01 manifest.\n"
            f"Expected: {expected_hash}\n"
            f"Observed: {observed_hash}"
        )

    return index, manifest


def select_validation_sequences(index: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministically select WT plus one example from each observed HD 1..4.

    These rows are used only to audit tokenization. They are not used for
    training, fitting, or any biological result.
    """

    index = index.copy()

    wt = index.loc[index["variant"].astype(str) == WT_VARIANT]

    if len(wt) != 1:
        raise RuntimeError(
            f"Expected exactly one WT row {WT_VARIANT}; found {len(wt)}."
        )

    rows = [wt.iloc[[0]]]

    for hd in (1, 2, 3, 4):
        subset = index.loc[
            pd.to_numeric(index["hd"], errors="coerce") == hd
        ]

        if len(subset) == 0:
            raise RuntimeError(
                f"No validation example found for Hamming distance {hd}."
            )

        # Stable selection independent of file row order.
        subset = subset.sort_values(
            ["variant", "sequence"],
            kind="mergesort",
        )

        rows.append(subset.iloc[[0]])

    selected = pd.concat(rows, ignore_index=True)

    if selected["sequence"].duplicated().any():
        raise RuntimeError(
            "Validation sequence selection unexpectedly contains duplicates."
        )

    return selected


# ============================================================
# ESM-2 loading
# ============================================================

def import_esm_stack():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed.\n"
            "Install a compatible PyTorch build before running Code 02."
        ) from exc

    try:
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "The `esm` package is not installed.\n"
            "Install the official package with:\n\n"
            "    pip install fair-esm\n"
        ) from exc

    return torch, esm


def load_model(esm_module: Any):
    if not hasattr(esm_module.pretrained, MODEL_NAME):
        raise RuntimeError(
            f"Installed `esm` package does not expose {MODEL_NAME}.\n"
            "Update/install the official fair-esm package."
        )

    loader = getattr(esm_module.pretrained, MODEL_NAME)

    model, alphabet = loader()

    model.eval()

    # Phase 1 uses a frozen foundation model.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(
            "Frozen-model audit failed: at least one parameter still "
            "requires gradients."
        )

    return model, alphabet


def audit_model_architecture(model: Any) -> dict:
    layers = getattr(model, "layers", None)

    if layers is None:
        raise RuntimeError(
            "Loaded ESM-2 model does not expose transformer layers."
        )

    observed_num_layers = len(layers)

    if observed_num_layers != EXPECTED_NUM_LAYERS:
        raise RuntimeError(
            "Unexpected ESM-2 transformer depth.\n"
            f"Expected: {EXPECTED_NUM_LAYERS}\n"
            f"Observed: {observed_num_layers}"
        )

    observed_embed_dim = int(
        getattr(model, "embed_dim", -1)
    )

    if observed_embed_dim != EXPECTED_EMBED_DIM:
        raise RuntimeError(
            "Unexpected ESM-2 embedding dimension.\n"
            f"Expected: {EXPECTED_EMBED_DIM}\n"
            f"Observed: {observed_embed_dim}"
        )

    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )

    return {
        "num_layers": observed_num_layers,
        "embed_dim": observed_embed_dim,
        "parameter_count": parameter_count,
        "training_flag": bool(model.training),
        "all_parameters_frozen": bool(
            not any(
                parameter.requires_grad
                for parameter in model.parameters()
            )
        ),
    }


# ============================================================
# Tokenization audit
# ============================================================

def audit_tokenization(
    alphabet: Any,
    selected: pd.DataFrame,
    gb1_manifest: dict,
    torch_module: Any,
) -> tuple[Any, dict]:
    batch_converter = alphabet.get_batch_converter()

    data = [
        (
            f"row_{int(row.pfm_row_id)}_{row.variant}",
            str(row.sequence),
        )
        for row in selected.itertuples(index=False)
    ]

    labels, batch_strs, batch_tokens = batch_converter(data)

    if batch_strs != [sequence for _, sequence in data]:
        raise RuntimeError(
            "ESM batch converter altered one or more input sequences."
        )

    sequence_lengths = [len(sequence) for _, sequence in data]

    if len(set(sequence_lengths)) != 1:
        raise RuntimeError(
            "Validation sequences unexpectedly have different lengths."
        )

    sequence_length = sequence_lengths[0]

    expected_token_length = sequence_length + 2

    if tuple(batch_tokens.shape) != (
        len(data),
        expected_token_length,
    ):
        raise RuntimeError(
            "Unexpected token tensor shape.\n"
            f"Expected: ({len(data)}, {expected_token_length})\n"
            f"Observed: {tuple(batch_tokens.shape)}"
        )

    # ESM-2 uses a BOS/CLS token at token index 0.
    if not hasattr(alphabet, "cls_idx"):
        raise RuntimeError(
            "Alphabet does not expose cls_idx; BOS indexing cannot be "
            "validated safely."
        )

    if not hasattr(alphabet, "eos_idx"):
        raise RuntimeError(
            "Alphabet does not expose eos_idx; EOS indexing cannot be "
            "validated safely."
        )

    if not torch_module.all(
        batch_tokens[:, 0] == int(alphabet.cls_idx)
    ).item():
        raise RuntimeError(
            "Token index 0 is not consistently the ESM BOS/CLS token."
        )

    if not torch_module.all(
        batch_tokens[:, sequence_length + 1]
        == int(alphabet.eos_idx)
    ).item():
        raise RuntimeError(
            "Expected EOS token was not found immediately after the "
            "last amino-acid token."
        )

    # Strongest residue-token check:
    # verify every residue in every validation sequence against the
    # alphabet token ID, not only the four assay positions.
    for row_idx, (_, sequence) in enumerate(data):
        expected_residue_ids = torch_module.tensor(
            [alphabet.get_idx(aa) for aa in sequence],
            dtype=batch_tokens.dtype,
        )

        observed_residue_ids = batch_tokens[
            row_idx,
            1 : sequence_length + 1,
        ].cpu()

        if not torch_module.equal(
            observed_residue_ids,
            expected_residue_ids,
        ):
            raise RuntimeError(
                "Residue-level ESM tokenization mismatch detected."
            )

    domain_to_stored = {
        int(k): int(v)
        for k, v in gb1_manifest[
            "domain_to_stored_position_1based"
        ].items()
    }

    target_records = []

    for row_idx, row in enumerate(
        selected.itertuples(index=False)
    ):
        variant = str(row.variant)
        sequence = str(row.sequence)

        for variant_offset, domain_position in enumerate(
            GB1_DOMAIN_POSITIONS_1BASED
        ):
            stored_position_1based = domain_to_stored[
                domain_position
            ]

            # Because token 0 is BOS/CLS, a 1-based residue position p
            # is represented at ESM token tensor index p.
            token_index = stored_position_1based

            residue_from_sequence = sequence[
                stored_position_1based - 1
            ]

            residue_from_variant = variant[variant_offset]

            token_id = int(
                batch_tokens[row_idx, token_index].item()
            )
            expected_token_id = int(
                alphabet.get_idx(residue_from_sequence)
            )

            if residue_from_sequence != residue_from_variant:
                raise RuntimeError(
                    "Code-01 sequence/variant mapping was not preserved "
                    "in the validation index."
                )

            if token_id != expected_token_id:
                raise RuntimeError(
                    "Target residue token-index mapping FAILED.\n"
                    f"Variant: {variant}\n"
                    f"GB1 domain position: {domain_position}\n"
                    f"Stored position: {stored_position_1based}\n"
                    f"ESM token index: {token_index}"
                )

            target_records.append(
                {
                    "variant": variant,
                    "gb1_domain_position_1based": (
                        domain_position
                    ),
                    "stored_position_1based": (
                        stored_position_1based
                    ),
                    "esm_token_index_0based_tensor": (
                        token_index
                    ),
                    "residue": residue_from_sequence,
                    "token_id": token_id,
                }
            )

    return batch_tokens, {
        "n_validation_sequences": len(selected),
        "sequence_length": sequence_length,
        "token_tensor_shape": list(batch_tokens.shape),
        "bos_token_index": 0,
        "eos_token_index": sequence_length + 1,
        "target_records": target_records,
    }


# ============================================================
# Forward-pass / representation audit
# ============================================================

def audit_forward_pass(
    model: Any,
    batch_tokens: Any,
    torch_module: Any,
    device: Any,
    gb1_manifest: dict,
) -> dict:
    # Forward only ONE sequence (WT) to keep Code 02 light on CPU.
    wt_tokens = batch_tokens[0:1].to(device)

    model = model.to(device)

    repr_layers = list(VALIDATION_REPR_LAYERS)

    with torch_module.inference_mode():
        results = model(
            wt_tokens,
            repr_layers=repr_layers,
            return_contacts=False,
        )

    representations = results.get("representations")

    if not isinstance(representations, dict):
        raise RuntimeError(
            "ESM-2 output does not contain a representations dictionary."
        )

    if set(representations.keys()) != set(repr_layers):
        raise RuntimeError(
            "Unexpected representation-layer keys.\n"
            f"Expected: {repr_layers}\n"
            f"Observed: {sorted(representations.keys())}"
        )

    expected_shape = (
        1,
        int(wt_tokens.shape[1]),
        EXPECTED_EMBED_DIM,
    )

    layer_shapes = {}

    for layer in repr_layers:
        tensor = representations[layer]

        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected hidden-state shape at layer {layer}.\n"
                f"Expected: {expected_shape}\n"
                f"Observed: {tuple(tensor.shape)}"
            )

        if not torch_module.isfinite(tensor).all().item():
            raise RuntimeError(
                f"Non-finite hidden-state values detected at layer {layer}."
            )

        layer_shapes[str(layer)] = list(tensor.shape)

    domain_to_stored = {
        int(k): int(v)
        for k, v in gb1_manifest[
            "domain_to_stored_position_1based"
        ].items()
    }

    target_token_indices = [
        domain_to_stored[position]
        for position in GB1_DOMAIN_POSITIONS_1BASED
    ]

    # Confirm that the final-layer four-site tensor has the expected
    # shape [4 residues, 640 dimensions].
    final_targets = representations[
        EXPECTED_NUM_LAYERS
    ][
        0,
        target_token_indices,
        :,
    ]

    if tuple(final_targets.shape) != (
        4,
        EXPECTED_EMBED_DIM,
    ):
        raise RuntimeError(
            "Unexpected four-site hidden-state shape.\n"
            f"Expected: (4, {EXPECTED_EMBED_DIM})\n"
            f"Observed: {tuple(final_targets.shape)}"
        )

    if not torch_module.isfinite(final_targets).all().item():
        raise RuntimeError(
            "Non-finite values detected in final-layer target residues."
        )

    return {
        "device": str(device),
        "repr_layers_checked": repr_layers,
        "layer_shapes": layer_shapes,
        "target_token_indices": target_token_indices,
        "final_target_tensor_shape": list(
            final_targets.shape
        ),
        "dtype": str(
            representations[EXPECTED_NUM_LAYERS].dtype
        ),
    }


# ============================================================
# Manifest
# ============================================================

def write_manifest(
    model_audit: dict,
    token_audit: dict,
    forward_audit: dict,
    gb1_manifest: dict,
    torch_module: Any,
    esm_module: Any,
) -> None:
    PFM_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "PASS",
        "code": "02_validate_esm2_loading.py",
        "model_name": MODEL_NAME,
        "model_role": "frozen Phase-1 protein foundation model",
        "architecture": model_audit,
        "tokenization": token_audit,
        "forward_pass": forward_audit,
        "gb1_domain_positions_1based": list(
            GB1_DOMAIN_POSITIONS_1BASED
        ),
        "domain_to_stored_position_1based": (
            gb1_manifest[
                "domain_to_stored_position_1based"
            ]
        ),
        "source_pfm_index_path": str(PFM_INDEX_PATH),
        "source_pfm_index_sha256": sha256_file(
            PFM_INDEX_PATH
        ),
        "source_gb1_manifest_path": str(
            GB1_MANIFEST_PATH
        ),
        "source_gb1_manifest_sha256": sha256_file(
            GB1_MANIFEST_PATH
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(torch_module.__version__),
            "fair_esm_distribution": package_version(
                "fair-esm"
            ),
            "esm_module_file": str(
                getattr(esm_module, "__file__", "unknown")
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "critical_invariants": [
            "model is esm2_t30_150M_UR50D",
            "transformer depth is 30",
            "embedding dimension is 640",
            "model is in eval mode",
            "all model parameters are frozen",
            "PFM input contains no fitness label",
            "ESM token 0 is BOS/CLS",
            "first amino acid is ESM token index 1",
            "residue tokenization exactly matches all validation sequences",
            "GB1 assay residues map to validated ESM token indices",
            "representations at layers 0,1,15,30 have expected shape",
            "target-site final representation has shape [4,640]",
            "no bulk representation extraction occurs in Code 02",
        ],
    }

    with OUTPUT_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help=(
            "Device for the single-sequence validation forward pass. "
            "Default: cpu."
        ),
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run all audits without writing the validation manifest.",
    )

    args = parser.parse_args()

    print("=" * 76)
    print("ORBIT-FMIB CODE 02 — ESM-2 LOADING / TOKEN-INDEX AUDIT")
    print("=" * 76)
    print()

    index, gb1_manifest = load_inputs()

    selected = select_validation_sequences(index)

    print("Code-01 inputs: PASS")
    print(f"  PFM rows: {len(index):,}")
    print("  fitness present in PFM input: NO")
    print(
        "  tokenization examples: "
        + ", ".join(
            selected["variant"].astype(str).tolist()
        )
    )
    print()

    torch, esm = import_esm_stack()

    device = choose_device(torch, args.device)

    print(f"Requested device: {args.device}")
    print(f"Validation device: {device}")
    print(
        "Loading official pretrained model "
        f"{MODEL_NAME} ..."
    )
    print(
        "(First run may download the checkpoint if it is not cached.)"
    )
    print()

    model, alphabet = load_model(esm)

    model_audit = audit_model_architecture(model)

    if model_audit["training_flag"]:
        raise RuntimeError(
            "Model remains in training mode after model.eval()."
        )

    print("ESM-2 architecture audit: PASS")
    print(f"  model: {MODEL_NAME}")
    print(
        f"  transformer layers: "
        f"{model_audit['num_layers']}"
    )
    print(
        f"  embedding dimension: "
        f"{model_audit['embed_dim']}"
    )
    print(
        f"  parameters: "
        f"{model_audit['parameter_count']:,}"
    )
    print("  eval mode: PASS")
    print("  frozen parameters: PASS")
    print()

    batch_tokens, token_audit = audit_tokenization(
        alphabet=alphabet,
        selected=selected,
        gb1_manifest=gb1_manifest,
        torch_module=torch,
    )

    print("ESM-2 tokenization audit: PASS")
    print(
        f"  sequence length: "
        f"{token_audit['sequence_length']}"
    )
    print(
        f"  token length including BOS/EOS: "
        f"{token_audit['token_tensor_shape'][1]}"
    )
    print("  BOS/CLS token index: 0")
    print(
        f"  EOS token index: "
        f"{token_audit['eos_token_index']}"
    )

    domain_to_stored = {
        int(k): int(v)
        for k, v in gb1_manifest[
            "domain_to_stored_position_1based"
        ].items()
    }

    print("  GB1 domain -> ESM token tensor index:")
    for domain_position in GB1_DOMAIN_POSITIONS_1BASED:
        stored_position = domain_to_stored[
            domain_position
        ]
        print(
            f"    domain {domain_position} -> "
            f"stored residue {stored_position} -> "
            f"token index {stored_position}"
        )

    print("  residue-token identity checks: PASS")
    print()

    forward_audit = audit_forward_pass(
        model=model,
        batch_tokens=batch_tokens,
        torch_module=torch,
        device=device,
        gb1_manifest=gb1_manifest,
    )

    print("Single-sequence forward-pass audit: PASS")
    print(
        "  layers checked: "
        + ", ".join(
            str(x)
            for x in forward_audit[
                "repr_layers_checked"
            ]
        )
    )
    print(
        f"  target tensor at final layer: "
        f"{tuple(forward_audit['final_target_tensor_shape'])}"
    )
    print(
        f"  representation dtype: "
        f"{forward_audit['dtype']}"
    )
    print()

    if args.no_write:
        print("NO-WRITE MODE: manifest not written.")
    else:
        write_manifest(
            model_audit=model_audit,
            token_audit=token_audit,
            forward_audit=forward_audit,
            gb1_manifest=gb1_manifest,
            torch_module=torch,
            esm_module=esm,
        )

        print("Saved:")
        print(f"  {OUTPUT_MANIFEST_PATH}")
        print()

    print("ORBIT-FMIB CODE 02: PASS")


if __name__ == "__main__":
    main()
