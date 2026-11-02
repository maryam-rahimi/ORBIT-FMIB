#!/usr/bin/env python3
"""
ORBIT-FMIB — GB1 IID SPLIT REPLICATION DIAGNOSTIC

Question
--------
Are the widespread negative held-out DV/MINE scores in fixed-subset GB1
primarily caused by using the AA-identity-strict Code-07 split as the
critic train/validation/test split?

Design
------
Keep fixed:
- same 8,210 biological GB1 strict cubes already used by fixed-S Code 11
- same R_{l,C,S} representation Walsh components
- same E_{C,S} experimental epistasis targets
- same Code-04 MINE critic implementation/hyperparameters
- same critic seeds
- same sample counts: 7,352 / 245 / 613
- same diagnostic layers: 0, 15, 30
- all 15 fixed positional subsets

Change only:
- AA-strict estimator split -> deterministic IID cube reassignment across
  the same 8,210 observations.

This script executes one of three pre-specified IID replication split seeds
(20260904, 20260905, 20260906) after the initial supportive split diagnostic
(seed 20260903). It preserves seed-specific outputs and does NOT overwrite the
initial diagnostic or Code 11/12/13 production outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

CODE_NAME = "ORBIT-FMIB — GB1 IID SPLIT REPLICATION DIAGNOSTIC"
SCHEMA_VERSION = 1

DIAGNOSTIC_LAYERS = (0, 15, 30)
ALL_MASKS = tuple(range(1, 16))
CRITIC_SEEDS = (11, 23, 37)

N_TRAIN = 7352
N_VAL = 245
N_TEST = 613
N_TOTAL = N_TRAIN + N_VAL + N_TEST

# Frozen before executing the replication diagnostics.
REPLICATION_IID_SPLIT_SEEDS = (20260904, 20260905, 20260906)
IID_SPLIT_SEED = None

# These are assigned to seed-specific names in main() after argument parsing.
IID_SEED_OUT = None
CELL_OUT = None
LAYER_OUT = None
STATE_OUT = None
MANIFEST_OUT = None

STRICT_DIAG_SEED_FILE = "gb1_subset_conditioned_diag_seed_results.csv"

IID_SEED_FIELDS = [
    "split_protocol",
    "iid_split_seed",
    "layer",
    "order_k",
    "subset_mask",
    "critic_seed",
    "representation_dim",
    "target_dim",
    "n_samples",
    "n_train",
    "n_val",
    "n_test",
    "best_step",
    "final_training_step",
    "best_val_mi",
    "test_mi",
    "test_product_shuffle_sd",
    "heldout_joint_minus_mismatch_score",
]

CELL_FIELDS = [
    "layer",
    "order_k",
    "subset_mask",
    "n_critic_seeds",
    "strict_mean_best_val",
    "strict_mean_test_mi",
    "strict_test_sd_across_seeds",
    "strict_all3_test_negative",
    "strict_val_positive_test_negative",
    "iid_mean_best_val",
    "iid_mean_test_mi",
    "iid_test_sd_across_seeds",
    "iid_all3_test_negative",
    "iid_val_positive_test_negative",
    "iid_minus_strict_mean_test",
    "strict_mean_val_minus_test_gap",
    "iid_mean_val_minus_test_gap",
]

LAYER_FIELDS = [
    "layer",
    "n_cells",
    "strict_cells_all3_test_negative",
    "iid_cells_all3_test_negative",
    "strict_cells_mean_test_negative",
    "iid_cells_mean_test_negative",
    "strict_cells_val_positive_test_negative",
    "iid_cells_val_positive_test_negative",
    "strict_mean_test_across_cells",
    "iid_mean_test_across_cells",
    "mean_iid_minus_strict_test_across_cells",
    "strict_mean_val_minus_test_gap_across_cells",
    "iid_mean_val_minus_test_gap_across_cells",
]


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json_atomic(path: Path, obj: Mapping) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def append_csv_row(path: Path, fieldnames: Sequence[str], row: Mapping) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise ContractError(f"Missing required CSV: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sample_sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def deterministic_iid_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(IID_SPLIT_SEED)
    perm = rng.permutation(N_TOTAL).astype(np.int64)
    tr = np.sort(perm[:N_TRAIN])
    va = np.sort(perm[N_TRAIN:N_TRAIN + N_VAL])
    te = np.sort(perm[N_TRAIN + N_VAL:])
    if (len(tr), len(va), len(te)) != (N_TRAIN, N_VAL, N_TEST):
        raise ContractError("IID split count mismatch.")
    if np.intersect1d(tr, va).size or np.intersect1d(tr, te).size or np.intersect1d(va, te).size:
        raise ContractError("IID split overlap.")
    if len(np.unique(np.concatenate([tr, va, te]))) != N_TOTAL:
        raise ContractError("IID split does not partition all observations.")
    return tr, va, te


def source_block(idx: int) -> str:
    if idx < N_TRAIN:
        return "strict_train_source"
    if idx < N_TRAIN + N_VAL:
        return "strict_val_source"
    return "strict_test_source"


def split_source_composition(split: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, Dict[str, int]]:
    names = ("iid_train", "iid_val", "iid_test")
    out: Dict[str, Dict[str, int]] = {}
    for name, idx in zip(names, split):
        c = defaultdict(int)
        for i in idx.tolist():
            c[source_block(int(i))] += 1
        out[name] = dict(c)
    return out


def expected_primary_strict_keys() -> set:
    return {
        (layer, mask, cs)
        for layer in DIAGNOSTIC_LAYERS
        for mask in ALL_MASKS
        for cs in CRITIC_SEEDS
    }


def load_strict_primary(path: Path) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    rows = read_csv(path)
    out = {}
    for r in rows:
        if r.get("family") != "primary":
            continue
        layer = int(float(r["layer"]))
        mask = int(float(r["subset_mask"]))
        cs = int(float(r["critic_seed"]))
        key = (layer, mask, cs)
        if layer not in DIAGNOSTIC_LAYERS or mask not in ALL_MASKS or cs not in CRITIC_SEEDS:
            continue
        if key in out:
            raise ContractError(f"Duplicate strict primary key: {key}")
        out[key] = {
            "order_k": int(float(r["order_k"])),
            "best_val_mi": float(r["best_val_mi"]),
            "test_mi": float(r["test_mi"]),
            "test_product_shuffle_sd": float(r["test_product_shuffle_sd"]),
            "heldout_joint_minus_mismatch_score": float(r["heldout_joint_minus_mismatch_score"]),
        }
    exp = expected_primary_strict_keys()
    if set(out) != exp:
        raise ContractError(
            f"Strict diagnostic primary grid mismatch: missing={len(exp - set(out))}, "
            f"extra={len(set(out) - exp)}"
        )
    return out


def load_iid_existing(path: Path) -> Dict[Tuple[int, int, int], Dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_csv(path)
    out = {}
    for r in rows:
        key = (
            int(float(r["layer"])),
            int(float(r["subset_mask"])),
            int(float(r["critic_seed"])),
        )
        if key in out:
            raise ContractError(f"Duplicate IID result key: {key}")
        out[key] = r
    if not set(out).issubset(expected_primary_strict_keys()):
        raise ContractError("IID result CSV contains unexpected keys.")
    return out


def build_cell_comparison(
    strict: Mapping[Tuple[int, int, int], Mapping],
    iid_rows: Mapping[Tuple[int, int, int], Mapping],
) -> List[Dict]:
    rows = []
    for layer in DIAGNOSTIC_LAYERS:
        for mask in ALL_MASKS:
            order = int(mask).bit_count()
            s = [strict[(layer, mask, cs)] for cs in CRITIC_SEEDS]
            q = [iid_rows[(layer, mask, cs)] for cs in CRITIC_SEEDS]

            s_val = [float(r["best_val_mi"]) for r in s]
            s_test = [float(r["test_mi"]) for r in s]
            q_val = [float(r["best_val_mi"]) for r in q]
            q_test = [float(r["test_mi"]) for r in q]

            s_mv = sum(s_val) / len(s_val)
            s_mt = sum(s_test) / len(s_test)
            q_mv = sum(q_val) / len(q_val)
            q_mt = sum(q_test) / len(q_test)

            rows.append({
                "layer": layer,
                "order_k": order,
                "subset_mask": mask,
                "n_critic_seeds": len(CRITIC_SEEDS),
                "strict_mean_best_val": s_mv,
                "strict_mean_test_mi": s_mt,
                "strict_test_sd_across_seeds": sample_sd(s_test),
                "strict_all3_test_negative": all(x < 0 for x in s_test),
                "strict_val_positive_test_negative": s_mv > 0 and s_mt < 0,
                "iid_mean_best_val": q_mv,
                "iid_mean_test_mi": q_mt,
                "iid_test_sd_across_seeds": sample_sd(q_test),
                "iid_all3_test_negative": all(x < 0 for x in q_test),
                "iid_val_positive_test_negative": q_mv > 0 and q_mt < 0,
                "iid_minus_strict_mean_test": q_mt - s_mt,
                "strict_mean_val_minus_test_gap": s_mv - s_mt,
                "iid_mean_val_minus_test_gap": q_mv - q_mt,
            })
    return rows


def build_layer_summary(cell_rows: Sequence[Mapping]) -> List[Dict]:
    out = []
    for layer in DIAGNOSTIC_LAYERS:
        rs = [r for r in cell_rows if int(r["layer"]) == layer]
        n = len(rs)
        if n != 15:
            raise ContractError(f"Layer {layer}: expected 15 comparison cells, found {n}")
        out.append({
            "layer": layer,
            "n_cells": n,
            "strict_cells_all3_test_negative": sum(bool(r["strict_all3_test_negative"]) for r in rs),
            "iid_cells_all3_test_negative": sum(bool(r["iid_all3_test_negative"]) for r in rs),
            "strict_cells_mean_test_negative": sum(float(r["strict_mean_test_mi"]) < 0 for r in rs),
            "iid_cells_mean_test_negative": sum(float(r["iid_mean_test_mi"]) < 0 for r in rs),
            "strict_cells_val_positive_test_negative": sum(bool(r["strict_val_positive_test_negative"]) for r in rs),
            "iid_cells_val_positive_test_negative": sum(bool(r["iid_val_positive_test_negative"]) for r in rs),
            "strict_mean_test_across_cells": sum(float(r["strict_mean_test_mi"]) for r in rs) / n,
            "iid_mean_test_across_cells": sum(float(r["iid_mean_test_mi"]) for r in rs) / n,
            "mean_iid_minus_strict_test_across_cells": sum(float(r["iid_minus_strict_mean_test"]) for r in rs) / n,
            "strict_mean_val_minus_test_gap_across_cells": sum(float(r["strict_mean_val_minus_test_gap"]) for r in rs) / n,
            "iid_mean_val_minus_test_gap_across_cells": sum(float(r["iid_mean_val_minus_test_gap"]) for r in rs) / n,
        })
    return out


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def self_test() -> None:
    split = deterministic_iid_split()
    tr, va, te = split
    assert (len(tr), len(va), len(te)) == (7352, 245, 613)
    assert len(np.unique(np.concatenate(split))) == 8210
    comp = split_source_composition(split)
    assert set(comp) == {"iid_train", "iid_val", "iid_test"}
    assert sum(comp["iid_train"].values()) == 7352
    assert sum(comp["iid_val"].values()) == 245
    assert sum(comp["iid_test"].values()) == 613
    print(f"{CODE_NAME} SELF-TEST: PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--iid-split-seed",
        type=int,
        required=True,
        choices=REPLICATION_IID_SPLIT_SEEDS,
        help="Pre-specified IID replication split seed.",
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--chunk-cubes", type=int, default=64)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    args = ap.parse_args()

    global IID_SPLIT_SEED, IID_SEED_OUT, CELL_OUT, LAYER_OUT, STATE_OUT, MANIFEST_OUT
    IID_SPLIT_SEED = int(args.iid_split_seed)
    prefix = f"gb1_iid_split_rep_seed_{IID_SPLIT_SEED}"
    IID_SEED_OUT = f"{prefix}_seed_results.csv"
    CELL_OUT = f"{prefix}_cell_comparison.csv"
    LAYER_OUT = f"{prefix}_layer_summary.csv"
    STATE_OUT = f"{prefix}_state.json"
    MANIFEST_OUT = f"{prefix}_manifest.json"

    if args.self_test:
        self_test()
        return 0
    if args.chunk_cubes < 1:
        raise ContractError("--chunk-cubes must be >= 1")

    root = args.project_root.resolve()
    code_dir = root / "03 Code"
    exp = root / "04 Experiments" / "EXP02_GB1_ESM2"

    subset_diag_source = code_dir / "11_validate_gb1_subset_conditioning.py"
    code11_source = code_dir / "11_run_gb1_target_mi.py"
    code04_source = code_dir / "04_validate_mi_estimator.py"
    code11_manifest = exp / "gb1_target_mi_manifest.json"
    code11_seed_results = exp / "gb1_target_mi_seed_results.csv"
    code08_manifest = exp / "gb1_esm2_extraction_manifest.json"
    code09_manifest = exp / "ESM2_Site_Reps" / "esm2_extraction_manifest.json"
    code10_manifest = exp / "gb1_order_targets_manifest.json"
    strict_diag_seed = exp / STRICT_DIAG_SEED_FILE

    iid_seed_out = exp / IID_SEED_OUT
    cell_out = exp / CELL_OUT
    layer_out = exp / LAYER_OUT
    state_out = exp / STATE_OUT
    manifest_out = exp / MANIFEST_OUT

    if args.overwrite:
        for p in (iid_seed_out, cell_out, layer_out, state_out, manifest_out):
            if p.exists():
                p.unlink()

    subset_mod = import_module_from_path(
        subset_diag_source, "orbit_subset_conditioning_diag"
    )

    (
        code11,
        mine04,
        c11,
        c08,
        c09,
        c10,
        maps,
        samples,
        _perms,
        _perm_meta,
        _eta2,
    ) = subset_mod.preflight(
        code11_source,
        code11_manifest,
        code11_seed_results,
        code04_source,
        code08_manifest,
        code09_manifest,
        code10_manifest,
    )

    strict = load_strict_primary(strict_diag_seed)

    iid_split = deterministic_iid_split()
    source_mix = split_source_composition(iid_split)

    # Verify fixed counts and that IID reassignment actually mixes the original
    # AA-strict source blocks across estimator partitions.
    for name, expected_n in zip(
        ("iid_train", "iid_val", "iid_test"),
        (N_TRAIN, N_VAL, N_TEST),
    ):
        if sum(source_mix[name].values()) != expected_n:
            raise ContractError(f"{name} source composition count mismatch.")

    print("=" * 86)
    print(CODE_NAME)
    print("=" * 86)
    print("Preflight: PASS")
    print("Same biological fixed-S GB1 observations: YES")
    print("Same Code-04 MINE implementation: YES")
    print(f"Diagnostic layers: {DIAGNOSTIC_LAYERS}")
    print("Fixed subsets: 15")
    print(f"Critic seeds: {CRITIC_SEEDS}")
    print(f"IID split seed: {IID_SPLIT_SEED}")
    print(f"Counts: train={N_TRAIN} val={N_VAL} test={N_TEST}")
    print("IID source-block composition:")
    for k, v in source_mix.items():
        print(f"  {k}: {v}")
    print("Existing AA-strict primary comparison rows: 135")

    if args.preflight_only:
        print("New IID critic fits run: 0")
        print("Production Code 11/12/13 outputs modified: NO")
        return 0

    torch, nn = mine04.import_torch()
    device = mine04.choose_device(torch, args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    existing = load_iid_existing(iid_seed_out)
    if existing and not args.resume:
        raise ContractError(
            f"{iid_seed_out.name} already exists. Use --resume or --overwrite."
        )

    done = set(existing)
    expected = expected_primary_strict_keys()
    X = code11.chi_matrix()

    total = len(expected)
    print(f"Expected new IID fits: {total}")

    for layer_id in DIAGNOSTIC_LAYERS:
        layer_keys = {k for k in expected if k[0] == layer_id}
        if layer_keys.issubset(done):
            continue
        print(f"\nLayer {layer_id:02d}", flush=True)
        layer = np.load(c09["layer_paths"][layer_id], mmap_mode="r", allow_pickle=False)

        for mask in ALL_MASKS:
            missing = [
                cs for cs in CRITIC_SEEDS
                if (layer_id, mask, cs) not in done
            ]
            if not missing:
                continue

            U, V, strict_split, counts = subset_mod.build_subset_dataset(
                code11, layer, maps, samples, mask, X, args.chunk_cubes
            )
            if len(U) != N_TOTAL or len(V) != N_TOTAL:
                raise ContractError("Fixed-S dataset size changed.")
            tr0, va0, te0 = strict_split
            if (len(tr0), len(va0), len(te0)) != (N_TRAIN, N_VAL, N_TEST):
                raise ContractError("AA-strict split counts changed.")

            order = int(mask).bit_count()
            for cs in missing:
                print(
                    f"  mask={mask:02d} k={order} iid critic_seed={cs}",
                    flush=True,
                )
                res = mine04.train_mine(
                    torch, nn, U, V, int(cs), device, iid_split
                )
                row = {
                    "split_protocol": "iid_cube_reassignment",
                    "iid_split_seed": IID_SPLIT_SEED,
                    "layer": layer_id,
                    "order_k": order,
                    "subset_mask": mask,
                    "critic_seed": cs,
                    "representation_dim": int(U.shape[1]),
                    "target_dim": int(V.shape[1]),
                    "n_samples": len(U),
                    "n_train": len(iid_split[0]),
                    "n_val": len(iid_split[1]),
                    "n_test": len(iid_split[2]),
                    **res,
                }
                append_csv_row(iid_seed_out, IID_SEED_FIELDS, row)
                done.add((layer_id, mask, cs))
                existing[(layer_id, mask, cs)] = {
                    k: str(v) for k, v in row.items()
                }
                write_json_atomic(
                    state_out,
                    {
                        "code_name": CODE_NAME,
                        "schema_version": SCHEMA_VERSION,
                        "status": "running",
                        "iid_split_seed": IID_SPLIT_SEED,
                        "completed_fits": len(done),
                        "expected_fits": total,
                        "source_composition": source_mix,
                    },
                )

    if done != expected:
        raise ContractError(
            f"IID run incomplete: completed={len(done)}, expected={len(expected)}"
        )

    # Reload authoritative written IID rows and build exact comparison.
    iid = load_iid_existing(iid_seed_out)
    cell_rows = build_cell_comparison(strict, iid)
    layer_rows = build_layer_summary(cell_rows)

    write_csv(cell_out, CELL_FIELDS, cell_rows)
    write_csv(layer_out, LAYER_FIELDS, layer_rows)

    # Overall descriptive summary
    overall = {
        "n_cells": len(cell_rows),
        "strict_cells_all3_test_negative": sum(
            bool(r["strict_all3_test_negative"]) for r in cell_rows
        ),
        "iid_cells_all3_test_negative": sum(
            bool(r["iid_all3_test_negative"]) for r in cell_rows
        ),
        "strict_cells_mean_test_negative": sum(
            float(r["strict_mean_test_mi"]) < 0 for r in cell_rows
        ),
        "iid_cells_mean_test_negative": sum(
            float(r["iid_mean_test_mi"]) < 0 for r in cell_rows
        ),
        "strict_cells_val_positive_test_negative": sum(
            bool(r["strict_val_positive_test_negative"]) for r in cell_rows
        ),
        "iid_cells_val_positive_test_negative": sum(
            bool(r["iid_val_positive_test_negative"]) for r in cell_rows
        ),
        "strict_mean_test_across_cells": sum(
            float(r["strict_mean_test_mi"]) for r in cell_rows
        ) / len(cell_rows),
        "iid_mean_test_across_cells": sum(
            float(r["iid_mean_test_mi"]) for r in cell_rows
        ) / len(cell_rows),
        "mean_iid_minus_strict_test_across_cells": sum(
            float(r["iid_minus_strict_mean_test"]) for r in cell_rows
        ) / len(cell_rows),
        "strict_mean_val_minus_test_gap_across_cells": sum(
            float(r["strict_mean_val_minus_test_gap"]) for r in cell_rows
        ) / len(cell_rows),
        "iid_mean_val_minus_test_gap_across_cells": sum(
            float(r["iid_mean_val_minus_test_gap"]) for r in cell_rows
        ) / len(cell_rows),
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "status": "complete_pass",
        "overall_pass": True,
        "purpose": (
            "Test whether AA-identity-strict critic transfer is responsible "
            "for widespread negative held-out fixed-S GB1 DV/MINE scores."
        ),
        "diagnostic_contract": {
            "same_biological_observations": True,
            "same_representation_targets": True,
            "same_mine_implementation": True,
            "same_critic_seeds": list(CRITIC_SEEDS),
            "same_sample_counts": {
                "train": N_TRAIN,
                "val": N_VAL,
                "test": N_TEST,
            },
            "layers": list(DIAGNOSTIC_LAYERS),
            "subset_masks": list(ALL_MASKS),
            "only_changed_factor": (
                "estimator train/val/test assignment: "
                "AA-identity-strict -> deterministic IID cube reassignment"
            ),
            "iid_split_seed": IID_SPLIT_SEED,
            "iid_split_source_composition": source_mix,
            "replication_iid_split": True,
            "pre_specified_replication_seeds": list(REPLICATION_IID_SPLIT_SEEDS),
            "production_code11_modified": False,
            "production_code12_modified": False,
            "production_code13_modified": False,
        },
        "inputs": {
            "subset_diagnostic_source": str(subset_diag_source.relative_to(root)),
            "subset_diagnostic_source_sha256": sha256_file(subset_diag_source),
            "code04_source": str(code04_source.relative_to(root)),
            "code04_source_sha256": sha256_file(code04_source),
            "strict_primary_seed_results": str(strict_diag_seed.relative_to(root)),
            "strict_primary_seed_results_sha256": sha256_file(strict_diag_seed),
            "code08_manifest_sha256": c08["sha"],
            "code09_manifest_sha256": c09["sha"],
            "code10_manifest_sha256": c10["sha"],
        },
        "outputs": {
            "iid_seed_results": IID_SEED_OUT,
            "cell_comparison": CELL_OUT,
            "layer_summary": LAYER_OUT,
            "state": STATE_OUT,
        },
        "overall_descriptive_summary": overall,
        "interpretation": {
            "supports_split_hypothesis_if": (
                "IID reassignment materially reduces negative held-out scores "
                "and validation-test gaps relative to the AA-strict protocol."
            ),
            "does_not_by_itself_authorize_production_change": True,
            "next_if_supportive": (
                "After all three pre-specified replication seeds are complete, "
                "apply the frozen replication gate. If all pass, freeze and "
                "globally revalidate the revised Code-11 estimator protocol "
                "before rerunning 465 biological cells."
            ),
        },
        "production_source": str(Path(__file__).resolve()),
        "production_source_sha256": sha256_file(Path(__file__).resolve()),
    }

    manifest["output_sha256"] = {
        IID_SEED_OUT: sha256_file(iid_seed_out),
        CELL_OUT: sha256_file(cell_out),
        LAYER_OUT: sha256_file(layer_out),
        STATE_OUT: sha256_file(state_out),
    }
    write_json_atomic(manifest_out, manifest)

    # Final state
    write_json_atomic(
        state_out,
        {
            "code_name": CODE_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "complete_pass",
            "iid_split_seed": IID_SPLIT_SEED,
            "completed_fits": len(done),
            "expected_fits": total,
            "source_composition": source_mix,
            "overall_descriptive_summary": overall,
        },
    )

    print("\nDIAGNOSTIC COMPLETE: PASS")
    print(f"New IID fits: {len(done)}")
    print(
        "All-3-seed negative cells "
        f"strict={overall['strict_cells_all3_test_negative']}/45 "
        f"iid={overall['iid_cells_all3_test_negative']}/45"
    )
    print(
        "Positive-val/negative-test cells "
        f"strict={overall['strict_cells_val_positive_test_negative']}/45 "
        f"iid={overall['iid_cells_val_positive_test_negative']}/45"
    )
    print(
        "Mean test DV across cells "
        f"strict={overall['strict_mean_test_across_cells']:.8g} "
        f"iid={overall['iid_mean_test_across_cells']:.8g}"
    )
    print(
        "Mean val-test gap across cells "
        f"strict={overall['strict_mean_val_minus_test_gap_across_cells']:.8g} "
        f"iid={overall['iid_mean_val_minus_test_gap_across_cells']:.8g}"
    )
    print(f"Manifest: {manifest_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
