#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 16 — PAPER FIGURES FOR LAYERWISE EPISTATIC DYNAMICS

Purpose
-------
Create paper-ready, provenance-locked visualizations from the completed
Code-15 descriptive layerwise analysis. This code performs visualization only.

Scientific boundary
-------------------
- Reads ONLY frozen Code-15 outputs from the corrected-IID production run.
- Does NOT refit ESM-2.
- Does NOT refit MINE / DV critics.
- Does NOT alter neural-dependence scores.
- Does NOT subtract null scores.
- Does NOT clamp negative scores.
- Does NOT perform statistical inference.
- Order-level curves are descriptive averages of independently peak-normalized
  fixed-subset trajectories; they are not pooled mutual-information estimates.
- Critic-seed peak ranges are estimator-stability diagnostics, not biological
  confidence intervals.

Primary figure
--------------
A. Order-level peak-normalized layerwise dynamics across ESM-2 depth.
B. 15-subset "epistatic information transit map" across all 31 stages.
C. Peak-to-final survival / retention for every fixed positional subset.

Secondary figure
----------------
Peak timing and critic-seed stability:
- central marker: peak layer from the mean fixed-subset trajectory
- horizontal range: min..max peak layer across critic seeds
- triangle: first layer reaching 90% of the mean-trajectory peak

Default input
-------------
04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910/Code15/

Default output
--------------
04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910/Code16/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CODE_NAME = "ORBIT-FMIB CODE 16 — PAPER FIGURES FOR LAYERWISE EPISTATIC DYNAMICS"
SCHEMA_VERSION = 1

LAYERS = tuple(range(31))
POSITIONS = (39, 40, 41, 54)
SUBSETS_BY_ORDER = {
    1: (1, 2, 4, 8),
    2: (3, 5, 6, 9, 10, 12),
    3: (7, 11, 13, 14),
    4: (15,),
}
EXPECTED_MASKS = tuple(mask for k in (1, 2, 3, 4) for mask in SUBSETS_BY_ORDER[k])
SUBSET_TO_ORDER = {mask: k for k, masks in SUBSETS_BY_ORDER.items() for mask in masks}

# =============================================================================
# EXTENSION: independent GPU replication overlay (added post-hoc, NOT part
# of the frozen Code15 provenance chain this script otherwise reads
# exclusively). Computed from raw per-(layer,mask,seed) replication critic
# scores using the same peak-normalization definition as Code15
# (N_l,S = D_l,S / max_j D_j,S), then order-level curves as the mean of
# independently peak-normalized fixed-subset trajectories -- identical
# methodology to Code15's own order-normalized output, applied to the
# replication run instead of the frozen production run.
# =============================================================================

REPLICATION_SEED_RESULTS_REL = Path("REPLICATION_INPUT_seed_results.csv")


def load_replication_normalized(root: Path):
    import pandas as pd

    path = root / REPLICATION_SEED_RESULTS_REL
    if not path.exists():
        return None

    df = pd.read_csv(root / REPLICATION_SEED_RESULTS_REL)
    per_cell = df.groupby(["layer", "subset_mask"])["test_mi"].mean().reset_index()

    norm_by_mask = {}
    subset_retain = {}
    for mask in EXPECTED_MASKS:
        sub = per_cell[per_cell["subset_mask"] == mask].sort_values("layer")
        if len(sub) != 31:
            raise ContractError(
                f"Replication overlay: mask {mask} has {len(sub)} layers, expected 31."
            )
        vals = sub["test_mi"].to_numpy()
        layers = sub["layer"].to_numpy()
        peak = float(vals.max())
        norm_by_mask[mask] = {int(l): float(v) / peak for l, v in zip(layers, vals)}
        subset_retain[mask] = float(vals[list(layers).index(30)]) / peak

    order_norm = {}
    for k in (1, 2, 3, 4):
        masks = SUBSETS_BY_ORDER[k]
        for l in LAYERS:
            vals = [norm_by_mask[m][l] for m in masks]
            order_norm.setdefault(k, {})[l] = float(np.mean(vals))

    order_summary = {}
    for k in (1, 2, 3, 4):
        layers_sorted = sorted(order_norm[k])
        means = np.array([order_norm[k][l] for l in layers_sorted])
        peak_idx = int(np.argmax(means))
        order_summary[k] = {
            "peak_layer": layers_sorted[peak_idx],
            "peak_value": float(means[peak_idx]),
            "final_value": float(means[-1]),
        }

    return {"order_norm": order_norm, "order_summary": order_summary, "subset_retain": subset_retain}

PROD_REL = Path("04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910")
CODE15_REL = PROD_REL / "Code15"
OUT_REL = PROD_REL / "Code16"

IN_SUBSET_SUMMARY = "gb1_layerwise_subset_summary.csv"
IN_NORMALIZED_LONG = "gb1_layerwise_subset_normalized_long.csv"
IN_ORDER_NORMALIZED = "gb1_order_layerwise_peak_normalized.csv"
IN_ORDER_SUMMARY = "gb1_layerwise_order_summary.csv"
IN_MANIFEST = "gb1_layerwise_dynamics_manifest.json"

OUT_MAIN_PDF = "figure_gb1_layerwise_dynamics_main.pdf"
OUT_MAIN_PNG = "figure_gb1_layerwise_dynamics_main.png"
OUT_PANEL_A = "panelA_order_trajectories.pdf"
OUT_PANEL_B = "panelB_subset_transit_map.pdf"
OUT_PANEL_C = "panelC_peak_to_final_survival.pdf"
OUT_TIMING = "figure_gb1_peak_timing_stability.pdf"
OUT_VALUES = "figure_gb1_layerwise_dynamics_key_values.txt"
OUT_MANIFEST = "gb1_layerwise_figures_manifest.json"


class ContractError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Mapping) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_int(x: object, label: str) -> int:
    try:
        return int(str(x))
    except Exception as exc:
        raise ContractError(f"Could not parse integer {label}: {x!r}") from exc


def parse_float(x: object, label: str) -> float:
    try:
        v = float(str(x))
    except Exception as exc:
        raise ContractError(f"Could not parse float {label}: {x!r}") from exc
    if not math.isfinite(v):
        raise ContractError(f"Non-finite value for {label}: {x!r}")
    return v


def mask_positions(mask: int) -> Tuple[int, ...]:
    return tuple(POSITIONS[i] for i in range(4) if mask & (1 << i))


def subset_label(mask: int) -> str:
    return "{" + ",".join(str(p) for p in mask_positions(mask)) + "}"


def configure_matplotlib(font_size: float) -> None:
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": font_size + 2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def verify_code15_manifest(code15_dir: Path) -> Tuple[dict, Dict[str, Path]]:
    manifest_path = code15_dir / IN_MANIFEST
    if not manifest_path.exists():
        raise ContractError(f"Missing Code15 manifest: {manifest_path}")

    m = read_json(manifest_path)
    if m.get("overall_pass") is not True:
        raise ContractError("Code15 manifest does not report overall_pass=True.")
    if m.get("status") != "complete_pass":
        raise ContractError("Code15 manifest status is not complete_pass.")

    contract = m.get("scientific_contract", {})
    required_contract = {
        "n_subsets": 15,
        "n_layer_subset_cells": 465,
        "normalization_performed_within_fixed_subset": True,
        "pooled_by_order_mi_estimation": False,
        "mine_refitting_performed": False,
        "null_scores_subtracted": False,
        "negative_scores_clamped": False,
        "formal_inference_performed": False,
        "critic_seeds_are_estimator_replicates": True,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ContractError(
                f"Code15 scientific contract mismatch for {key}: "
                f"expected {expected!r}, observed {contract.get(key)!r}"
            )

    gates = m.get("provenance_gates", {})
    required_gates = (
        "code11_manifest_pass",
        "code12_manifest_pass",
        "code13_manifest_pass",
        "code13_retention_rederived_exactly_within_tolerance",
        "code13_order_retention_matches_final_normalized_order_curves",
    )
    for key in required_gates:
        if gates.get(key) is not True:
            raise ContractError(f"Code15 provenance gate is not PASS: {key}")

    expected_outputs = {
        IN_SUBSET_SUMMARY,
        IN_NORMALIZED_LONG,
        IN_ORDER_NORMALIZED,
        IN_ORDER_SUMMARY,
    }
    outputs = m.get("outputs", {})
    output_hashes = m.get("output_sha256", {})
    paths: Dict[str, Path] = {}

    for name in expected_outputs:
        if name not in outputs:
            raise ContractError(f"Code15 manifest does not list required output {name}.")
        p = code15_dir / name
        if not p.exists():
            raise ContractError(f"Missing Code15 output: {p}")
        expected_hash = output_hashes.get(name)
        if not expected_hash:
            raise ContractError(f"Code15 manifest lacks output hash for {name}.")
        observed_hash = sha256_file(p)
        if observed_hash.lower() != str(expected_hash).lower():
            raise ContractError(
                f"Code15 output hash mismatch for {name}.\n"
                f"Expected: {expected_hash}\nObserved: {observed_hash}"
            )
        paths[name] = p

    paths[IN_MANIFEST] = manifest_path
    return m, paths


def load_subset_summary(path: Path) -> List[dict]:
    rows = read_csv(path)
    if len(rows) != 15:
        raise ContractError(f"Expected 15 subset-summary rows; found {len(rows)}.")

    seen = set()
    out = []
    for r in rows:
        mask = parse_int(r["subset_mask"], "subset_mask")
        k = parse_int(r["order_k"], "order_k")
        if mask not in SUBSET_TO_ORDER or SUBSET_TO_ORDER[mask] != k:
            raise ContractError(f"Unexpected subset/order pair: mask={mask}, k={k}")
        if mask in seen:
            raise ContractError(f"Duplicate subset mask in summary: {mask}")
        seen.add(mask)

        expected_label = subset_label(mask)
        if str(r["subset_positions"]) != expected_label:
            raise ContractError(
                f"Subset label mismatch for mask={mask}: "
                f"{r['subset_positions']!r} != {expected_label!r}"
            )

        retain = parse_float(r["retain_s"], f"retain_s S{mask:02d}")
        peak_layer = parse_int(r["peak_layer"], f"peak_layer S{mask:02d}")
        first90 = parse_int(
            r["first_ge_90pct_peak_layer"],
            f"first_ge_90pct_peak_layer S{mask:02d}",
        )
        seed_min = parse_int(r["seed_peak_layer_min"], f"seed_peak_layer_min S{mask:02d}")
        seed_max = parse_int(r["seed_peak_layer_max"], f"seed_peak_layer_max S{mask:02d}")

        if not (0 <= peak_layer <= 30 and 0 <= first90 <= 30):
            raise ContractError(f"Layer outside 0..30 for mask={mask}.")
        if not (0 <= seed_min <= seed_max <= 30):
            raise ContractError(f"Seed peak range outside 0..30 for mask={mask}.")
        if retain > 1.0 + 1e-10:
            raise ContractError(f"Retention >1 for mask={mask}: {retain}")

        out.append({
            **r,
            "subset_mask": mask,
            "order_k": k,
            "retain_s": retain,
            "peak_layer": peak_layer,
            "first_ge_90pct_peak_layer": first90,
            "seed_peak_layer_min": seed_min,
            "seed_peak_layer_max": seed_max,
        })

    if seen != set(EXPECTED_MASKS):
        raise ContractError("Subset-summary mask set is incomplete.")
    out.sort(key=lambda r: (r["order_k"], r["subset_mask"]))
    return out


def load_normalized_long(path: Path) -> List[dict]:
    rows = read_csv(path)
    if len(rows) != 465:
        raise ContractError(f"Expected 465 normalized-long rows; found {len(rows)}.")
    seen = set()
    out = []
    for r in rows:
        layer = parse_int(r["layer"], "layer")
        mask = parse_int(r["subset_mask"], "subset_mask")
        k = parse_int(r["order_k"], "order_k")
        score = parse_float(r["peak_normalized_score"], "peak_normalized_score")
        if layer not in LAYERS:
            raise ContractError(f"Unexpected layer {layer}")
        if mask not in SUBSET_TO_ORDER or SUBSET_TO_ORDER[mask] != k:
            raise ContractError(f"Unexpected subset/order pair: mask={mask}, k={k}")
        key = (layer, mask)
        if key in seen:
            raise ContractError(f"Duplicate normalized cell {key}")
        seen.add(key)
        out.append({
            **r,
            "layer": layer,
            "subset_mask": mask,
            "order_k": k,
            "peak_normalized_score": score,
        })
    if len(seen) != 465:
        raise ContractError("Normalized trajectory grid is incomplete.")
    return out


def load_order_normalized(path: Path) -> List[dict]:
    rows = read_csv(path)
    if len(rows) != 124:
        raise ContractError(f"Expected 124 order-normalized rows; found {len(rows)}.")
    seen = set()
    out = []
    for r in rows:
        layer = parse_int(r["layer"], "layer")
        k = parse_int(r["order_k"], "order_k")
        mean_v = parse_float(r["mean_peak_normalized_score"], "mean_peak_normalized_score")
        sd_v = parse_float(
            r["sd_peak_normalized_score_across_subsets"],
            "sd_peak_normalized_score_across_subsets",
        )
        key = (layer, k)
        if key in seen:
            raise ContractError(f"Duplicate order-normalized cell {key}")
        seen.add(key)
        out.append({
            **r,
            "layer": layer,
            "order_k": k,
            "mean_peak_normalized_score": mean_v,
            "sd_peak_normalized_score_across_subsets": sd_v,
        })
    if seen != {(l, k) for l in LAYERS for k in (1, 2, 3, 4)}:
        raise ContractError("Order-normalized grid is incomplete.")
    return out


def load_order_summary(path: Path) -> List[dict]:
    rows = read_csv(path)
    if len(rows) != 4:
        raise ContractError(f"Expected 4 order-summary rows; found {len(rows)}.")
    out = []
    for r in rows:
        k = parse_int(r["order_k"], "order_k")
        out.append({
            **r,
            "order_k": k,
            "order_normalized_curve_peak_layer": parse_int(
                r["order_normalized_curve_peak_layer"], "curve peak layer"
            ),
            "order_normalized_curve_peak_value": parse_float(
                r["order_normalized_curve_peak_value"], "curve peak value"
            ),
            "final_mean_peak_normalized_score": parse_float(
                r["final_mean_peak_normalized_score"], "final normalized score"
            ),
            "largest_adjacent_drop": parse_float(
                r["largest_adjacent_drop"], "largest adjacent drop"
            ),
            "largest_adjacent_drop_from_layer": parse_int(
                r["largest_adjacent_drop_from_layer"], "drop from layer"
            ),
            "largest_adjacent_drop_to_layer": parse_int(
                r["largest_adjacent_drop_to_layer"], "drop to layer"
            ),
        })
    out.sort(key=lambda r: r["order_k"])
    if [r["order_k"] for r in out] != [1, 2, 3, 4]:
        raise ContractError("Order summary does not contain k=1..4 exactly.")
    return out


def order_color(k: int):
    cmap = plt.get_cmap("tab10")
    return cmap(k - 1)


def add_order_separators(ax, subset_rows: Sequence[Mapping[str, object]]) -> None:
    # Rows are in order k=1,2,3,4. Separators appear between order groups.
    counts = [len(SUBSETS_BY_ORDER[k]) for k in (1, 2, 3, 4)]
    pos = 0
    for count in counts[:-1]:
        pos += count
        ax.axhline(pos - 0.5, color="0.75", linewidth=0.8)


def draw_panel_a(ax, order_norm: Sequence[Mapping[str, object]],
                 order_summary: Sequence[Mapping[str, object]],
                 repl=None) -> None:
    by_k = {k: [] for k in (1, 2, 3, 4)}
    for r in order_norm:
        by_k[int(r["order_k"])].append(r)
    summary_by_k = {int(r["order_k"]): r for r in order_summary}

    for k in (1, 2, 3, 4):
        rows = sorted(by_k[k], key=lambda r: int(r["layer"]))
        x = np.array([int(r["layer"]) for r in rows], dtype=float)
        y = np.array([float(r["mean_peak_normalized_score"]) for r in rows], dtype=float)
        sd = np.array(
            [float(r["sd_peak_normalized_score_across_subsets"]) for r in rows],
            dtype=float,
        )
        s = summary_by_k[k]
        label = (
            f"$k={k}$ (production, "
            f"final {float(s['final_mean_peak_normalized_score']):.3f})"
        )
        color = order_color(k)
        ax.plot(x, y, linewidth=2.0, label=label, color=color)
        if len(SUBSETS_BY_ORDER[k]) > 1:
            lo = np.clip(y - sd, 0.0, None)
            hi = y + sd
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)

        peak_l = int(s["order_normalized_curve_peak_layer"])
        peak_y = float(s["order_normalized_curve_peak_value"])
        final_y = float(s["final_mean_peak_normalized_score"])
        ax.scatter([peak_l], [peak_y], s=42, marker="o", color=color, zorder=4)
        ax.scatter([30], [final_y], s=45, marker="s", color=color, zorder=4)

        if repl is not None:
            rx = sorted(repl["order_norm"][k])
            ry = [repl["order_norm"][k][l] for l in rx]
            ax.plot(rx, ry, linewidth=1.4, linestyle="--", color=color, alpha=0.8,
                    label=f"$k={k}$ (replication, final {repl['order_summary'][k]['final_value']:.3f})")

    title = "A  Order-level dynamics through ESM-2"
    if repl is not None:
        title += " (solid=production, dashed=replication)"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("ESM-2 representation stage")
    ax.set_ylabel("Mean fraction of within-subset peak")
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5 if repl is not None else 8)


def draw_panel_b(ax, norm_long: Sequence[Mapping[str, object]],
                 subset_summary: Sequence[Mapping[str, object]],
                 add_colorbar: bool = True) -> None:
    rows_sorted = list(subset_summary)
    row_index = {int(r["subset_mask"]): i for i, r in enumerate(rows_sorted)}
    matrix = np.full((15, 31), np.nan, dtype=float)
    for r in norm_long:
        i = row_index[int(r["subset_mask"])]
        l = int(r["layer"])
        matrix[i, l] = float(r["peak_normalized_score"])
    if not np.all(np.isfinite(matrix)):
        raise ContractError("Heatmap matrix contains missing/non-finite cells.")

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    ylabels = [str(r["subset_positions"]) for r in rows_sorted]
    ax.set_yticks(np.arange(15))
    ax.set_yticklabels(ylabels)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("ESM-2 representation stage")
    ax.set_ylabel("Fixed positional subset")
    ax.set_title(
        "B  Epistatic information transit map",
        loc="left",
        fontweight="bold",
    )

    # Mark mean-trajectory peak and append final retention.
    for i, r in enumerate(rows_sorted):
        peak_l = int(r["peak_layer"])
        ax.scatter(
            [peak_l], [i],
            marker="*",
            s=55,
            facecolor="white",
            edgecolor="black",
            linewidth=0.45,
            zorder=5,
        )
        ax.text(
            30.9, i,
            f"{float(r['retain_s']):.3f}",
            va="center",
            ha="left",
            fontsize=max(6.5, plt.rcParams["font.size"] - 1.5),
            clip_on=False,
        )

    ax.text(
        30.9, -1.15, "L30 /\npeak",
        ha="left", va="bottom",
        fontsize=max(6.5, plt.rcParams["font.size"] - 1.5),
        fontweight="bold",
        clip_on=False,
    )

    add_order_separators(ax, rows_sorted)

    # Group labels on the left.
    start = 0
    for k in (1, 2, 3, 4):
        n = len(SUBSETS_BY_ORDER[k])
        mid = start + (n - 1) / 2
        ax.text(
            -3.8, mid, f"$k={k}$",
            va="center", ha="center",
            fontweight="bold",
            color=order_color(k),
            clip_on=False,
        )
        start += n

    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.08)
        cbar.set_label("Fraction of subset peak")
    return im


def draw_panel_c(ax, subset_summary: Sequence[Mapping[str, object]], repl=None) -> None:
    rows = list(subset_summary)
    y = np.arange(len(rows))
    retains = np.array([float(r["retain_s"]) for r in rows], dtype=float)
    xmin_candidates = [float(np.min(retains)) - 0.05]
    if repl is not None:
        xmin_candidates.append(min(repl["subset_retain"].values()) - 0.05)
    xmin = max(0.0, min(xmin_candidates))

    for i, r in enumerate(rows):
        k = int(r["order_k"])
        retain = float(r["retain_s"])
        color = order_color(k)
        ax.plot([retain, 1.0], [i, i], linewidth=2.0, color=color, alpha=0.75)
        ax.scatter([1.0], [i], s=28, facecolor="white", edgecolor=color, linewidth=1.2)
        ax.scatter([retain], [i], s=38, color=color, zorder=4)
        if repl is not None:
            repl_retain = repl["subset_retain"][int(r["subset_mask"])]
            ax.scatter([repl_retain], [i], s=38, facecolor="white", edgecolor=color,
                       linewidth=1.4, marker="D", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([str(r["subset_positions"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(xmin, 1.015)
    ax.set_xlabel("Peak-to-final retention")
    title = "C  What survives to the final representation?"
    if repl is not None:
        title += " (\u25cf production, \u25c7 replication)"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axvline(1.0, color="0.75", linewidth=0.8)
    ax.grid(axis="x", alpha=0.2)
    add_order_separators(ax, rows)

    for i, r in enumerate(rows):
        retain = float(r["retain_s"])
        ax.text(
            retain - 0.006, i,
            f"{retain:.3f}",
            va="center", ha="right",
            fontsize=max(6.5, plt.rcParams["font.size"] - 1.5),
        )


def draw_timing_stability(ax, subset_summary: Sequence[Mapping[str, object]]) -> None:
    rows = list(subset_summary)
    y = np.arange(len(rows))

    for i, r in enumerate(rows):
        k = int(r["order_k"])
        color = order_color(k)
        lo = int(r["seed_peak_layer_min"])
        hi = int(r["seed_peak_layer_max"])
        peak = int(r["peak_layer"])
        first90 = int(r["first_ge_90pct_peak_layer"])

        ax.plot([lo, hi], [i, i], color=color, linewidth=2.2, alpha=0.55)
        ax.scatter([peak], [i], s=42, marker="o", color=color, zorder=4)
        ax.scatter(
            [first90], [i],
            s=38, marker="^",
            facecolor="white", edgecolor=color, linewidth=1.2, zorder=4
        )

    ax.set_yticks(y)
    ax.set_yticklabels([str(r["subset_positions"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("ESM-2 representation stage")
    ax.set_ylabel("Fixed positional subset")
    ax.set_title(
        "Peak timing and critic-seed stability",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2)
    add_order_separators(ax, rows)

    legend = [
        Line2D([0], [0], marker="o", linestyle="None", color="0.2",
               label="Peak of mean trajectory"),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="white",
               markeredgecolor="0.2", color="0.2",
               label="First layer ≥90% of mean-trajectory peak"),
        Line2D([0], [0], linewidth=2.2, color="0.4",
               label="Min–max peak layer across critic seeds"),
    ]
    ax.legend(handles=legend, frameon=False, loc="lower right")


def save_single_panel(draw_fn, output: Path, figsize: Tuple[float, float], *args) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    draw_fn(ax, *args)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_main_figure(
    subset_summary: Sequence[Mapping[str, object]],
    norm_long: Sequence[Mapping[str, object]],
    order_norm: Sequence[Mapping[str, object]],
    order_summary: Sequence[Mapping[str, object]],
    pdf_path: Path,
    png_path: Path,
    dpi: int,
    repl=None,
) -> None:
    fig = plt.figure(figsize=(15.0, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.82, 1.32],
        width_ratios=[1.25, 0.82],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])

    draw_panel_a(ax_a, order_norm, order_summary, repl)
    draw_panel_c(ax_c, subset_summary, repl)
    draw_panel_b(ax_b, norm_long, subset_summary, add_colorbar=True)

    title = "Order-resolved epistatic information dynamics across ESM-2 depth"
    if repl is not None:
        title += "\n(A, C: production vs. independent GPU replication; B: production only)"
    fig.suptitle(title, fontweight="bold")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_key_values(
    path: Path,
    order_summary: Sequence[Mapping[str, object]],
    subset_summary: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ORBIT-FMIB CODE 16 — KEY VALUES USED BY THE FIGURES\n")
        f.write("Generated directly from frozen Code-15 outputs.\n\n")
        f.write("ORDER-LEVEL PEAK-NORMALIZED DYNAMICS\n")
        for r in order_summary:
            f.write(
                f"k={int(r['order_k'])}: "
                f"curve_peak=L{int(r['order_normalized_curve_peak_layer'])}; "
                f"peak_value={float(r['order_normalized_curve_peak_value']):.6f}; "
                f"final={float(r['final_mean_peak_normalized_score']):.6f}; "
                f"largest_adjacent_drop="
                f"L{int(r['largest_adjacent_drop_from_layer'])}"
                f"->L{int(r['largest_adjacent_drop_to_layer'])} "
                f"({float(r['largest_adjacent_drop']):.6f})\n"
            )
        f.write("\nFIXED-SUBSET PEAK LAYERS AND RETENTION\n")
        for r in subset_summary:
            f.write(
                f"S{int(r['subset_mask']):02d} {r['subset_positions']} "
                f"k={int(r['order_k'])}: "
                f"peak=L{int(r['peak_layer'])}; "
                f"first90=L{int(r['first_ge_90pct_peak_layer'])}; "
                f"retain={float(r['retain_s']):.6f}; "
                f"seed_peak_range=L{int(r['seed_peak_layer_min'])}"
                f"..L{int(r['seed_peak_layer_max'])}\n"
            )
        f.write("\nINTERPRETATION BOUNDARY\n")
        f.write("- Descriptive visualization only; no formal inference.\n")
        f.write("- Order curves are means of independently peak-normalized fixed-subset trajectories.\n")
        f.write("- k=4 contains one positional subset.\n")
        f.write("- Critic-seed peak ranges are estimator-stability diagnostics, not biological CIs.\n")


def self_test() -> None:
    # Basic mapping tests.
    assert mask_positions(1) == (39,)
    assert mask_positions(3) == (39, 40)
    assert mask_positions(15) == (39, 40, 41, 54)

    # Generate a complete toy dataset and render every figure in a temp dir.
    subset_summary = []
    norm_long = []
    order_norm = []
    order_summary = []

    for k, masks in SUBSETS_BY_ORDER.items():
        curves = []
        for j, mask in enumerate(masks):
            peak_l = min(30, 8 + 3 * k + j)
            vals = []
            for l in LAYERS:
                # Smooth rise and gentle decline, always positive.
                rise = 0.15 + 0.85 * min(1.0, (l + 1) / (peak_l + 1))
                decline = 1.0 if l <= peak_l else max(0.65, 1.0 - 0.012 * (l - peak_l))
                v = rise * decline
                vals.append(v)
            peak = max(vals)
            vals = [v / peak for v in vals]
            curves.append(vals)

            retain = vals[-1]
            subset_summary.append({
                "order_k": k,
                "subset_mask": mask,
                "subset_positions": subset_label(mask),
                "retain_s": retain,
                "peak_layer": int(np.argmax(vals)),
                "first_ge_90pct_peak_layer": next(i for i, v in enumerate(vals) if v >= 0.9),
                "seed_peak_layer_min": max(0, int(np.argmax(vals)) - 1),
                "seed_peak_layer_max": min(30, int(np.argmax(vals)) + 1),
            })
            for l, v in enumerate(vals):
                norm_long.append({
                    "layer": l,
                    "order_k": k,
                    "subset_mask": mask,
                    "subset_positions": subset_label(mask),
                    "peak_normalized_score": v,
                })

        arr = np.asarray(curves, dtype=float)
        means = arr.mean(axis=0)
        sds = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(31)
        for l in LAYERS:
            order_norm.append({
                "layer": l,
                "order_k": k,
                "mean_peak_normalized_score": float(means[l]),
                "sd_peak_normalized_score_across_subsets": float(sds[l]),
            })
        peak_l = int(np.argmax(means))
        deltas = np.diff(means)
        drop_from = int(np.argmin(deltas))
        order_summary.append({
            "order_k": k,
            "order_normalized_curve_peak_layer": peak_l,
            "order_normalized_curve_peak_value": float(means[peak_l]),
            "final_mean_peak_normalized_score": float(means[-1]),
            "largest_adjacent_drop": float(deltas[drop_from]),
            "largest_adjacent_drop_from_layer": drop_from,
            "largest_adjacent_drop_to_layer": drop_from + 1,
        })

    subset_summary.sort(key=lambda r: (r["order_k"], r["subset_mask"]))
    configure_matplotlib(9)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        make_main_figure(
            subset_summary, norm_long, order_norm, order_summary,
            td / "main.pdf", td / "main.png", 120
        )
        save_single_panel(
            draw_panel_a, td / "a.pdf", (8.0, 4.8), order_norm, order_summary
        )
        save_single_panel(
            draw_panel_b, td / "b.pdf", (11.0, 6.0), norm_long, subset_summary, True
        )
        save_single_panel(
            draw_panel_c, td / "c.pdf", (7.0, 6.0), subset_summary
        )
        save_single_panel(
            draw_timing_stability, td / "timing.pdf", (9.0, 6.0), subset_summary
        )
        write_key_values(td / "values.txt", order_summary, subset_summary)

        for name in ("main.pdf", "main.png", "a.pdf", "b.pdf", "c.pdf", "timing.pdf", "values.txt"):
            p = td / name
            assert p.exists() and p.stat().st_size > 0

    print(f"{CODE_NAME} SELF-TEST: PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        default=None,
        help="ORBIT-FMIB project root; default inferred from this source file.",
    )
    ap.add_argument(
        "--code15-dir",
        default=None,
        help="Override Code15 input directory.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Override Code16 output directory.",
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font-size", type=float, default=9.5)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    root = (
        Path(args.project_root).resolve()
        if args.project_root is not None
        else project_root().resolve()
    )
    code15_dir = (
        Path(args.code15_dir).resolve()
        if args.code15_dir is not None
        else (root / CODE15_REL).resolve()
    )
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (root / OUT_REL).resolve()
    )

    m15, input_paths = verify_code15_manifest(code15_dir)

    subset_summary = load_subset_summary(input_paths[IN_SUBSET_SUMMARY])
    norm_long = load_normalized_long(input_paths[IN_NORMALIZED_LONG])
    order_norm = load_order_normalized(input_paths[IN_ORDER_NORMALIZED])
    order_summary = load_order_summary(input_paths[IN_ORDER_SUMMARY])

    # Cross-check the final order curve against the mean subset retention.
    for k in (1, 2, 3, 4):
        subset_vals = [
            float(r["retain_s"]) for r in subset_summary if int(r["order_k"]) == k
        ]
        mean_subset_retain = float(np.mean(subset_vals))
        final_curve = next(
            float(r["final_mean_peak_normalized_score"])
            for r in order_summary
            if int(r["order_k"]) == k
        )
        if not math.isclose(mean_subset_retain, final_curve, rel_tol=0.0, abs_tol=1e-12):
            raise ContractError(
                f"k={k}: mean subset retention {mean_subset_retain} "
                f"does not equal final order-normalized curve {final_curve}."
            )

    print("=" * 92)
    print(f"{CODE_NAME} PREFLIGHT: PASS")
    print(f"Code15 directory: {code15_dir}")
    print("Code15 provenance and output hashes: PASS")
    print("Trajectory grid: 15 subsets x 31 layers = 465 cells")
    print("Order-normalized grid: 4 orders x 31 layers = 124 cells")
    print("Figure values rederived from Code15 outputs: PASS")
    print("No ESM-2 or MINE fitting: YES")
    print("Null subtraction: NO")
    print("Clamping: NO")
    print("Formal inference: NO")
    print("Critic-seed ranges interpreted as confidence intervals: NO")

    if args.preflight_only:
        print("Code16 figures written: NO")
        return 0

    if args.dpi < 72:
        raise ContractError("--dpi must be at least 72.")
    if args.font_size <= 0:
        raise ContractError("--font-size must be positive.")

    out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(args.font_size)

    repl = load_replication_normalized(root)
    if repl is not None:
        print("Replication input found: overlaying on Panels A and C.")
    else:
        print("No replication input found: output matches original (production only).")

    out_paths = {
        OUT_MAIN_PDF: out_dir / OUT_MAIN_PDF,
        OUT_MAIN_PNG: out_dir / OUT_MAIN_PNG,
        OUT_PANEL_A: out_dir / OUT_PANEL_A,
        OUT_PANEL_B: out_dir / OUT_PANEL_B,
        OUT_PANEL_C: out_dir / OUT_PANEL_C,
        OUT_TIMING: out_dir / OUT_TIMING,
        OUT_VALUES: out_dir / OUT_VALUES,
    }

    make_main_figure(
        subset_summary, norm_long, order_norm, order_summary,
        out_paths[OUT_MAIN_PDF], out_paths[OUT_MAIN_PNG], args.dpi, repl
    )
    save_single_panel(
        draw_panel_a, out_paths[OUT_PANEL_A], (8.2, 4.8),
        order_norm, order_summary, repl
    )
    save_single_panel(
        draw_panel_b, out_paths[OUT_PANEL_B], (11.2, 6.2),
        norm_long, subset_summary, True
    )
    save_single_panel(
        draw_panel_c, out_paths[OUT_PANEL_C], (7.2, 6.0),
        subset_summary
    )
    save_single_panel(
        draw_timing_stability, out_paths[OUT_TIMING], (9.2, 6.2),
        subset_summary
    )
    write_key_values(out_paths[OUT_VALUES], order_summary, subset_summary)

    for p in out_paths.values():
        if not p.exists() or p.stat().st_size <= 0:
            raise ContractError(f"Expected output was not created correctly: {p}")

    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "status": "complete_pass",
        "overall_pass": True,
        "purpose": (
            "Paper-ready visualization of frozen Code15 layerwise epistatic "
            "dynamics without new biological estimation or inference."
        ),
        "scientific_contract": {
            "input_source": "frozen corrected-IID Code15 outputs only",
            "mine_refitting_performed": False,
            "esm2_refitting_performed": False,
            "biological_scores_modified": False,
            "null_scores_subtracted": False,
            "negative_scores_clamped": False,
            "formal_inference_performed": False,
            "order_curve_definition": (
                "descriptive mean of independently peak-normalized fixed-subset trajectories"
            ),
            "heatmap_definition": "within-subset peak-normalized score N_l,S",
            "retention_panel_definition": "Code15 retain_s = final / within-subset peak",
            "seed_peak_range_definition": (
                "min..max peak layer across critic seeds; estimator-stability diagnostic only"
            ),
        },
        "upstream_code15": {
            "manifest": os.path.relpath(input_paths[IN_MANIFEST], root),
            "manifest_sha256": sha256_file(input_paths[IN_MANIFEST]),
            "production_source_sha256": m15.get("production_source_sha256"),
        },
        "inputs": {
            name: {
                "path": os.path.relpath(path, root),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "figure_contract": {
            "main_panel_A": "order-level peak-normalized trajectories across L0..L30",
            "main_panel_B": "15-subset epistatic information transit map across L0..L30",
            "main_panel_C": "subset-level peak-to-final retention",
            "secondary_peak_timing": (
                "mean-trajectory peak, first >=90%-of-peak layer, and critic-seed peak range"
            ),
            "subset_row_order": [
                {
                    "order_k": int(r["order_k"]),
                    "subset_mask": int(r["subset_mask"]),
                    "subset_positions": str(r["subset_positions"]),
                }
                for r in subset_summary
            ],
        },
        "outputs": {
            name: os.path.relpath(path, root)
            for name, path in out_paths.items()
        },
        "output_sha256": {
            name: sha256_file(path)
            for name, path in out_paths.items()
        },
        "production_source": (
            os.path.relpath(source_path, root)
            if source_path.is_relative_to(root)
            else str(source_path)
        ),
        "production_source_sha256": sha256_file(source_path),
        "plot_parameters": {
            "dpi": args.dpi,
            "font_size": args.font_size,
            "matplotlib_backend": matplotlib.get_backend(),
            "heatmap_colormap": "viridis",
            "order_colors": "matplotlib tab10 first four entries",
        },
        "interpretation_limits": [
            "Figures are descriptive and do not constitute formal statistical inference.",
            "Neural dependence scores are comparative and are not calibrated exact MI in nats.",
            "Order-level curves are descriptive averages after within-subset peak normalization.",
            "The fourth-order curve represents one fixed positional subset.",
            "Critic-seed peak ranges are estimator-stability diagnostics, not biological confidence intervals.",
            "No causal-use claim is supported by these figures.",
        ],
    }
    manifest_path = out_dir / OUT_MANIFEST
    write_json(manifest_path, manifest)

    print("\nPAPER FIGURES")
    for name, path in out_paths.items():
        print(f"  {name}: {path}")
    print(f"  {OUT_MANIFEST}: {manifest_path}")

    print("\nORDER-LEVEL VALUES USED IN PANEL A")
    for r in order_summary:
        print(
            f"  k={int(r['order_k'])}: "
            f"peak=L{int(r['order_normalized_curve_peak_layer'])}; "
            f"final={float(r['final_mean_peak_normalized_score']):.6f}; "
            f"largest_drop=L{int(r['largest_adjacent_drop_from_layer'])}"
            f"->L{int(r['largest_adjacent_drop_to_layer'])} "
            f"({float(r['largest_adjacent_drop']):.6f})"
        )

    print("\nORBIT-FMIB CODE 16 — PAPER FIGURES: PASS")
    print(f"Outputs: {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
