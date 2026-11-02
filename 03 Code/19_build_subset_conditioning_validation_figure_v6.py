#!/usr/bin/env python3
"""
Build the ORBIT-FMIB four-panel estimator-validation figure.

IMPORTANT
---------
All plotted numerical values are read directly from authoritative pipeline CSV
outputs. No result value is manually hard-coded.

Panels
------
A. Pooled estimator fails cube-derangement null
B. Positional subset identity predicts epistatic-target distribution
C. Fixed-subset conditioning collapses the randomized null
D. Exact-sample calibration preserves strong-signal detection

Expected project layout
-----------------------
ORBIT-FMIB/
├── 03 Code/
├── 04 Experiments/
│   ├── EXP02_GB1_ESM2/
│   └── EXP03_MI/
└── 05 Figure/

Outputs
-------
05 Figure/Figure_subset_conditioning_validation_paper_v6.pdf
05 Figure/Figure_subset_conditioning_validation_paper_v6.png
05 Figure/Figure_subset_conditioning_validation_paper_v6_provenance.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXP02 = ROOT / "04 Experiments" / "EXP02_GB1_ESM2"
EXP03 = ROOT / "04 Experiments" / "EXP03_MI"
FIGDIR = ROOT / "05 Figure"
FIGDIR.mkdir(parents=True, exist_ok=True)

PRIMARY_FILE = EXP02 / "gb1_target_mi_seed_results.csv"
POOLED_NULL_FILE = EXP02 / "gb1_random_derangement_null_by_derangement.csv"
ETA2_FILE = EXP02 / "gb1_subset_conditioned_diag_subset_effects.csv"
FIXED_S_FILE = EXP02 / "gb1_subset_conditioned_diag_comparison.csv"
CAL_REP_FILE = EXP03 / "fixed_subset_sample_regime_replicate_summary.csv"
CAL_GAIN_FILE = EXP03 / "fixed_subset_sample_regime_gain_summary.csv"

SOURCE_FILES = [
    PRIMARY_FILE,
    POOLED_NULL_FILE,
    ETA2_FILE,
    FIXED_S_FILE,
    CAL_REP_FILE,
    CAL_GAIN_FILE,
]


# Colorblind-friendly interaction-order palette (Okabe-Ito inspired).
# Color has scientific meaning: the same interaction order uses the same color
# across Panels A and C.
ORDER_COLORS = {
    1: "#0072B2",  # blue
    2: "#E69F00",  # orange
    3: "#009E73",  # bluish green
    4: "#CC79A7",  # reddish purple
}


def read_dicts(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = list(xs)
    if not xs:
        raise ValueError("mean() received an empty sequence")
    return sum(xs) / len(xs)


def jitter_offsets(n, width=0.14):
    if n <= 1:
        return [0.0]
    step = 2 * width / (n - 1)
    return [-width + i * step for i in range(n)]


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Load authoritative sources
# ---------------------------------------------------------------------

primary_all = read_dicts(PRIMARY_FILE)
primary_rows = [r for r in primary_all if r["family"] == "primary"]

pooled_null_rows = read_dicts(POOLED_NULL_FILE)
eta2_rows = read_dicts(ETA2_FILE)
fixed_s_rows = read_dicts(FIXED_S_FILE)
cal_rep_rows = read_dicts(CAL_REP_FILE)
cal_gain_rows = read_dicts(CAL_GAIN_FILE)

# ---------------------------------------------------------------------
# Hard audit checks
# ---------------------------------------------------------------------

assert len(primary_rows) > 0, "No primary rows found in pooled biological file"
assert len(pooled_null_rows) == 60, (
    f"Expected 60 pooled-null derangement rows, got {len(pooled_null_rows)}"
)
assert len(eta2_rows) == 12, (
    f"Expected 12 subset-effect rows, got {len(eta2_rows)}"
)
assert len(fixed_s_rows) == 45, (
    f"Expected 45 fixed-S diagnostic cells, got {len(fixed_s_rows)}"
)
assert len(cal_rep_rows) == 15, (
    f"Expected 15 calibration replicate rows, got {len(cal_rep_rows)}"
)
assert len(cal_gain_rows) == 5, (
    f"Expected 5 calibration gain-summary rows, got {len(cal_gain_rows)}"
)

diag_layers = sorted({int(r["layer"]) for r in pooled_null_rows})
assert diag_layers == [0, 15, 30], (
    f"Unexpected diagnostic layers in pooled-null source: {diag_layers}"
)

diag_orders = sorted({int(r["order_k"]) for r in pooled_null_rows})
assert diag_orders == [1, 2, 3, 4], (
    f"Unexpected interaction orders in pooled-null source: {diag_orders}"
)

# Verify primary pooled source contains all diagnostic layer/order cells.
primary_keys = {
    (int(r["layer"]), int(r["order_k"]))
    for r in primary_rows
}
null_keys = {
    (int(r["layer"]), int(r["order_k"]))
    for r in pooled_null_rows
}
assert null_keys.issubset(primary_keys), (
    "Pooled primary source is missing one or more diagnostic layer/order cells"
)

# Verify fixed-S subset cardinalities at each diagnostic layer.
expected_subset_counts = {1: 4, 2: 6, 3: 4, 4: 1}
for layer in [0, 15, 30]:
    for k, expected in expected_subset_counts.items():
        n = sum(
            1 for r in fixed_s_rows
            if int(r["layer"]) == layer and int(r["order_k"]) == k
        )
        assert n == expected, (
            f"Layer {layer}, k={k}: expected {expected} fixed-S rows, got {n}"
        )

# Verify exact fixed-S biological sample regime in calibration replicate file.
for r in cal_rep_rows:
    # replicate summary does not repeat counts; counts were audited in seed-level
    # source. Here we check only structure of the summary used for plotting.
    assert r["true_mi_nats"] != ""
    assert r["estimate_mean"] != ""

# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), layout="constrained")
axA, axB, axC, axD = axes.flatten()

# ============================================================
# A. Pooled estimator fails cube-derangement null
# ============================================================

# Arrange the 12 layer/order cells in three visually separated layer groups.
layer_gap = 1.0
keys_a = sorted(null_keys)
x_a = {}
for layer_index, layer in enumerate([0, 15, 30]):
    base = layer_index * (4 + layer_gap)
    for order_index, k in enumerate([1, 2, 3, 4]):
        x_a[(layer, k)] = base + order_index

primary_group = defaultdict(list)
for r in primary_rows:
    key = (int(r["layer"]), int(r["order_k"]))
    if key in x_a:
        primary_group[key].append(float(r["test_mi"]))

null_group = defaultdict(list)
for r in pooled_null_rows:
    key = (int(r["layer"]), int(r["order_k"]))
    null_group[key].append(float(r["score_mean"]))

for key in keys_a:
    layer, k = key
    x = x_a[key]
    pvals = primary_group[key]
    nvals = null_group[key]
    color = ORDER_COLORS[k]

    assert len(pvals) == 3, f"{key}: expected 3 primary critic seeds"
    assert len(nvals) == 5, f"{key}: expected 5 null derangements"

    # Color encodes interaction order. Marker/fill encodes primary vs null.
    for off, y in zip(jitter_offsets(len(pvals), 0.07), pvals):
        axA.scatter(
            x - 0.16 + off,
            y,
            s=18,
            marker="o",
            color=color,
            edgecolors="none",
            zorder=3,
        )

    for off, y in zip(jitter_offsets(len(nvals), 0.09), nvals):
        axA.scatter(
            x + 0.16 + off,
            y,
            s=20,
            marker="s",
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )

    # Mean markers computed directly from source rows.
    axA.plot(
        [x - 0.27, x - 0.05],
        [mean(pvals), mean(pvals)],
        linewidth=1.3,
        color=color,
        zorder=4,
    )
    axA.plot(
        [x + 0.05, x + 0.27],
        [mean(nvals), mean(nvals)],
        linewidth=1.3,
        color=color,
        linestyle="--",
        zorder=4,
    )

axA.axhline(0.0, linewidth=0.7)

ordered_positions = [x_a[key] for key in keys_a]
axA.set_xticks(ordered_positions)
axA.set_xticklabels(
    [f"L{layer}\nk{k}" for layer, k in keys_a],
    fontsize=6.2,
)
axA.set_ylabel("Neural dependence score", fontsize=7.3)
axA.set_title(
    "A  Pooled null persists after cube derangement",
    loc="left",
    fontsize=8.2,
    pad=5,
)

# Compact semantic legend: marker shape = family, color = order.
from matplotlib.lines import Line2D
family_handles = [
    Line2D([0], [0], marker="o", linestyle="None", color="black",
           markerfacecolor="black", markeredgecolor="black",
           markersize=4, label="Primary"),
    Line2D([0], [0], marker="s", linestyle="None", color="black",
           markerfacecolor="none", markeredgecolor="black",
           markersize=4, label="Deranged null"),
]
order_handles = [
    Line2D([0], [0], marker="o", linestyle="None", color=ORDER_COLORS[k],
           markersize=4, label=f"k{k}")
    for k in [1, 2, 3, 4]
]
leg1 = axA.legend(
    handles=family_handles,
    frameon=False,
    fontsize=5.7,
    loc="upper left",
    handletextpad=0.35,
    borderaxespad=0.2,
)
axA.add_artist(leg1)
axA.legend(
    handles=order_handles,
    frameon=False,
    fontsize=5.5,
    loc="upper right",
    ncol=2,
    handletextpad=0.25,
    columnspacing=0.5,
    borderaxespad=0.2,
)

# ============================================================
# B. Positional subset identity predicts target distribution
# ============================================================

split_order = ["estimator_train", "estimator_val", "estimator_test"]
split_label = {
    "estimator_train": "Train",
    "estimator_val": "Validation",
    "estimator_test": "Test",
}

for split in split_order:
    rows = [
        r for r in eta2_rows
        if r["split"] == split and r["subset_mean_effect_eta2"] != ""
    ]
    rows.sort(key=lambda r: int(r["order_k"]))

    xs = [int(r["order_k"]) for r in rows]
    ys = [float(r["subset_mean_effect_eta2"]) for r in rows]

    # k=4 is absent by construction because only one fourth-order subset exists.
    assert xs == [1, 2, 3], f"{split}: unexpected eta2 orders {xs}"
    axB.plot(xs, ys, marker="o", label=split_label[split])

axB.set_xticks([1, 2, 3])
axB.set_xlabel("Interaction order k")
axB.set_ylabel(r"Subset mean-effect $\eta^2$")
axB.set_ylim(bottom=0)
axB.set_title(
    "B  Subset identity predicts epistatic-target distribution",
    loc="left",
    fontsize=8.2,
)
axB.legend(frameon=False, fontsize=8)

# ============================================================
# C. Fixed-S conditioning collapses randomized null
# ============================================================

fixed_group = defaultdict(list)
for r in fixed_s_rows:
    key = (int(r["layer"]), int(r["order_k"]))
    fixed_group[key].append(float(r["random_null_mean"]))

keys_c = sorted(fixed_group)
x_c = {}
for layer_index, layer in enumerate([0, 15, 30]):
    base = layer_index * (4 + layer_gap)
    for order_index, k in enumerate([1, 2, 3, 4]):
        x_c[(layer, k)] = base + order_index

for key in keys_c:
    layer, k = key
    x = x_c[key]
    vals = fixed_group[key]
    color = ORDER_COLORS[k]

    for off, y in zip(jitter_offsets(len(vals), 0.12), vals):
        axC.scatter(
            x + off,
            y,
            s=18,
            marker="o",
            color=color,
            edgecolors="none",
            zorder=3,
        )

    axC.plot(
        [x - 0.18, x + 0.18],
        [mean(vals), mean(vals)],
        linewidth=1.2,
        color=color,
        zorder=4,
    )

axC.axhline(0.0, linewidth=0.7)
ordered_positions_c = [x_c[key] for key in keys_c]
axC.set_xticks(ordered_positions_c)
axC.set_xticklabels(
    [f"L{layer}\nk{k}" for layer, k in keys_c],
    fontsize=6.2,
)
axC.set_ylabel("Cube-deranged null score", fontsize=7.3)
axC.set_title(
    "C  Fixed-S conditioning collapses deranged null",
    loc="left",
    fontsize=8.2,
    pad=5,
)

# ============================================================
# D. Exact-sample calibration preserves strong-signal detection
# ============================================================

rep_group = defaultdict(list)
for r in cal_rep_rows:
    x = float(r["true_mi_nats"])
    rep_group[x].append(float(r["estimate_mean"]))

rep_x_all, rep_y_all = [], []
for x in sorted(rep_group):
    vals = rep_group[x]
    assert len(vals) == 3, f"true MI {x}: expected 3 calibration replicates"
    rep_x_all.extend([x + off for off in jitter_offsets(len(vals), 0.006)])
    rep_y_all.extend(vals)

axD.scatter(
    rep_x_all,
    rep_y_all,
    s=18,
    marker="o",
    label="Calibration replicates",
)

summary_rows = sorted(
    cal_gain_rows,
    key=lambda r: float(r["true_mi_nats"]),
)
summary_x = [float(r["true_mi_nats"]) for r in summary_rows]
summary_y = [float(r["estimate_mean_across_replicates"]) for r in summary_rows]
summary_sd = [float(r["estimate_sd_across_replicates"]) for r in summary_rows]

axD.errorbar(
    summary_x,
    summary_y,
    yerr=summary_sd,
    marker="o",
    linewidth=1.1,
    capsize=2,
)

axD.axhline(0.0, linewidth=0.8)
axD.set_xlabel("True synthetic mutual information (nats)")
axD.set_ylabel("Estimated neural dependence score")
axD.set_title(
    "D  Exact-sample calibration preserves strong-signal detection",
    loc="left",
    fontsize=8.2,
)

# ---------------------------------------------------------------------
# Final layout
# ---------------------------------------------------------------------

pdf_path = FIGDIR / "Figure_subset_conditioning_validation_paper_v6.pdf"
png_path = FIGDIR / "Figure_subset_conditioning_validation_paper_v6.png"
prov_path = FIGDIR / "Figure_subset_conditioning_validation_paper_v6_provenance.json"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------

provenance = {
    "figure": "Figure_subset_conditioning_validation_paper_v6",
    "principle": "No plotted numerical result is hard-coded; all values are read from source CSV outputs.",
    "sources": {},
    "panel_sources": {
        "A": [
            str(PRIMARY_FILE.relative_to(ROOT)),
            str(POOLED_NULL_FILE.relative_to(ROOT)),
        ],
        "B": [str(ETA2_FILE.relative_to(ROOT))],
        "C": [str(FIXED_S_FILE.relative_to(ROOT))],
        "D": [
            str(CAL_REP_FILE.relative_to(ROOT)),
            str(CAL_GAIN_FILE.relative_to(ROOT)),
        ],
    },
    "outputs": [
        str(pdf_path.relative_to(ROOT)),
        str(png_path.relative_to(ROOT)),
    ],
}

for p in SOURCE_FILES:
    provenance["sources"][str(p.relative_to(ROOT))] = {
        "sha256": sha256_file(p),
        "rows": len(read_dicts(p)),
    }

with prov_path.open("w") as f:
    json.dump(provenance, f, indent=2)

print("FIGURE BUILD: PASS")
print("PDF:", pdf_path)
print("PNG:", png_path)
print("PROVENANCE:", prov_path)
