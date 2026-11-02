#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_NAME = "ORBIT-FMIB CODE 15 — OVERLEAF RESULTS EXPORT"
SCHEMA_VERSION = "1.0"

EXPECTED_ORDERS = (1, 2, 3, 4)
EXPECTED_SUBSETS = tuple(range(1, 16))


# =============================================================================
# Utilities
# =============================================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def latex_num(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def latex_bool(x: bool) -> str:
    return "Yes" if x else "No"


# =============================================================================
# Load authoritative inputs
# =============================================================================

def load_inputs(root: Path):
    iid = (
        root
        / "04 Experiments"
        / "EXP02_GB1_ESM2"
        / "IID_Production_20260910"
    )

    primary = iid / "Code13"
    sensitivity = (
        iid
        / "Code14"
        / "DeleteOneIdentityPilot"
        / "position41_L"
    )

    paths = {
        "primary_order":
            primary / "gb1_order_retention_summary.csv",
        "primary_contrast":
            primary / "gb1_retention_contrast.csv",
        "primary_subset":
            primary / "gb1_subset_retention.csv",
        "sensitivity_contrast":
            sensitivity / "delete_one_identity_contrast.json",
        "sensitivity_order":
            sensitivity / "delete_one_identity_order_retention.csv",
        "sensitivity_subset":
            sensitivity / "delete_one_identity_subset_retention.csv",
        "sensitivity_manifest":
            sensitivity / "delete_one_identity_manifest.json",
    }

    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")

    primary_order = read_csv(paths["primary_order"])
    primary_contrast_rows = read_csv(paths["primary_contrast"])
    primary_subset = read_csv(paths["primary_subset"])

    sensitivity_order = read_csv(paths["sensitivity_order"])
    sensitivity_subset = read_csv(paths["sensitivity_subset"])

    with paths["sensitivity_contrast"].open("r", encoding="utf-8") as f:
        sensitivity_contrast = json.load(f)

    with paths["sensitivity_manifest"].open("r", encoding="utf-8") as f:
        sensitivity_manifest = json.load(f)

    if len(primary_contrast_rows) != 1:
        raise RuntimeError(
            f"Expected one primary contrast row; got {len(primary_contrast_rows)}"
        )

    primary_contrast = primary_contrast_rows[0]

    return (
        paths,
        primary_order,
        primary_contrast,
        primary_subset,
        sensitivity_order,
        sensitivity_subset,
        sensitivity_contrast,
        sensitivity_manifest,
    )


# =============================================================================
# Audit
# =============================================================================

def audit_inputs(
    primary_order,
    primary_contrast,
    primary_subset,
    sensitivity_order,
    sensitivity_subset,
    sensitivity_contrast,
    sensitivity_manifest,
):
    po = {int(r["order_k"]): r for r in primary_order}
    so = {int(r["order_k"]): r for r in sensitivity_order}

    if tuple(sorted(po)) != EXPECTED_ORDERS:
        raise RuntimeError("Primary order summary does not contain k=1..4.")

    if tuple(sorted(so)) != EXPECTED_ORDERS:
        raise RuntimeError("Sensitivity order summary does not contain k=1..4.")

    ps_masks = tuple(sorted(int(r["subset_mask"]) for r in primary_subset))
    ss_masks = tuple(sorted(int(r["subset_mask"]) for r in sensitivity_subset))

    if ps_masks != EXPECTED_SUBSETS:
        raise RuntimeError("Primary subset masks are not exactly 1..15.")

    if ss_masks != EXPECTED_SUBSETS:
        raise RuntimeError("Sensitivity subset masks are not exactly 1..15.")

    for r in primary_order:
        if r["all_subsets_estimable"] != "True":
            raise RuntimeError(
                f"Primary order {r['order_k']} not fully estimable."
            )

    for r in sensitivity_order:
        if r["order_estimable"] != "True":
            raise RuntimeError(
                f"Sensitivity order {r['order_k']} not fully estimable."
            )

    for r in primary_subset:
        if r["retention_eligible"] != "True":
            raise RuntimeError(
                f"Primary subset {r['subset_mask']} not retention eligible."
            )
        if r["final_near_floor"] != "False":
            raise RuntimeError(
                f"Primary subset {r['subset_mask']} final score near floor."
            )

    for r in sensitivity_subset:
        if r["retention_eligible"] != "True":
            raise RuntimeError(
                f"Sensitivity subset {r['subset_mask']} not retention eligible."
            )
        if r["final_near_floor"] != "False":
            raise RuntimeError(
                f"Sensitivity subset {r['subset_mask']} final score near floor."
            )

    if primary_contrast["contrast_estimable"] != "True":
        raise RuntimeError("Primary contrast is not estimable.")

    if primary_contrast["inferential_test_performed"] != "False":
        raise RuntimeError(
            "Unexpected formal inferential test recorded in Code 13."
        )

    if not bool(sensitivity_contrast["contrast_estimable"]):
        raise RuntimeError("Sensitivity contrast is not estimable.")

    if sensitivity_manifest["status"] != "complete_pass":
        raise RuntimeError("Sensitivity manifest did not complete successfully.")

    if not bool(sensitivity_manifest["overall_pass"]):
        raise RuntimeError("Sensitivity manifest overall_pass is false.")

    if bool(sensitivity_manifest["included_in_formal_inference"]):
        raise RuntimeError(
            "Sensitivity pilot must remain excluded from formal inference."
        )

    if int(sensitivity_manifest["computational_audit"]["observed_seed_fits"]) != 1395:
        raise RuntimeError("Sensitivity pilot does not contain 1,395 fits.")

    return po, so


# =============================================================================
# Overleaf macros
# =============================================================================

def make_macros(
    primary_contrast,
    sensitivity_contrast,
    sensitivity_manifest,
) -> str:

    p = {k: float(primary_contrast[k]) for k in (
        "retain_1",
        "retain_2",
        "retain_3",
        "retain_4",
        "lo_retention",
        "ho_retention",
        "delta_ho_minus_lo",
    )}

    s = {k: float(sensitivity_contrast[k]) for k in (
        "retain_1",
        "retain_2",
        "retain_3",
        "retain_4",
        "lo_retention",
        "ho_retention",
        "delta_ho_minus_lo",
    )}

    counts = sensitivity_manifest["surviving_counts"]

    lines = [
        "% AUTO-GENERATED BY 03 Code/15_export_overleaf_results.py",
        "% DO NOT EDIT RESULT VALUES BY HAND.",
        "",
        rf"\providecommand{{\OrbitPrimaryRetainOne}}{{{latex_num(p['retain_1'])}}}",
        rf"\providecommand{{\OrbitPrimaryRetainTwo}}{{{latex_num(p['retain_2'])}}}",
        rf"\providecommand{{\OrbitPrimaryRetainThree}}{{{latex_num(p['retain_3'])}}}",
        rf"\providecommand{{\OrbitPrimaryRetainFour}}{{{latex_num(p['retain_4'])}}}",
        rf"\providecommand{{\OrbitPrimaryLowOrderRetention}}{{{latex_num(p['lo_retention'])}}}",
        rf"\providecommand{{\OrbitPrimaryHighOrderRetention}}{{{latex_num(p['ho_retention'])}}}",
        rf"\providecommand{{\OrbitPrimaryDelta}}{{{latex_num(p['delta_ho_minus_lo'])}}}",
        "",
        rf"\providecommand{{\OrbitSensitivityRetainOne}}{{{latex_num(s['retain_1'])}}}",
        rf"\providecommand{{\OrbitSensitivityRetainTwo}}{{{latex_num(s['retain_2'])}}}",
        rf"\providecommand{{\OrbitSensitivityRetainThree}}{{{latex_num(s['retain_3'])}}}",
        rf"\providecommand{{\OrbitSensitivityRetainFour}}{{{latex_num(s['retain_4'])}}}",
        rf"\providecommand{{\OrbitSensitivityLowOrderRetention}}{{{latex_num(s['lo_retention'])}}}",
        rf"\providecommand{{\OrbitSensitivityHighOrderRetention}}{{{latex_num(s['ho_retention'])}}}",
        rf"\providecommand{{\OrbitSensitivityDelta}}{{{latex_num(s['delta_ho_minus_lo'])}}}",
        "",
        rf"\providecommand{{\OrbitSensitivityTrainN}}{{{int(counts['train'])}}}",
        rf"\providecommand{{\OrbitSensitivityValN}}{{{int(counts['val'])}}}",
        rf"\providecommand{{\OrbitSensitivityTestN}}{{{int(counts['test'])}}}",
        rf"\providecommand{{\OrbitSensitivityTotalN}}{{{int(counts['all'])}}}",
        "",
        r"\providecommand{\OrbitSensitivityDeletion}{position 41 = L}",
        r"\providecommand{\OrbitFormalInferencePerformed}{No}",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# LaTeX tables
# =============================================================================

def make_order_table(po, so) -> str:
    lines = [
        "% AUTO-GENERATED. DO NOT EDIT RESULT VALUES BY HAND.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Order-level retention in the primary GB1 analysis and the "
        r"dependence-aware whole-identity deletion sensitivity analysis. "
        r"The sensitivity run removes all cubes containing the alternative "
        r"identity $41=L$ and is not used as a formal inferential replicate.}",
        r"\label{tab:order_retention_sensitivity}",
        r"\begin{tabular}{c c c}",
        r"\toprule",
        r"Interaction order $k$ & Primary retention & Delete $41=L$ \\",
        r"\midrule",
    ]

    for k in EXPECTED_ORDERS:
        lines.append(
            f"{k} & "
            f"{latex_num(float(po[k]['retain_k']))} & "
            f"{latex_num(float(so[k]['retain_k']))} \\\\"
        )

    lines += [
        r"\midrule",
        r"$\Delta_{\mathrm{HO-LO}}$ & "
        r"\OrbitPrimaryDelta & \OrbitSensitivityDelta \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    return "\n".join(lines)


def make_primary_subset_table(primary_subset) -> str:
    rows = sorted(primary_subset, key=lambda r: int(r["subset_mask"]))

    lines = [
        "% AUTO-GENERATED. DO NOT EDIT RESULT VALUES BY HAND.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Primary fixed-subset retention results. Peak layer is "
        r"reselected independently for each positional subset.}",
        r"\label{tab:subset_retention_primary}",
        r"\begin{tabular}{c c c c c c}",
        r"\toprule",
        r"Subset mask & Order $k$ & Peak layer & Peak score & Final score & Retention \\",
        r"\midrule",
    ]

    for r in rows:
        lines.append(
            f"{int(r['subset_mask'])} & "
            f"{int(r['order_k'])} & "
            f"{int(r['peak_layer'])} & "
            f"{latex_num(float(r['peak_score']))} & "
            f"{latex_num(float(r['final_score']))} & "
            f"{latex_num(float(r['retain_s']))} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]

    return "\n".join(lines)


def make_sensitivity_subset_table(sensitivity_subset) -> str:
    rows = sorted(sensitivity_subset, key=lambda r: int(r["subset_mask"]))

    lines = [
        "% AUTO-GENERATED. DO NOT EDIT RESULT VALUES BY HAND.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Whole-identity deletion sensitivity analysis after removing "
        r"all cubes containing $41=L$. These values are descriptive sensitivity "
        r"results and are not formal inferential replicates.}",
        r"\label{tab:subset_retention_sensitivity}",
        r"\begin{tabular}{c c c c c c}",
        r"\toprule",
        r"Subset mask & Order $k$ & Peak layer & Peak score & Final score & Retention \\",
        r"\midrule",
    ]

    for r in rows:
        lines.append(
            f"{int(r['subset_mask'])} & "
            f"{int(r['order_k'])} & "
            f"{int(r['peak_layer'])} & "
            f"{latex_num(float(r['peak_score']))} & "
            f"{latex_num(float(r['final_score']))} & "
            f"{latex_num(float(r['retain_s']))} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# Figures
# =============================================================================

def figure_order_retention(
    out: Path,
    po,
    so,
):
    ks = np.asarray(EXPECTED_ORDERS, dtype=int)
    primary = np.asarray(
        [float(po[k]["retain_k"]) for k in ks],
        dtype=float,
    )
    sensitivity = np.asarray(
        [float(so[k]["retain_k"]) for k in ks],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.5))

    ax.plot(
        ks,
        primary,
        marker="o",
        linewidth=1.8,
        label="Primary",
    )
    ax.plot(
        ks,
        sensitivity,
        marker="s",
        linewidth=1.8,
        label=r"Delete $41=L$",
    )

    ax.set_xlabel("Interaction order $k$")
    ax.set_ylabel("Final-layer retention")
    ax.set_xticks(ks)
    ax.set_ylim(0.60, 1.02)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_subset_primary(
    out: Path,
    primary_subset,
):
    rows = sorted(primary_subset, key=lambda r: int(r["subset_mask"]))

    masks = np.asarray(
        [int(r["subset_mask"]) for r in rows],
        dtype=int,
    )
    retention = np.asarray(
        [float(r["retain_s"]) for r in rows],
        dtype=float,
    )
    orders = np.asarray(
        [int(r["order_k"]) for r in rows],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    for k in EXPECTED_ORDERS:
        sel = orders == k
        ax.scatter(
            masks[sel],
            retention[sel],
            s=38,
            label=rf"$k={k}$",
        )

    ax.set_xlabel("Fixed positional subset mask")
    ax.set_ylabel("Final-layer retention")
    ax.set_xticks(masks)
    ax.set_ylim(0.70, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_subset_sensitivity_compare(
    out: Path,
    primary_subset,
    sensitivity_subset,
):
    p = {
        int(r["subset_mask"]): float(r["retain_s"])
        for r in primary_subset
    }
    s = {
        int(r["subset_mask"]): float(r["retain_s"])
        for r in sensitivity_subset
    }

    masks = np.asarray(EXPECTED_SUBSETS, dtype=int)
    primary = np.asarray([p[m] for m in masks], dtype=float)
    sensitivity = np.asarray([s[m] for m in masks], dtype=float)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))

    ax.scatter(primary, sensitivity, s=42)

    lo = min(primary.min(), sensitivity.min()) - 0.02
    hi = max(primary.max(), sensitivity.max()) + 0.02

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)

    for m, x, y in zip(masks, primary, sensitivity):
        ax.annotate(
            str(m),
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Primary subset retention")
    ax.set_ylabel(r"Retention after deleting $41=L$")
    ax.grid(alpha=0.20)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    root = project_root()

    (
        paths,
        primary_order,
        primary_contrast,
        primary_subset,
        sensitivity_order,
        sensitivity_subset,
        sensitivity_contrast,
        sensitivity_manifest,
    ) = load_inputs(root)

    po, so = audit_inputs(
        primary_order,
        primary_contrast,
        primary_subset,
        sensitivity_order,
        sensitivity_subset,
        sensitivity_contrast,
        sensitivity_manifest,
    )

    out = root / "05 Results" / "Overleaf"
    out.mkdir(parents=True, exist_ok=True)

    macros_path = out / "orbit_result_macros.tex"
    order_table_path = (
        out / "table_order_retention_primary_sensitivity.tex"
    )
    primary_subset_table_path = (
        out / "table_subset_retention_primary.tex"
    )
    sensitivity_subset_table_path = (
        out / "table_subset_retention_sensitivity.tex"
    )

    order_fig_path = (
        out / "figure_order_retention_primary_sensitivity.pdf"
    )
    subset_fig_path = (
        out / "figure_subset_retention_primary.pdf"
    )
    robustness_fig_path = (
        out / "figure_subset_retention_primary_vs_sensitivity.pdf"
    )

    manifest_path = out / "overleaf_export_manifest.json"

    write_text(
        macros_path,
        make_macros(
            primary_contrast,
            sensitivity_contrast,
            sensitivity_manifest,
        ),
    )

    write_text(
        order_table_path,
        make_order_table(po, so),
    )

    write_text(
        primary_subset_table_path,
        make_primary_subset_table(primary_subset),
    )

    write_text(
        sensitivity_subset_table_path,
        make_sensitivity_subset_table(sensitivity_subset),
    )

    figure_order_retention(
        order_fig_path,
        po,
        so,
    )

    figure_subset_primary(
        subset_fig_path,
        primary_subset,
    )

    figure_subset_sensitivity_compare(
        robustness_fig_path,
        primary_subset,
        sensitivity_subset,
    )

    outputs = [
        macros_path,
        order_table_path,
        primary_subset_table_path,
        sensitivity_subset_table_path,
        order_fig_path,
        subset_fig_path,
        robustness_fig_path,
    ]

    manifest = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_pass",
        "overall_pass": True,
        "code15_source_sha256": sha256_file(Path(__file__)),
        "policy": {
            "paper_result_values_manually_entered": False,
            "tables_generated_directly_from_authoritative_outputs": True,
            "figures_generated_directly_from_authoritative_outputs": True,
            "latex_macros_generated_directly_from_authoritative_outputs": True,
            "sensitivity_analysis_is_formal_inference": False,
        },
        "authoritative_inputs": {
            label: {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
            for label, path in paths.items()
        },
        "outputs": {
            path.name: {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        },
    }

    write_json(manifest_path, manifest)

    print(f"{CODE_NAME}: PASS")
    print("Authoritative result inputs:", len(paths))
    print("Generated Overleaf artifacts:", len(outputs))
    print("Manual result-value entry: NO")
    print("Formal inference claimed: NO")
    print(f"Output directory: {out}")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
