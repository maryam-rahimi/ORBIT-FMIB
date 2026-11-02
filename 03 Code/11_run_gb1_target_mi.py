#!/usr/bin/env python3
"""ORBIT-FMIB CODE 11 — GB1 target-aligned order-resolved neural-MI scores.

Combines Code-09 frozen ESM-2 site representations and Code-10 experimental
Walsh coefficients to estimate held-out MINE/Donsker–Varadhan lower-bound
scores for I(R_{l,k}; E_k), layers 0..30 and interaction orders 1..4.

The four saved site vectors are concatenated, so R is 4*640 = 2560-D.
No PCA, target-supervised projection, clamping, or null subtraction is used.

Important interpretation contract
---------------------------------
The original 2560-D *absolute-MI* calibration showed substantial
underestimation and therefore failed.  Absolute MI accuracy in nats is NOT
claimed.  Before any biological GB1 critic is trained, this revised Code 11
requires two independent 2560-D validation manifests generated with the exact
live Code-04 estimator:

  1. relative-retention trajectory calibration: PASS;
  2. hypothesis-adversarial retention suite (negative / flat / positive): PASS.

These validations establish suitability for comparative layerwise/order-
retention analysis under the frozen synthetic contract, not exact absolute MI.

Recommended sequence:
  python "03 Code/11_run_gb1_target_mi.py" --self-test
  python "03 Code/11_run_gb1_target_mi.py" --preflight-only

On Newton / GPU, after preflight PASS:
  python "03 Code/11_run_gb1_target_mi.py" --device cuda --overwrite

Use --overwrite only for an intentional clean start.  It removes Code-11
biological state/results but deliberately preserves the historical failed
absolute-calibration files and the standalone validation artifacts in EXP03_MI.

If a revised biological run is interrupted:
  python "03 Code/11_run_gb1_target_mi.py" --device cuda --resume
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, os, sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

CODE_NAME = "ORBIT-FMIB CODE 11"
SCHEMA_VERSION = 2
EXPECTED_CODE04 = "ORBIT-FMIB CODE 04"
EXPECTED_CODE05 = "ORBIT-FMIB CODE 05"
EXPECTED_CODE08 = "ORBIT-FMIB CODE 08"
EXPECTED_CODE09 = "ORBIT-FMIB CODE 09"
EXPECTED_CODE10 = "ORBIT-FMIB CODE 10"
EXPECTED_RELATIVE_VALIDATION = "ORBIT-FMIB RELATIVE-RETENTION CALIBRATION"
EXPECTED_ADVERSARIAL_VALIDATION = "ORBIT-FMIB ADVERSARIAL RELATIVE-RETENTION SUITE"

# Both standalone 2560-D validations were executed against this pre-revision
# Code-11 source.  The validations only depended on Code-11's frozen
# REPRESENTATION_DIM and CRITIC_SEEDS contract, which remain unchanged here.
CALIBRATION_REFERENCE_CODE11_SHA256 = "37deee5565d2d9423c0fb2c37b27ac766bece969f229b589e228c76b4aab0fae"

D = 4
N_VERTICES = 16
POSITIONS = (39, 40, 41, 54)
REPR_LAYERS = tuple(range(31))
ORDERS = (1, 2, 3, 4)
EMBED_DIM = 640
N_SITES = 4
REPRESENTATION_DIM = 2560
N_EXTRACT = 13_150
CRITIC_SEEDS = (11, 23, 37)

SPLIT_ORDER = ("estimator_train", "estimator_val", "estimator_test")
SPLIT_CODES = {"estimator_train": 0, "estimator_val": 1, "estimator_test": 2}
STRICT_CUBE_COUNTS = {"estimator_train": 7_352, "estimator_val": 245, "estimator_test": 613}
MAP_KEYS = {
    "estimator_train": "estimator_train_extract_rows",
    "estimator_val": "estimator_val_extract_rows",
    "estimator_test": "estimator_test_extract_rows",
}
SUBSETS_BY_ORDER = {k: tuple(m for m in range(1,16) if int(m).bit_count()==k) for k in ORDERS}
EXPECTED_TOTAL_TARGET_SAMPLES = 123_150

NULL_LAYERS = (0, 15, 30)
NULL_CUBE_SHIFT = 1

EXPECTED_PRIMARY_FITS = 31*4*3
EXPECTED_NULL_FITS = 3*4*3
EXPECTED_BIOLOGICAL_FITS = EXPECTED_PRIMARY_FITS + EXPECTED_NULL_FITS


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Mapping) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",",":")).encode()).hexdigest()


def load_json(path: Path) -> Dict:
    if not path.exists(): raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f: return json.load(f)


def write_json_atomic(path: Path, obj: Mapping) -> None:
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix+".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists(): raise FileNotFoundError(f"Missing {label}: {path}")
    got = sha256_file(path)
    if got.lower() != str(expected).lower():
        raise RuntimeError(f"{label} SHA256 mismatch.\nExpected: {expected}\nObserved: {got}\nPath: {path}")


def import_code04(path: Path):
    if not path.exists(): raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_orbitfmib_code04_exact_code11", path)
    if spec is None or spec.loader is None: raise RuntimeError("Could not import Code 04.")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    required = (
        "CODE_NAME","import_torch","choose_device","train_mine","TrainingConfig",
        "CRITIC_SEEDS","N_TOTAL","N_TRAIN","N_VAL","N_TEST",
    )
    missing = [x for x in required if not hasattr(mod,x)]
    if missing: raise RuntimeError(f"Code-04 source missing required objects: {missing}")
    if mod.CODE_NAME != EXPECTED_CODE04: raise RuntimeError("Wrong Code-04 source.")
    if tuple(map(int,mod.CRITIC_SEEDS)) != CRITIC_SEEDS: raise RuntimeError("Code-04 critic seeds changed.")
    return mod


def chi_matrix() -> np.ndarray:
    idx = np.arange(16, dtype=np.int64)[:,None]
    bits = (idx >> np.arange(4,dtype=np.int64)) & 1
    pts = np.where(bits==1,1.0,-1.0)
    X = np.ones((16,16),dtype=np.float64)
    for s in range(16):
        cols = [b for b in range(4) if (s>>b)&1]
        if cols: X[:,s] = np.prod(pts[:,cols],axis=1)
    if not np.array_equal(X.T@X,16*np.eye(16)): raise RuntimeError("Walsh basis orthogonality failed.")
    return X


def verify_estimator_chain(root: Path, source04: Path, manifest04: Path, manifest05: Path) -> Dict:
    mine04 = import_code04(source04)
    sha04src = sha256_file(source04)
    m04 = load_json(manifest04); sha04 = sha256_file(manifest04)
    if m04.get("code_name") != EXPECTED_CODE04 or not bool(m04.get("overall_pass",False)):
        raise RuntimeError("Code-04 manifest is not a PASS.")
    if not bool(m04.get("acceptance_results",{}).get("overall_pass",False)):
        raise RuntimeError("Code-04 acceptance_results did not pass.")
    if tuple(map(int,m04.get("critic_seeds",[]))) != CRITIC_SEEDS:
        raise RuntimeError("Code-04 manifest critic seeds changed.")
    if m04.get("training_config",{}) != asdict(mine04.TrainingConfig()):
        raise RuntimeError("Code-04 manifest/source TrainingConfig mismatch.")

    m05 = load_json(manifest05); sha05 = sha256_file(manifest05)
    if m05.get("code_name") != EXPECTED_CODE05 or not bool(m05.get("overall_pass",False)):
        raise RuntimeError("Code-05 synthetic end-to-end manifest is not a PASS.")
    if not bool(m05.get("acceptance_results",{}).get("overall_pass",False)):
        raise RuntimeError("Code-05 acceptance_results did not pass.")
    up = m05.get("upstream_provenance",{})
    if str(up.get("code04_manifest_sha256","")).lower() != sha04.lower():
        raise RuntimeError("Code 05 does not pin the current Code-04 manifest.")
    if str(up.get("code04_source_sha256","")).lower() != sha04src.lower():
        raise RuntimeError("Code 05 does not pin the current Code-04 source bytes.")
    return {"mine04":mine04,"m04":m04,"sha04":sha04,"sha04src":sha04src,"m05":m05,"sha05":sha05}


def _require_all_declared_hard_gates(gates: Mapping, label: str) -> None:
    hard = gates.get("hard_gates")
    if not isinstance(hard, list) or not hard:
        raise RuntimeError(f"{label} manifest does not declare hard_gates.")
    failed = [name for name in hard if not bool(gates.get(name, False))]
    if failed:
        raise RuntimeError(f"{label} hard gates are not all PASS: {failed}")


def _verify_validation_estimator_protocol(m: Mapping, est: Mapping, label: str) -> None:
    p = m.get("estimator_protocol", {})
    if str(p.get("code04_source_sha256", "")).lower() != str(est["sha04src"]).lower():
        raise RuntimeError(f"{label} was not run with the current exact Code-04 source.")
    if str(p.get("code11_source_sha256", "")).lower() != CALIBRATION_REFERENCE_CODE11_SHA256.lower():
        raise RuntimeError(f"{label} does not pin the frozen calibration-reference Code-11 source.")
    if tuple(map(int, p.get("critic_seeds", []))) != CRITIC_SEEDS:
        raise RuntimeError(f"{label} critic seeds mismatch.")
    expected_counts = {
        "n_total": int(est["mine04"].N_TOTAL),
        "n_train": int(est["mine04"].N_TRAIN),
        "n_val": int(est["mine04"].N_VAL),
        "n_test": int(est["mine04"].N_TEST),
    }
    for key, val in expected_counts.items():
        if int(p.get(key, -1)) != val:
            raise RuntimeError(f"{label} estimator count mismatch for {key}.")
    if bool(p.get("absolute_mi_accuracy_claimed", True)):
        raise RuntimeError(f"{label} incorrectly claims absolute MI accuracy.")
    for key in ("no_pca_projection", "no_clamping", "no_null_subtraction"):
        if not bool(p.get(key, False)):
            raise RuntimeError(f"{label} must certify {key}=true.")


def verify_relative_validation(path: Path, est: Mapping) -> Dict:
    m = load_json(path); sha = sha256_file(path)
    if m.get("code_name") != EXPECTED_RELATIVE_VALIDATION or m.get("status") != "PASS":
        raise RuntimeError("Relative-retention validation manifest is not a PASS.")
    _verify_validation_estimator_protocol(m, est, "Relative-retention validation")
    c = m.get("synthetic_contract", {})
    if int(c.get("representation_dim", -1)) != REPRESENTATION_DIM or int(c.get("target_dim", -1)) != 1:
        raise RuntimeError("Relative-retention validation dimension contract mismatch.")
    g = m.get("gates", {})
    _require_all_declared_hard_gates(g, "Relative-retention validation")
    if not bool(g.get("overall_pass", False)):
        raise RuntimeError("Relative-retention validation overall_pass is false.")
    required = (
        "null_abs_le_code04_tolerance",
        "seed_stability",
        "common_trajectory_monotonic",
        "mean_peak_localized",
        "signal_separation_ge_min",
        "estimated_retention_order_k1_gt_k2_gt_k3_gt_k4",
        "delta_sign_recovered",
        "all_seed_deltas_negative",
        "all_summary_diagnostics_finite",
        "all_retention_diagnostics_finite",
    )
    missing_or_failed = [x for x in required if not bool(g.get(x, False))]
    if missing_or_failed:
        raise RuntimeError(f"Relative-retention validation required gates failed/missing: {missing_or_failed}")
    return {"manifest": m, "sha": sha}


def verify_adversarial_validation(path: Path, est: Mapping) -> Dict:
    m = load_json(path); sha = sha256_file(path)
    if m.get("code_name") != EXPECTED_ADVERSARIAL_VALIDATION or m.get("status") != "PASS":
        raise RuntimeError("Adversarial relative-retention validation manifest is not a PASS.")
    _verify_validation_estimator_protocol(m, est, "Adversarial relative-retention validation")
    c = m.get("synthetic_contract", {})
    if int(c.get("representation_dim", -1)) != REPRESENTATION_DIM or int(c.get("target_dim", -1)) != 1:
        raise RuntimeError("Adversarial validation dimension contract mismatch.")
    expected_scenarios = {
        "ho_loss": ("negative", (1.20, 1.00, 0.50, 0.15)),
        "flat_strong": ("flat", (1.00, 1.00, 1.00, 1.00)),
        "ho_advantage": ("positive", (0.15, 0.50, 1.00, 1.20)),
        "flat_moderate": ("flat", (0.50, 0.50, 0.50, 0.50)),
    }
    scenarios = c.get("scenarios", {})
    if set(scenarios) != set(expected_scenarios):
        raise RuntimeError("Adversarial validation scenario set mismatch.")
    for name, (expected_label, gains) in expected_scenarios.items():
        cfg = scenarios.get(name, {})
        if cfg.get("expected") != expected_label:
            raise RuntimeError(f"Adversarial scenario {name} expected-label mismatch.")
        if not np.allclose(np.asarray(cfg.get("gains", []), dtype=float), np.asarray(gains), atol=0, rtol=0):
            raise RuntimeError(f"Adversarial scenario {name} gains mismatch.")
    flat = c.get("flat_test", {})
    if not np.isclose(float(flat.get("alpha", np.nan)), 0.05, atol=0, rtol=0):
        raise RuntimeError("Adversarial flat-case permutation alpha mismatch.")
    if "24-permutation" not in str(flat.get("method", "")):
        raise RuntimeError("Adversarial flat-case exact 24-permutation test is not declared.")

    gates = m.get("gates", {})
    gg = gates.get("global", {})
    _require_all_declared_hard_gates(gg, "Adversarial global validation")
    if not bool(gg.get("overall_pass", False)):
        raise RuntimeError("Adversarial global validation overall_pass is false.")
    sg = gates.get("scenarios", {})
    if set(sg) != set(expected_scenarios):
        raise RuntimeError("Adversarial scenario gate set mismatch.")
    for name in expected_scenarios:
        _require_all_declared_hard_gates(sg[name], f"Adversarial scenario {name}")
        if not bool(sg[name].get("overall_pass", False)):
            raise RuntimeError(f"Adversarial scenario {name} overall_pass is false.")
    return {"manifest": m, "sha": sha}


def verify_code08(path: Path) -> Dict:
    m = load_json(path); sha = sha256_file(path)
    if m.get("code_name") != EXPECTED_CODE08 or not bool(m.get("overall_pass",False)):
        raise RuntimeError("Code-08 manifest is not a PASS.")
    a = m.get("audit",{})
    if not bool(a.get("overall_pass",False)) or int(a.get("union_unique_sequences",-1)) != N_EXTRACT:
        raise RuntimeError("Code-08 audit/union contract failed.")
    if int(a.get("cross_split_nonwt_pfm_row_overlap",-1)) != 0:
        raise RuntimeError("Code-08 non-WT leakage is not zero.")
    if bool(a.get("fitness_present_in_extraction_index",True)) or bool(a.get("mixed_cubes_included",True)):
        raise RuntimeError("Code-08 label/mixed boundary failed.")
    c = m.get("esm2_contract",{})
    if c.get("model_name") != "esm2_t30_150M_UR50D" or int(c.get("embedding_dim",-1)) != 640:
        raise RuntimeError("Code-08 ESM-2 contract mismatch.")
    if tuple(map(int,c.get("representation_layers",[]))) != REPR_LAYERS:
        raise RuntimeError("Code-08 layer contract mismatch.")
    if tuple(map(int,c.get("biological_positions_1based",[]))) != POSITIONS:
        raise RuntimeError("Code-08 position contract mismatch.")
    name = m.get("outputs",{}).get("strict_cube_extract_rows")
    if not name: raise RuntimeError("Code-08 map output missing.")
    map_path = path.parent/str(name)
    expected = m.get("file_sha256",{}).get(map_path.name)
    if not expected: raise RuntimeError("Code-08 map hash missing.")
    verify_hash(map_path, expected, "Code-08 cube extraction map")
    return {"manifest":m,"sha":sha,"map_path":map_path,"map_sha":str(expected)}


def load_cube_maps(path: Path) -> Dict[str,np.ndarray]:
    out = {}
    with np.load(path,allow_pickle=False) as z:
        for s in SPLIT_ORDER:
            key = MAP_KEYS[s]
            if key not in z.files: raise RuntimeError(f"Code-08 map missing {key}.")
            a = np.asarray(z[key],dtype=np.int64)
            exp = (STRICT_CUBE_COUNTS[s],16)
            if a.shape != exp: raise RuntimeError(f"{s} map shape {a.shape} != {exp}.")
            if a.min()<0 or a.max()>=N_EXTRACT: raise RuntimeError(f"{s} map rows out of range.")
            out[s] = a
    wt = {int(a[0,0]) for a in out.values()}
    if len(wt)!=1: raise RuntimeError("Strict maps do not share one WT extraction row.")
    w = next(iter(wt))
    for s,a in out.items():
        if not np.all(a[:,0]==w): raise RuntimeError(f"{s} vertex 0 is not common WT.")
    return out


def verify_code09(path: Path, code08_sha: str, do_hashes: bool=True) -> Dict:
    m=load_json(path); sha=sha256_file(path)
    if m.get("code_name")!=EXPECTED_CODE09 or not bool(m.get("overall_pass",False)):
        raise RuntimeError("Code-09 manifest is not a PASS.")
    if str(m.get("upstream",{}).get("code08_manifest_sha256","")).lower()!=code08_sha.lower():
        raise RuntimeError("Code-09 does not pin current Code 08.")
    model=m.get("model",{}); ext=m.get("extraction",{})
    if model.get("name")!="esm2_t30_150M_UR50D" or tuple(map(int,model.get("representation_layers",[])))!=REPR_LAYERS:
        raise RuntimeError("Code-09 model/layer contract mismatch.")
    if int(model.get("embedding_dim",-1))!=640 or not bool(model.get("eval_mode",False)) or not bool(model.get("all_parameters_frozen",False)):
        raise RuntimeError("Code-09 frozen-model contract mismatch.")
    if int(ext.get("n_sequences",-1))!=N_EXTRACT or tuple(map(int,ext.get("saved_shape_per_layer",[])))!=(N_EXTRACT,4,640):
        raise RuntimeError("Code-09 extraction shape/count mismatch.")
    if ext.get("saved_dtype")!="float16" or bool(ext.get("fitness_read_or_used",True)) or bool(ext.get("mixed_cubes_included",True)):
        raise RuntimeError("Code-09 dtype/label/mixed contract mismatch.")
    layer_names=m.get("outputs",{}).get("layer_files",{}); hashes=m.get("file_sha256",{})
    paths={}
    for l in REPR_LAYERS:
        name=layer_names.get(str(l))
        if not name: raise RuntimeError(f"Code-09 layer {l} name missing.")
        p=path.parent/str(name); expected=hashes.get(p.name)
        if not expected: raise RuntimeError(f"Code-09 layer {l} hash missing.")
        if do_hashes: verify_hash(p,expected,f"Code-09 layer {l}")
        a=np.load(p,mmap_mode="r",allow_pickle=False)
        if a.shape!=(N_EXTRACT,4,640) or a.dtype!=np.float16: raise RuntimeError(f"Code-09 layer {l} shape/dtype mismatch.")
        paths[l]=p
    return {"manifest":m,"sha":sha,"layer_paths":paths}


def verify_code10(path: Path, m08: Mapping) -> Dict:
    m=load_json(path); sha=sha256_file(path)
    if m.get("code_name")!=EXPECTED_CODE10 or not bool(m.get("overall_pass",False)):
        raise RuntimeError("Code-10 manifest is not a PASS.")
    a=m.get("audit",{})
    if not bool(a.get("overall_pass",False)) or int(a.get("strict_cubes",-1))!=8210 or int(a.get("total_order_samples",-1))!=123150:
        raise RuntimeError("Code-10 audit/count contract failed.")
    if not bool(a.get("all_targets_finite",False)) or bool(a.get("esm2_representations_used",True)) or bool(a.get("mixed_cubes_used",True)):
        raise RuntimeError("Code-10 target/mixed boundary failed.")
    w=m.get("walsh_contract",{})
    if tuple(map(int,w.get("biological_positions",[])))!=POSITIONS or list(w.get("vertex_mask_order",[]))!=list(range(16)):
        raise RuntimeError("Code-10 Walsh position/vertex order mismatch.")
    if w.get("sign_convention")!="bit 0 / WT -> -1; bit 1 / mutation -> +1":
        raise RuntimeError("Code-10 Walsh sign convention mismatch.")
    u08=m08.get("upstream",{}); u10=m.get("upstream",{})
    for key in ("code06_manifest_sha256","code07_manifest_sha256"):
        if str(u08.get(key,"")).lower()!=str(u10.get(key,"")).lower(): raise RuntimeError(f"Code08/10 provenance mismatch: {key}")
    name=m.get("outputs",{}).get("order_samples")
    if not name: raise RuntimeError("Code-10 order_samples output missing.")
    p=path.parent/str(name); expected=m.get("file_sha256",{}).get(p.name)
    if not expected: raise RuntimeError("Code-10 order_samples hash missing.")
    verify_hash(p,expected,"Code-10 order samples")
    return {"manifest":m,"sha":sha,"samples_path":p,"samples_sha":str(expected)}


def load_order_samples(path: Path) -> Dict:
    keys=("sample_id","split_code","cube_id","subset_mask","order_k","fitness_walsh")
    with np.load(path,allow_pickle=False) as z:
        missing=[k for k in keys if k not in z.files]
        if missing: raise RuntimeError(f"Code-10 samples missing {missing}")
        d={k:np.asarray(z[k]) for k in keys}
    n=len(d["sample_id"])
    if n!=123150 or any(len(v)!=n for v in d.values()): raise RuntimeError("Code-10 sample arrays misaligned.")
    if not np.array_equal(d["sample_id"],np.arange(n,dtype=np.int64)): raise RuntimeError("Code-10 sample_id not contiguous.")
    if not np.isfinite(d["fitness_walsh"]).all(): raise RuntimeError("Non-finite Code-10 targets.")
    cube_orders={}
    for s in SPLIT_ORDER:
        ref=None; nc=STRICT_CUBE_COUNTS[s]; code=SPLIT_CODES[s]
        for k in ORDERS:
            masks=SUBSETS_BY_ORDER[k]; sel=(d["split_code"]==code)&(d["order_k"]==k)
            cubes=d["cube_id"][sel].astype(np.int64).reshape(nc,len(masks))
            subs=d["subset_mask"][sel].astype(np.int64).reshape(nc,len(masks))
            if not np.all(cubes==cubes[:,[0]]): raise RuntimeError(f"{s},k{k}: cube-major order failed.")
            if not np.array_equal(subs,np.repeat(np.asarray(masks)[None,:],nc,axis=0)): raise RuntimeError(f"{s},k{k}: subset order failed.")
            order=cubes[:,0]
            if np.any(np.diff(order)<=0): raise RuntimeError(f"{s},k{k}: cube IDs not increasing.")
            if ref is None: ref=order
            elif not np.array_equal(ref,order): raise RuntimeError(f"{s}: cube order differs by k.")
        cube_orders[s]=ref
    d["cube_ids_by_split"]=cube_orders
    return d


def compute_repr_walsh(layer_array: np.ndarray, cube_rows: np.ndarray, masks: Sequence[int], X: np.ndarray, chunk: int) -> np.ndarray:
    if layer_array.shape!=(N_EXTRACT,4,640): raise RuntimeError("Layer shape mismatch.")
    weights=(X[:,tuple(masks)]/16.0).astype(np.float32)
    nc=len(cube_rows); m=len(masks)
    out=np.empty((nc*m,REPRESENTATION_DIM),dtype=np.float32); pos=0
    for start in range(0,nc,chunk):
        stop=min(nc,start+chunk); rows=cube_rows[start:stop]
        h=np.asarray(layer_array[rows],dtype=np.float32).reshape(stop-start,16,REPRESENTATION_DIM)
        r=np.einsum("vm,cvd->cmd",weights,h,optimize=True)
        if not np.isfinite(r).all(): raise RuntimeError("Non-finite representation Walsh values.")
        block=r.reshape(-1,REPRESENTATION_DIM); out[pos:pos+len(block)]=block; pos+=len(block)
    if pos!=len(out): raise RuntimeError("Representation Walsh write-count mismatch.")
    return out


def targets_for(d: Mapping, split: str, k: int) -> np.ndarray:
    sel=(d["split_code"]==SPLIT_CODES[split])&(d["order_k"]==k)
    v=np.asarray(d["fitness_walsh"][sel],dtype=np.float32)[:,None]
    exp=STRICT_CUBE_COUNTS[split]*len(SUBSETS_BY_ORDER[k])
    if v.shape!=(exp,1): raise RuntimeError(f"{split},k{k}: target count mismatch.")
    return v


def cube_shift_null(v: np.ndarray, n_cubes: int, n_subsets: int) -> np.ndarray:
    if v.shape!=(n_cubes*n_subsets,1): raise RuntimeError("Null target shape mismatch.")
    return np.roll(v.reshape(n_cubes,n_subsets,1),shift=NULL_CUBE_SHIFT,axis=0).reshape(-1,1).copy()


def build_dataset(layer: np.ndarray, maps: Mapping[str,np.ndarray], samples: Mapping, k: int, X: np.ndarray, chunk: int):
    us=[]; vs=[]; idxs=[]; off=0; counts={}
    for s in SPLIT_ORDER:
        u=compute_repr_walsh(layer,maps[s],SUBSETS_BY_ORDER[k],X,chunk)
        v=targets_for(samples,s,k)
        if len(u)!=len(v): raise RuntimeError(f"{s},k{k}: R/E count mismatch.")
        us.append(u); vs.append(v); idxs.append(np.arange(off,off+len(u),dtype=np.int64)); counts[s]=len(u); off+=len(u)
    U=np.concatenate(us); V=np.concatenate(vs)
    if U.shape!=(off,2560) or V.shape!=(off,1) or not np.isfinite(U).all() or not np.isfinite(V).all():
        raise RuntimeError("Combined layer/order dataset failed shape/finiteness audit.")
    if any(len(np.intersect1d(idxs[i],idxs[j])) for i in range(3) for j in range(i+1,3)): raise RuntimeError("Estimator split overlap.")
    return U,V,(idxs[0],idxs[1],idxs[2]),counts


def build_null(samples: Mapping, k: int) -> np.ndarray:
    pieces=[]; m=len(SUBSETS_BY_ORDER[k])
    for s in SPLIT_ORDER: pieces.append(cube_shift_null(targets_for(samples,s,k),STRICT_CUBE_COUNTS[s],m))
    return np.concatenate(pieces)


BIO_COLS=("family","layer","order_k","critic_seed","representation_dim","target_dim","n_samples","n_train","n_val","n_test","subset_masks","null_cube_shift",
          "best_step","final_training_step","best_val_mi","test_mi","test_product_shuffle_sd","heldout_joint_minus_mismatch_score")


def expected_bio_keys():
    out=set()
    for l in REPR_LAYERS:
        for k in ORDERS:
            for seed in CRITIC_SEEDS:
                out.add(("primary",l,k,seed))
                if l in NULL_LAYERS: out.add(("null_control",l,k,seed))
    return out


def load_bio(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame(columns=BIO_COLS)
    d=pd.read_csv(path); missing=set(BIO_COLS)-set(d.columns)
    if missing: raise RuntimeError(f"Existing Code-11 seed CSV missing {sorted(missing)}")
    if d.duplicated(["family","layer","order_k","critic_seed"]).any(): raise RuntimeError("Duplicate Code-11 critic rows.")
    return d


def save_bio_row(d: pd.DataFrame,row: Mapping,path: Path) -> pd.DataFrame:
    d=pd.concat([d,pd.DataFrame([row])],ignore_index=True).sort_values(["layer","order_k","family","critic_seed"]).reset_index(drop=True)
    write_csv_atomic(d,path); return d


def run_bio(mine04,torch,nn,device,maps,samples,layer_paths,X,chunk,result_path,state_path,fpsha,validation_contract):
    d=load_bio(result_path); expected=expected_bio_keys()
    done={(str(r.family),int(r.layer),int(r.order_k),int(r.critic_seed)) for r in d.itertuples(index=False)}
    if not done.issubset(expected): raise RuntimeError("Existing Code-11 biological CSV has unexpected conditions.")
    for l in REPR_LAYERS:
        layer_needed=any(("primary",l,k,s) not in done or (l in NULL_LAYERS and ("null_control",l,k,s) not in done) for k in ORDERS for s in CRITIC_SEEDS)
        if not layer_needed: continue
        print(f"\nLayer {l:02d}/30",flush=True); layer=np.load(layer_paths[l],mmap_mode="r",allow_pickle=False)
        for k in ORDERS:
            pmiss=[s for s in CRITIC_SEEDS if ("primary",l,k,s) not in done]
            nmiss=[s for s in CRITIC_SEEDS if l in NULL_LAYERS and ("null_control",l,k,s) not in done]
            if not pmiss and not nmiss: continue
            print(f"  k={k}: constructing 2560-D R_l,C,S ...",flush=True)
            U,V,split,counts=build_dataset(layer,maps,samples,k,X,chunk); tr,va,te=split
            for seed in pmiss:
                print(f"    primary seed={seed}",flush=True); res=mine04.train_mine(torch,nn,U,V,int(seed),device,split)
                row={"family":"primary","layer":l,"order_k":k,"critic_seed":int(seed),"representation_dim":2560,"target_dim":1,
                     "n_samples":len(U),"n_train":len(tr),"n_val":len(va),"n_test":len(te),"subset_masks":";".join(map(str,SUBSETS_BY_ORDER[k])),"null_cube_shift":0,**res}
                d=save_bio_row(d,row,result_path); done.add(("primary",l,k,int(seed)))
                write_json_atomic(state_path,{"code_name":CODE_NAME,"fingerprint_sha256":fpsha,"status":"biological_running","estimator_validation":validation_contract,
                                              "completed_biological_fits":len(done),"expected_biological_fits":len(expected),"last_completed":{"family":"primary","layer":l,"order_k":k,"critic_seed":int(seed)}})
            if nmiss:
                Vnull=build_null(samples,k)
                if Vnull.shape!=V.shape or np.array_equal(Vnull,V): raise RuntimeError("Null-target construction failed.")
                for seed in nmiss:
                    print(f"    null seed={seed}",flush=True); res=mine04.train_mine(torch,nn,U,Vnull,int(seed),device,split)
                    row={"family":"null_control","layer":l,"order_k":k,"critic_seed":int(seed),"representation_dim":2560,"target_dim":1,
                         "n_samples":len(U),"n_train":len(tr),"n_val":len(va),"n_test":len(te),"subset_masks":";".join(map(str,SUBSETS_BY_ORDER[k])),"null_cube_shift":1,**res}
                    d=save_bio_row(d,row,result_path); done.add(("null_control",l,k,int(seed)))
                    write_json_atomic(state_path,{"code_name":CODE_NAME,"fingerprint_sha256":fpsha,"status":"biological_running","estimator_validation":validation_contract,
                                                  "completed_biological_fits":len(done),"expected_biological_fits":len(expected),"last_completed":{"family":"null_control","layer":l,"order_k":k,"critic_seed":int(seed)}})
            del U,V
        del layer
    if done!=expected: raise RuntimeError(f"Biological grid incomplete; {len(expected-done)} conditions missing.")
    return d


def audit_bio(d: pd.DataFrame) -> Dict:
    obs={(str(r.family),int(r.layer),int(r.order_k),int(r.critic_seed)) for r in d.itertuples(index=False)}
    if obs!=expected_bio_keys(): raise RuntimeError("Final biological condition grid mismatch.")
    nums=("best_step","final_training_step","best_val_mi","test_mi","test_product_shuffle_sd","heldout_joint_minus_mismatch_score")
    if not np.isfinite(d[list(nums)].to_numpy(float)).all(): raise RuntimeError("Non-finite final biological diagnostics.")
    p=int((d.family=="primary").sum()); n=int((d.family=="null_control").sum())
    if p!=372 or n!=36: raise RuntimeError("Final primary/null row counts mismatch.")
    return {"primary":p,"null_control":n,"total":len(d)}


def fingerprint(est,c08,c09,c10,relval,advval,device_type,chunk):
    return {"schema_version":SCHEMA_VERSION,"code04_manifest_sha256":est["sha04"],"code04_source_sha256":est["sha04src"],"code05_manifest_sha256":est["sha05"],
            "code08_manifest_sha256":c08["sha"],"code09_manifest_sha256":c09["sha"],"code10_manifest_sha256":c10["sha"],
            "code08_map_sha256":c08["map_sha"],"code10_samples_sha256":c10["samples_sha"],
            "relative_validation_manifest_sha256":relval["sha"],"adversarial_validation_manifest_sha256":advval["sha"],
            "calibration_reference_code11_sha256":CALIBRATION_REFERENCE_CODE11_SHA256,
            "device_type":device_type,"chunk_cubes":int(chunk),
            "layers":list(REPR_LAYERS),"orders":list(ORDERS),"representation_dim":REPRESENTATION_DIM,"critic_seeds":list(CRITIC_SEEDS),
            "null_layers":list(NULL_LAYERS),"null_cube_shift":NULL_CUBE_SHIFT,
            "absolute_mi_accuracy_claimed":False,"analysis_contract":"relative_layerwise_order_retention"}


def self_test():
    X=chi_matrix(); assert REPRESENTATION_DIM==2560 and EXPECTED_PRIMARY_FITS==372 and EXPECTED_NULL_FITS==36 and EXPECTED_BIOLOGICAL_FITS==408
    assert CRITIC_SEEDS==(11,23,37)
    assert len(CALIBRATION_REFERENCE_CODE11_SHA256)==64
    rng=np.random.default_rng(7); coeff=rng.normal(size=(3,16,32)).astype(np.float32)
    H=np.einsum("vs,csd->cvd",X.astype(np.float32),coeff,optimize=True)
    for k in ORDERS:
        masks=SUBSETS_BY_ORDER[k]; rec=np.einsum("vm,cvd->cmd",(X[:,masks]/16).astype(np.float32),H,optimize=True)
        np.testing.assert_allclose(rec,coeff[:,masks,:],atol=5e-6,rtol=1e-5)
    t=np.arange(20,dtype=np.float32)[:,None]; sh=cube_shift_null(t,5,4)
    np.testing.assert_array_equal(sh.reshape(5,4)[1],t.reshape(5,4)[0]); assert not np.array_equal(t,sh)
    assert {k:8210*len(SUBSETS_BY_ORDER[k]) for k in ORDERS}=={1:32840,2:49260,3:32840,4:8210}
    print("SELF-TEST: PASS")
    print("Verified: exact Walsh basis/order masks, representation-Walsh round trip, cube-shift null, 372+36 biological fit-grid arithmetic, and frozen relative-MI contract constants.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--preflight-only",action="store_true")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--overwrite",action="store_true")
    ap.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    ap.add_argument("--chunk-cubes",type=int,default=64)
    ap.add_argument("--code04-source",type=Path)
    ap.add_argument("--code04-manifest",type=Path)
    ap.add_argument("--code05-manifest",type=Path)
    ap.add_argument("--code08-manifest",type=Path)
    ap.add_argument("--code09-manifest",type=Path)
    ap.add_argument("--code10-manifest",type=Path)
    ap.add_argument("--relative-validation-manifest",type=Path)
    ap.add_argument("--adversarial-validation-manifest",type=Path)
    ap.add_argument("--output-dir",type=Path)
    a=ap.parse_args()

    if a.self_test:
        self_test(); return
    if a.chunk_cubes<1:
        raise ValueError("--chunk-cubes must be >=1")

    root=project_root()
    exp=root/"04 Experiments"/"EXP02_GB1_ESM2"
    exp03=root/"04 Experiments"/"EXP03_MI"
    out=a.output_dir or exp
    out.mkdir(parents=True,exist_ok=True)

    p04s=a.code04_source or root/"03 Code"/"04_validate_mi_estimator.py"
    p04m=a.code04_manifest or exp03/"mine_validation_manifest.json"
    p05=a.code05_manifest or root/"04 Experiments"/"EXP01_ORBIT_Synth"/"fmib_code05_validation_manifest.json"
    p08=a.code08_manifest or exp/"gb1_esm2_extraction_manifest.json"
    p09=a.code09_manifest or exp/"ESM2_Site_Reps"/"esm2_extraction_manifest.json"
    p10=a.code10_manifest or exp/"gb1_order_targets_manifest.json"
    prel=a.relative_validation_manifest or exp03/"relative_retention_2560d_manifest.json"
    padv=a.adversarial_validation_manifest or exp03/"relative_retention_adversarial_2560d_manifest.json"

    bio=out/"gb1_target_mi_seed_results.csv"
    state=out/"gb1_target_mi_run_state.json"
    final=out/"gb1_target_mi_manifest.json"

    # IMPORTANT: --overwrite intentionally removes only Code-11 biological
    # state/results.  It does NOT delete the historical failed absolute
    # calibration outputs nor either standalone validation suite.
    if a.overwrite:
        for p in (bio,state,final):
            if p.exists():
                p.unlink()

    print("="*76)
    print(f"{CODE_NAME} — GB1 TARGET-ALIGNED ORDER-RESOLVED RELATIVE NEURAL-MI")
    print("="*76)

    X=chi_matrix()
    est=verify_estimator_chain(root,p04s,p04m,p05)
    relval=verify_relative_validation(prel,est)
    advval=verify_adversarial_validation(padv,est)
    c08=verify_code08(p08)
    maps=load_cube_maps(c08["map_path"])
    c09=verify_code09(p09,c08["sha"],True)
    c10=verify_code10(p10,c08["manifest"])
    samples=load_order_samples(c10["samples_path"])

    for s in SPLIT_ORDER:
        if len(samples["cube_ids_by_split"][s])!=len(maps[s]):
            raise RuntimeError(f"{s} Code08/10 cube-row count mismatch.")

    print("\nUpstream provenance/data/hash gate: PASS")
    print("  Code 04 exact estimator + manifest: PASS")
    print("  Code 05 synthetic end-to-end gate: PASS")
    print("  2560-D relative-retention validation: PASS")
    print("  2560-D adversarial negative/flat/positive validation: PASS")
    print("  absolute MI accuracy claimed: NO")
    print("  Code 08/09/10 compatibility: PASS")
    print("  all 31 Code-09 layer hashes/shapes/dtypes: PASS")
    print("  Code-10 cube/subset target ordering: PASS")

    mine04=est["mine04"]
    torch,nn=mine04.import_torch()
    device=mine04.choose_device(torch,a.device)
    if device.type=="cpu":
        torch.set_num_threads(1)

    fp=fingerprint(est,c08,c09,c10,relval,advval,device.type,a.chunk_cubes)
    fpsha=sha256_obj(fp)
    validation_contract={
        "absolute_mi_accuracy_claimed":False,
        "interpretation":"relative neural-MI lower-bound scores for layerwise/order-retention comparisons",
        "calibration_reference_code11_sha256":CALIBRATION_REFERENCE_CODE11_SHA256,
        "relative_validation_manifest_sha256":relval["sha"],
        "relative_validation_status":"PASS",
        "adversarial_validation_manifest_sha256":advval["sha"],
        "adversarial_validation_status":"PASS",
        "validated_directions":["negative","flat","positive"],
        "representation_dim":REPRESENTATION_DIM,
        "critic_seeds":list(CRITIC_SEEDS),
        "no_pca_or_projection":True,
        "no_clamping":True,
        "no_null_subtraction":True,
    }

    print(f"\nContract: device={device}, dim={REPRESENTATION_DIM}, primary fits={EXPECTED_PRIMARY_FITS}, null fits={EXPECTED_NULL_FITS}, fingerprint={fpsha}")
    print("Interpretation: comparative within-order trajectories / relative retention; exact MI in nats is NOT claimed.")

    if a.preflight_only:
        print("PREFLIGHT: PASS\nNo critic was trained.")
        return

    if final.exists() and not a.overwrite:
        mf=load_json(final)
        if mf.get("run_fingerprint_sha256")==fpsha and bool(mf.get("overall_pass",False)):
            print(f"{CODE_NAME}: PASS\nMatching complete run already exists.")
            return
        raise RuntimeError("Existing final Code-11 manifest does not match this revised run.")

    if state.exists() and not a.overwrite:
        st=load_json(state)
        if st.get("fingerprint_sha256")!=fpsha:
            raise RuntimeError(
                "Existing Code-11 state fingerprint mismatch. This is expected if the old "
                "absolute-calibration Code 11 was run previously. Re-run once with --overwrite; "
                "historical calibration files are preserved."
            )
        if not a.resume:
            raise RuntimeError("Existing revised Code-11 state detected. Re-run with --resume.")
    elif not state.exists():
        write_json_atomic(state,{
            "code_name":CODE_NAME,
            "schema_version":SCHEMA_VERSION,
            "fingerprint_sha256":fpsha,
            "status":"initialized",
            "estimator_validation":validation_contract,
            "created_unix_time":time.time(),
        })

    print("\nEstimator validation contract: PASS")
    print("  relative-retention synthetic probe: PASS")
    print("  adversarial negative/flat/positive suite: PASS")
    print("  absolute-MI calibration: NOT used as an acceptance gate")
    print("  absolute-MI accuracy: NOT claimed")

    d=run_bio(
        mine04,torch,nn,device,maps,samples,c09["layer_paths"],X,a.chunk_cubes,
        bio,state,fpsha,validation_contract
    )
    counts=audit_bio(d)

    manifest={
        "schema_version":SCHEMA_VERSION,
        "code_name":CODE_NAME,
        "purpose":"Held-out target-aligned order-resolved GB1 relative neural-MI lower-bound scores across frozen ESM-2 depth.",
        "code11_source_sha256":sha256_file(Path(__file__)),
        "run_fingerprint":fp,
        "run_fingerprint_sha256":fpsha,
        "upstream":{
            "code04_manifest_sha256":est["sha04"],
            "code04_source_sha256":est["sha04src"],
            "code05_manifest_sha256":est["sha05"],
            "relative_validation_manifest_sha256":relval["sha"],
            "adversarial_validation_manifest_sha256":advval["sha"],
            "code08_manifest_sha256":c08["sha"],
            "code08_map_sha256":c08["map_sha"],
            "code09_manifest_sha256":c09["sha"],
            "code10_manifest_sha256":c10["sha"],
            "code10_samples_sha256":c10["samples_sha"],
        },
        "estimator_protocol":{
            "implementation":"exact Code-04 train_mine",
            "critic_seeds":list(CRITIC_SEEDS),
            "representation_dim":REPRESENTATION_DIM,
            "target_dim":1,
            "selection":"validation only",
            "reported_estimate":"held-out test neural DV/MINE lower-bound score",
            "train_standardization":"Code-04 train-only statistics",
            "no_pca_or_projection":True,
            "no_clamping":True,
            "no_null_subtraction":True,
            "absolute_mi_accuracy_claimed":False,
        },
        "estimator_validation":validation_contract,
        "walsh_contract":{
            "positions":list(POSITIONS),
            "sign_convention":"bit 0 / WT -> -1; bit 1 / mutation -> +1",
            "representation_formula":"R_l,C,S = 2^-4 * sum_x H_l(x) chi_S(x)",
            "subset_masks_by_order":{str(k):list(SUBSETS_BY_ORDER[k]) for k in ORDERS},
        },
        "biological_fit_grid":{
            "representation_layers":list(REPR_LAYERS),
            "orders":list(ORDERS),
            "critic_seeds":list(CRITIC_SEEDS),
            "primary_fits":EXPECTED_PRIMARY_FITS,
            "null_layers":list(NULL_LAYERS),
            "null_cube_shift":NULL_CUBE_SHIFT,
            "null_fits":EXPECTED_NULL_FITS,
            "observed_counts":counts,
        },
        "outputs":{"biological_seed_results":bio.name},
        "file_sha256":{bio.name:sha256_file(bio)},
        "overall_pass":True,
        "interpretation_limits":[
            "MINE values are finite-sample variational lower-bound comparative scores, not exact mutual information in nats.",
            "The original 2560-D absolute-MI calibration failed and is not used as an acceptance gate; absolute MI accuracy is not claimed.",
            "Two independent 2560-D synthetic validations passed for relative trajectory/retention fidelity, including negative, flat, and positive order-selective patterns.",
            "Primary interpretation emphasizes within-order layer trajectories and later relative-retention scores before comparing interaction orders.",
            "Very small or negative lower-bound estimates may reflect the estimator floor; Code 11 does not clamp them and they must not be interpreted as negative physical mutual information.",
            "Raw MI-score magnitudes are not used for direct cross-order biological comparison because orders have different sample counts and target distributions.",
            "Null estimates are diagnostics and are not subtracted from primary scores.",
            "The 2560-D relative validation contract does not guarantee detection of arbitrarily weak GB1 dependence.",
            "Context-side I(Q_k;R_l,k) is not part of the validated confirmatory analysis.",
        ],
    }

    write_json_atomic(final,manifest)
    write_json_atomic(state,{
        "code_name":CODE_NAME,
        "schema_version":SCHEMA_VERSION,
        "fingerprint_sha256":fpsha,
        "status":"complete_pass",
        "estimator_validation":validation_contract,
        "completed_biological_fits":counts["total"],
        "final_manifest_sha256":sha256_file(final),
    })

    print(f"\nFinal Code-11 execution audit: PASS\n  primary fits: {counts['primary']}\n  null fits: {counts['null_control']}\n  relative-retention validation: PASS\n  adversarial validation: PASS\n  absolute MI accuracy claimed: NO\n\n{CODE_NAME}: PASS")
    print("Per-seed GB1 neural-MI lower-bound scores are ready for revised Code 12 aggregation.")


if __name__=="__main__":
    main()
