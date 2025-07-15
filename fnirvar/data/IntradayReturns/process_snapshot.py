#!/usr/bin/env python
"""
process_snapshot.py
Estimate factors, loadings, GMM clusters and restricted-VAR(1)
for snapshot folders produced by snapshot_builder.py.

Usage
-----
python process_snapshot.py  snapshots/2010-01-31
"""
from __future__ import annotations
import argparse, json, pathlib, polars as pl, numpy as np

from fnirvar.modeling.train import ER                 
from fnirvar.modeling.train import GR               
from fnirvar.modeling.train import baing             
from fnirvar.modeling.train   import FactorAdjustment   
from fnirvar.modeling.train  import NIRVAR             

# ---------------- helper ----------------------------------------------------
def save_dataframe(mat: np.ndarray, path: pathlib.Path, col_prefix="c"):
    cols = [f"{col_prefix}{i+1}" for i in range(mat.shape[1])]
    pl.from_numpy(mat, schema=cols).write_parquet(path, compression="snappy")

# ---------------- main ------------------------------------------------------
def process_snapshot(snap_dir: pathlib.Path, kmax: int = int(20), lF: int = 5,
                     embed_method="Pearson Correlation",
                     clustering="GMM", gmm_seed: int = 432):
    print(f"[{snap_dir.name}]  start")

    # 1) load X
    X = pl.read_parquet(snap_dir / "X.parquet").to_numpy()

    N = X.shape[1]
    kmax = int(min(kmax, N)) # k cannot be greater than N

    # 2) eigenvalue-ratio test
    # k_hat = GR(X, kmax=kmax)
    # k_hat = int(5)
    
    print(f"N: {N}") 
    k_hat, _, _, _ = baing(X=X,kmax=kmax,jj=2) 


    # 3) factor adjustment
    FA  = FactorAdjustment(X, r=k_hat, lF=lF)
    F   = FA.static_factors().astype(np.float32)   # (T × k̂)
    L   = FA.loadings().astype(np.float32)         # (N × k̂)
    Xi  = FA.get_idiosyncratic_component().astype(np.float32)  # (T × N)
    Factor_Phi = FA.factor_linear_model().astype(np.float32)    # (r × r·lF)
    P_hat = Factor_Phi.reshape(k_hat, lF, k_hat).transpose(1,0,2)   # (lF, r, r)
    save_dataframe(P_hat.reshape(lF, -1),         # flatten for parquet row
                snap_dir / "P_hat.parquet",
                col_prefix="p")

    # 4) NIRVAR clustering & restricted-VAR(1) on Xi
    nir = NIRVAR(Xi, embedding_method=embed_method,
                 clustering_method=clustering,
                 gmm_random_int=gmm_seed)
    d_hat      = nir.d
    K          = nir.K
    sim_mat, labels = nir.gmm() if clustering=="GMM" else nir.gmm_bic()
    ols_phi        = nir.ols_parameters(sim_mat).astype(np.float32)  # (N × N)

    # 5) persist artefacts
    save_dataframe(F , snap_dir / "F.parquet",  col_prefix="f")
    save_dataframe(L , snap_dir / "L.parquet",  col_prefix="f")
    save_dataframe(Xi, snap_dir / "Xi.parquet", col_prefix="x")
    save_dataframe(ols_phi, snap_dir / "Phi.parquet", col_prefix="phi")
    pl.DataFrame({"ticker_id": np.arange(len(labels)),
                  "cluster":   labels.astype(np.int16)}
                ).write_csv(snap_dir / "gmm_labels.csv")

    # update meta-json
    meta_path = snap_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update({"k_hat": int(k_hat),
                 "lF": lF,
                 "d_hat": int(d_hat),
                 "K": int(K),
                 "factor_lags": lF,
                 "embedding": embed_method,
                 "clustering": clustering})
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[{snap_dir.name}]  k̂={k_hat}, d̂={d_hat}, K={K}  ✓")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import concurrent.futures as cf
    import multiprocessing, sys

    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot_dir", help="single snapshot, e.g. snapshots/2010-01-31")
    g.add_argument("--all", action="store_true",
                   help="process *every* sub-folder in snapshots/")
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                   help="parallel workers (default = all cores)")
    args = p.parse_args()

    if args.snapshot_dir:
        process_snapshot(pathlib.Path(args.snapshot_dir))
        sys.exit(0)

    # --all  → discover every folder snapshots/YYYY-MM-DD
    SNAP_ROOT = pathlib.Path("snapshots")
    month_dirs = sorted(d for d in SNAP_ROOT.iterdir() if d.is_dir())

    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(process_snapshot, month_dirs):
            pass            # forces exception propagation / progress
