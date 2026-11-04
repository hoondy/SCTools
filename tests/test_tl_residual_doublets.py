import importlib.util
import os
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import SCTools as sct
from SCTools.tl import _cell_ranger_hvg, _knn_numpy, _log_normalize_rows

GROUPS = ["Astro", "EN", "IN", "Oligo", "OPC"]


def _cuda_available():
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def _profiles(rng, n_shared=100, n_markers=20):
    n_genes = n_shared + n_markers * len(GROUPS)
    base = rng.gamma(2.0, 1.0, size=n_genes)
    base[n_shared:] *= 0.05
    profiles = {}
    for i, group in enumerate(GROUPS):
        p = base.copy()
        p[n_shared + i * n_markers : n_shared + (i + 1) * n_markers] *= 400
        profiles[group] = p / p.sum()
    return profiles, n_genes


def _make_adata(n_pure=200, n_hetero=60, n_homo=20, seed=0):
    """Pure cells of five types, heterotypic doublets (two types) and homotypic doublets (one type)."""
    rng = np.random.default_rng(seed)
    profiles, n_genes = _profiles(rng)

    def cell(group):
        return rng.multinomial(int(rng.integers(1500, 2500)), profiles[group])

    rows, labels, kinds = [], [], []
    for group in GROUPS:
        for _ in range(n_pure):
            rows.append(cell(group))
            labels.append(group)
            kinds.append("pure")
    for _ in range(n_hetero):
        a, b = rng.choice(GROUPS, size=2, replace=False)
        rows.append(cell(a) + cell(b))
        labels.append(a)
        kinds.append("heterotypic")
    for _ in range(n_homo):
        a = rng.choice(GROUPS)
        rows.append(cell(a) + cell(a))
        labels.append(a)
        kinds.append("homotypic")
    obs = pd.DataFrame(
        {
            "subclass": pd.Categorical(labels),
            "kind": kinds,
            "libraryID": rng.choice(["lib1", "lib2"], size=len(labels)),
        },
        index=[f"cell{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    return ad.AnnData(sparse.csr_matrix(np.vstack(rows).astype(np.float32)), obs=obs, var=var)


# groupby and library_key are left at their defaults ("subclass" and "libraryID")
PARAMS = dict(
    use_raw=False,
    n_top_genes=150,
    n_pcs=10,
    n_neighbors=20,
    n_real=1000,
    n_simulated=1000,
    hvg_cells_per_group=200,
    pca_cells_per_group=200,
    flag_rate=0.05,
)


class DetectResidualDoubletsTests(unittest.TestCase):
    def test_heterotypic_doublets_are_flagged(self):
        adata = _make_adata()
        summary = sct.tl.detect_residual_doublets(adata, backend="cpu", **PARAMS)
        kind = adata.obs["kind"]
        score = adata.obs["residual_doublet_score"]
        flag = adata.obs["residual_doublet_flag"].astype(bool)

        self.assertGreater(score[kind == "heterotypic"].median(), 0.8)
        # same-type pairs make up ~1/5 of the simulated doublets and sit on the pure clusters
        self.assertLess(score[kind == "pure"].median(), 0.3)
        self.assertGreaterEqual(flag[kind == "heterotypic"].mean(), 0.8)
        self.assertLessEqual(flag[kind == "pure"].mean(), 0.03)
        self.assertTrue(np.all(np.isclose(score * PARAMS["n_neighbors"], np.round(score * PARAMS["n_neighbors"]), atol=1e-4)))

        uns = adata.uns["residual_doublet"]
        self.assertEqual(uns["n_neighbors"], 20)
        self.assertEqual(uns["n_simulated"], 1000)
        self.assertEqual(uns["n_real"], 1000)
        self.assertEqual(uns["loadings"].shape, (len(uns["genes"]), 10))
        self.assertAlmostEqual(uns["threshold"], float(np.quantile(score, 1 - PARAMS["flag_rate"])), places=6)
        self.assertEqual(uns["library_key"], "libraryID")
        self.assertListEqual(list(summary.columns), ["group", "n_cells", "median_score", "n_flagged", "flag_fraction"])
        self.assertEqual(int(summary["n_cells"].sum()), adata.n_obs)

    def test_results_are_reproducible(self):
        first, second = _make_adata(), _make_adata()
        sct.tl.detect_residual_doublets(first, backend="cpu", random_state=3, **PARAMS)
        sct.tl.detect_residual_doublets(second, backend="cpu", random_state=3, **PARAMS)
        np.testing.assert_array_equal(first.obs["residual_doublet_score"], second.obs["residual_doublet_score"])

    @unittest.skipUnless(_cuda_available(), "needs PyTorch with CUDA")
    def test_cpu_and_gpu_backends_agree(self):
        cpu, gpu = _make_adata(), _make_adata()
        sct.tl.detect_residual_doublets(cpu, backend="cpu", **PARAMS)
        sct.tl.detect_residual_doublets(gpu, backend="gpu", **PARAMS)
        a, b = cpu.obs["residual_doublet_score"].to_numpy(), gpu.obs["residual_doublet_score"].to_numpy()
        self.assertGreaterEqual(np.mean(a == b), 0.99)
        self.assertLessEqual(np.abs(a - b).max(), 2.0 / PARAMS["n_neighbors"] + 1e-6)
        self.assertEqual(gpu.uns["residual_doublet"]["backend"], "gpu")

    @unittest.skipIf(_cuda_available(), "a CUDA device is available")
    def test_gpu_backend_needs_cuda(self):
        with self.assertRaises(RuntimeError):
            sct.tl.detect_residual_doublets(_make_adata(n_pure=30, n_hetero=5, n_homo=0), backend="gpu", **PARAMS)

    def test_backed_view_matches_in_memory(self):
        adata = _make_adata()
        keep = np.ones(adata.n_obs, dtype=bool)
        keep[::17] = False
        subset = adata[keep].copy()
        sct.tl.detect_residual_doublets(subset, backend="cpu", **PARAMS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cells.h5ad")
            adata.write_h5ad(path)
            backed = ad.read_h5ad(path, backed="r")
            sct.tl.detect_residual_doublets(backed[keep], backend="cpu", store_pca=True, **PARAMS)
            np.testing.assert_allclose(backed.obs.loc[keep, "residual_doublet_score"], subset.obs["residual_doublet_score"], atol=1e-6)
            self.assertTrue(backed.obs.loc[~keep, "residual_doublet_score"].isna().all())
            self.assertEqual(int(backed.obs.loc[~keep, "residual_doublet_flag"].sum()), 0)
            self.assertEqual(backed.obsm["residual_doublet_pca"].shape, (adata.n_obs, PARAMS["n_pcs"]))
            backed.file.close()

    def test_fit_cells_limit_the_pool_but_every_cell_is_scored(self):
        adata = _make_adata()
        adata.obs["clean"] = (adata.obs["kind"] == "pure").to_numpy()
        sct.tl.detect_residual_doublets(adata, backend="cpu", fit_cells="clean", **PARAMS)
        uns = adata.uns["residual_doublet"]
        self.assertEqual(uns["n_fit_cells"], int(adata.obs["clean"].sum()))
        score = adata.obs["residual_doublet_score"]
        self.assertTrue(score.notna().all())
        self.assertAlmostEqual(uns["threshold"], float(np.quantile(score, 1 - PARAMS["flag_rate"])), places=6)
        flag = adata.obs["residual_doublet_flag"].astype(bool)
        self.assertGreaterEqual(flag[adata.obs["kind"] == "heterotypic"].mean(), 0.8)

    def test_fixed_threshold_overrides_flag_rate(self):
        adata = _make_adata()
        sct.tl.detect_residual_doublets(adata, backend="cpu", threshold=0.5, **PARAMS)
        uns = adata.uns["residual_doublet"]
        self.assertEqual(uns["threshold"], 0.5)
        self.assertTrue(np.isnan(uns["flag_rate"]))
        score = adata.obs["residual_doublet_score"]
        np.testing.assert_array_equal(adata.obs["residual_doublet_flag"], (score >= 0.5).astype(int))

    def test_library_key_defaults_to_libraryID(self):
        adata = _make_adata(n_pure=40, n_hetero=5, n_homo=0)
        del adata.obs["libraryID"]
        with self.assertRaisesRegex(KeyError, "libraryID"):
            sct.tl.detect_residual_doublets(adata, backend="cpu", **PARAMS)
        sct.tl.detect_residual_doublets(adata, backend="cpu", library_key=None, **PARAMS)
        self.assertTrue(adata.obs["residual_doublet_score"].notna().all())

    def test_parameters_are_validated(self):
        adata = _make_adata(n_pure=40, n_hetero=5, n_homo=0)
        with self.assertRaises(ValueError):
            sct.tl.detect_residual_doublets(adata, backend="tpu", **PARAMS)
        with self.assertRaises(ValueError):
            sct.tl.detect_residual_doublets(adata, backend="cpu", **{**PARAMS, "flag_rate": 1.5})
        with self.assertRaises(KeyError):
            sct.tl.detect_residual_doublets(adata, backend="cpu", groupby="missing", **PARAMS)
        with self.assertRaises(ValueError):
            sct.tl.detect_residual_doublets(adata, backend="cpu", **{**PARAMS, "n_top_genes": 5})
        adata.X = adata.X * 0.5
        with self.assertRaises(ValueError):
            sct.tl.detect_residual_doublets(adata, backend="cpu", **PARAMS)

    def test_numpy_knn_is_exact(self):
        rng = np.random.default_rng(1)
        reference = rng.standard_normal((3000, 8)).astype(np.float32)
        queries = rng.standard_normal((500, 8)).astype(np.float32)
        found = _knn_numpy(queries, reference, 7, n_jobs=2, query_block=64, reference_block=700)
        d = ((queries[:, None, :] - reference[None, :, :]) ** 2).sum(-1)
        np.testing.assert_array_equal(found, np.argsort(d, axis=1, kind="stable")[:, :7])

    @unittest.skipUnless(importlib.util.find_spec("scanpy") is not None, "needs Scanpy")
    def test_gene_selection_matches_scanpy_cell_ranger(self):
        import scanpy as sc

        adata = _make_adata(n_pure=100, n_hetero=10, n_homo=0)
        X = adata.X.tocsr()
        logged = _log_normalize_rows(X, np.asarray(X.sum(1)).ravel(), None, 1e4).astype(np.float64)
        sums = np.asarray(logged.sum(0)).ravel()
        sumsq = np.asarray(logged.multiply(logged).sum(0)).ravel()
        mine = _cell_ranger_hvg(sums, sumsq, X.shape[0], 60, np.ones(X.shape[1], dtype=bool))
        ref = ad.AnnData(logged.astype(np.float32))
        sc.pp.highly_variable_genes(ref, flavor="cell_ranger", n_top_genes=60)
        np.testing.assert_array_equal(mine, np.flatnonzero(ref.var["highly_variable"].to_numpy()))


if __name__ == "__main__":
    unittest.main()
