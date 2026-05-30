import gc
import unittest
import weakref

import pandas as pd

import SCTools.pp as pp


class FakeFile:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeRaw:
    def __init__(self):
        self.X = "raw.X"
        self.varm = {"raw_loadings": object()}


class FakeAnnData:
    def __init__(self, owner, *, isbacked):
        self.owner = owner
        self.isbacked = isbacked
        self.is_view = False
        self.X = "X"
        self.raw = FakeRaw()
        self.var = pd.DataFrame(
            {
                "robust_protein_coding": [True, False, True],
                "protein_coding": [True, True, False],
                "gene_chrom": ["1", "2", "X"],
            },
            index=pd.Index(["gene1", "gene2", "gene3"]),
        )
        if not owner.include_robust:
            del self.var["robust_protein_coding"]
        self.uns = {"log1p": {"base": 2}, "large_uns": object()}
        self.obsm = {"X_pca": object()}
        self.varm = {"PCs": object()}
        self.layers = {"counts": object()}
        self.obsp = {"connectivities": object()}
        self.varp = {"correlations": object()}
        self.file = FakeFile() if isbacked else None
        if self.file is not None:
            owner.files.append(self.file)

    def __repr__(self):
        return "FakeAnnData"

    def __getitem__(self, index):
        if self.isbacked and self.is_view:
            raise ValueError("cannot index repeatedly into a backed AnnData")

        _, var_index = index
        adata = FakeAnnData(self.owner, isbacked=self.isbacked)
        adata.is_view = True
        adata.X = self.X
        adata.raw = self.raw
        adata.var = self.var.iloc[var_index].copy()
        adata.uns = dict(self.uns)
        adata.obsm = dict(self.obsm)
        adata.varm = dict(self.varm)
        adata.layers = dict(self.layers)
        adata.obsp = dict(self.obsp)
        adata.varp = dict(self.varp)
        self.owner.slice_calls += 1
        self.owner.adata_refs.append(weakref.ref(adata))
        return adata

    def to_memory(self):
        adata = FakeAnnData(self.owner, isbacked=False)
        adata.X = self.X
        adata.raw = self.raw
        adata.var = self.var.copy()
        adata.uns = dict(self.uns)
        adata.obsm = dict(self.obsm)
        adata.varm = dict(self.varm)
        adata.layers = dict(self.layers)
        adata.obsp = dict(self.obsp)
        adata.varp = dict(self.varp)
        self.owner.adata_refs.append(weakref.ref(adata))
        return adata


class FakeScanpyPP:
    def __init__(self, owner):
        self.owner = owner

    def highly_variable_genes(self, adata, **kwargs):
        highly_variable = [(idx % 2) == 0 for idx in range(len(adata.var.index))]
        self.owner.hvg_received_backed = adata.isbacked
        self.owner.hvg_received_slots = {
            "raw": adata.raw is not None,
            "x": adata.X,
            "uns": set(adata.uns),
            "obsm": set(adata.obsm),
            "varm": set(adata.varm),
            "layers": set(adata.layers),
            "obsp": set(adata.obsp),
            "varp": set(adata.varp),
        }
        return pd.DataFrame(
            {"highly_variable": highly_variable},
            index=adata.var.index,
        )


class FakeScanpyPL:
    def __init__(self, owner):
        self.owner = owner

    def highly_variable_genes(self, hvg):
        self.owner.plot_calls += 1
        return None


class FakeScanpy:
    def __init__(self):
        self.pp = FakeScanpyPP(self)
        self.pl = FakeScanpyPL(self)
        self.adata_refs = []
        self.files = []
        self.hvg_received_backed = None
        self.hvg_received_slots = None
        self.include_robust = True
        self.plot_calls = 0
        self.read_kwargs = None
        self.slice_calls = 0

    def read_h5ad(self, h5ad_file, **kwargs):
        self.read_kwargs = kwargs
        adata = FakeAnnData(self, isbacked=True)
        self.adata_refs.append(weakref.ref(adata))
        return adata


class FakePegasusData:
    def __init__(self, owner):
        self.var = pd.DataFrame(index=pd.Index(["gene1", "gene2", "gene3"]))
        self.adata = FakeAnnData(owner, isbacked=False)
        self.adata.var = self.var.copy()

    def to_anndata(self):
        return self.adata


class ScanpyHVFH5ADMemoryTests(unittest.TestCase):
    def setUp(self):
        self.original_require_scanpy = pp._require_scanpy
        self.scanpy = FakeScanpy()
        pp._require_scanpy = lambda: self.scanpy

    def tearDown(self):
        pp._require_scanpy = self.original_require_scanpy

    def test_scanpy_hvf_does_not_plot_by_default(self):
        data = FakePegasusData(self.scanpy)

        pp.scanpy_hvf(data)

        self.assertEqual(data.var.highly_variable_features.tolist(), [True, False, True])
        self.assertEqual(self.scanpy.plot_calls, 0)

    def test_scanpy_hvf_plots_when_requested(self):
        data = FakePegasusData(self.scanpy)

        pp.scanpy_hvf(data, plot=True)

        self.assertEqual(data.var.highly_variable_features.tolist(), [True, False, True])
        self.assertEqual(self.scanpy.plot_calls, 1)

    def test_scanpy_hvf_h5ad_does_not_retain_anndata_after_success(self):
        features = pp.scanpy_hvf_h5ad("input.h5ad")

        self.assertEqual(features, ["gene1", "gene3"])
        self.assertEqual(self.scanpy.read_kwargs, {"backed": "r"})
        self.assertFalse(self.scanpy.hvg_received_backed)
        self.assertEqual(
            self.scanpy.hvg_received_slots,
            {
                "raw": False,
                "x": "X",
                "uns": {"log1p"},
                "obsm": set(),
                "varm": set(),
                "layers": set(),
                "obsp": set(),
                "varp": set(),
            },
        )
        self.assertEqual(self.scanpy.plot_calls, 0)
        self.assertTrue(all(file.closed for file in self.scanpy.files))
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.scanpy.adata_refs))

    def test_scanpy_hvf_h5ad_combines_filters_before_backed_slice(self):
        features = pp.scanpy_hvf_h5ad(
            "input.h5ad",
            robust_protein_coding=True,
            protein_coding=True,
            autosome=True,
        )

        self.assertEqual(features, ["gene1"])
        self.assertEqual(self.scanpy.slice_calls, 1)

    def test_scanpy_hvf_h5ad_plots_when_requested(self):
        features = pp.scanpy_hvf_h5ad("input.h5ad", plot=True)

        self.assertEqual(features, ["gene1", "gene3"])
        self.assertEqual(self.scanpy.plot_calls, 1)

    def test_scanpy_hvf_h5ad_uses_raw_x_then_drops_unused_raw_for_seurat_v3(self):
        features = pp.scanpy_hvf_h5ad("input.h5ad", flavor="seurat_v3", n_top_genes=2)

        self.assertEqual(features, ["gene1", "gene3"])
        self.assertFalse(self.scanpy.hvg_received_backed)
        self.assertEqual(
            self.scanpy.hvg_received_slots,
            {
                "raw": False,
                "x": "raw.X",
                "uns": {"log1p"},
                "obsm": set(),
                "varm": set(),
                "layers": set(),
                "obsp": set(),
                "varp": set(),
            },
        )
        self.assertEqual(self.scanpy.plot_calls, 0)
        self.assertTrue(all(file.closed for file in self.scanpy.files))

    def test_scanpy_hvf_h5ad_clears_anndata_from_traceback_on_error(self):
        self.scanpy.include_robust = False

        with self.assertRaises(KeyError) as caught:
            pp.scanpy_hvf_h5ad("input.h5ad", robust_protein_coding=True)

        self.assertIsInstance(caught.exception, KeyError)
        self.assertEqual(self.scanpy.read_kwargs, {"backed": "r"})
        self.assertIsNone(self.scanpy.hvg_received_backed)
        self.assertIsNone(self.scanpy.hvg_received_slots)
        self.assertEqual(self.scanpy.plot_calls, 0)
        self.assertTrue(all(file.closed for file in self.scanpy.files))
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.scanpy.adata_refs))


if __name__ == "__main__":
    unittest.main()
