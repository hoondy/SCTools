import gc
import unittest
import weakref

import pandas as pd

import SCTools.pp as pp


class FakeAnnData:
    def __init__(self):
        self.var = pd.DataFrame(index=pd.Index(["gene1", "gene2", "gene3"]))
        self.file = None

    def __repr__(self):
        return "FakeAnnData"


class FakeScanpyPP:
    def highly_variable_genes(self, adata, **kwargs):
        return pd.DataFrame(
            {"highly_variable": [True, False, True]},
            index=adata.var.index,
        )


class FakeScanpyPL:
    def highly_variable_genes(self, hvg):
        return None


class FakeScanpy:
    def __init__(self):
        self.pp = FakeScanpyPP()
        self.pl = FakeScanpyPL()
        self.adata_ref = None

    def read_h5ad(self, h5ad_file):
        adata = FakeAnnData()
        self.adata_ref = weakref.ref(adata)
        return adata


class ScanpyHVFH5ADMemoryTests(unittest.TestCase):
    def setUp(self):
        self.original_require_scanpy = pp._require_scanpy
        self.scanpy = FakeScanpy()
        pp._require_scanpy = lambda: self.scanpy

    def tearDown(self):
        pp._require_scanpy = self.original_require_scanpy

    def test_scanpy_hvf_h5ad_does_not_retain_anndata_after_success(self):
        features = pp.scanpy_hvf_h5ad("input.h5ad")

        self.assertEqual(features, ["gene1", "gene3"])
        gc.collect()
        self.assertIsNone(self.scanpy.adata_ref())

    def test_scanpy_hvf_h5ad_clears_anndata_from_traceback_on_error(self):
        with self.assertRaises(KeyError) as caught:
            pp.scanpy_hvf_h5ad("input.h5ad", robust_protein_coding=True)

        self.assertIsInstance(caught.exception, KeyError)
        gc.collect()
        self.assertIsNone(self.scanpy.adata_ref())


if __name__ == "__main__":
    unittest.main()
