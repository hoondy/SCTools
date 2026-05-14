import unittest

import anndata as ad
import numpy as np
import pandas as pd

import SCTools as sct

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for outlier tests")
class DetectOutliersTests(unittest.TestCase):
    def _make_adata(self):
        rng = np.random.default_rng(0)
        dense = rng.normal(0, 0.05, size=(30, 2))
        outlier = np.array([[5.0, 5.0]])
        coords = np.vstack([dense, outlier])
        obs = pd.DataFrame(
            {
                "subclass": ["A"] * coords.shape[0],
                "subtype": ["ok"] * dense.shape[0] + ["bad"],
            },
            index=[f"cell{i}" for i in range(coords.shape[0])],
        )
        adata = ad.AnnData(np.zeros((coords.shape[0], 1)), obs=obs)
        adata.obsm["X_umap"] = coords
        return adata

    def test_dbscan_flags_isolated_cell(self):
        adata = self._make_adata()

        summary = sct.tl.detect_outliers(
            adata,
            method="dbscan",
            dbscan_eps=0.25,
            dbscan_min_samples=5,
            min_group_size=2,
        )

        self.assertEqual(int(adata.obs.loc["cell30", "outlier"]), 1)
        self.assertEqual(int(adata.obs["outlier"].sum()), 1)
        self.assertEqual(summary.loc[0, "method"], "dbscan")
        self.assertIn("outlier_method", adata.obs)
        self.assertIn("outlier_detection", adata.uns)

    def test_force_outlier_selector_overrides_model(self):
        adata = self._make_adata()

        sct.tl.detect_outliers(
            adata,
            method="dbscan",
            dbscan_eps=10.0,
            dbscan_min_samples=2,
            min_group_size=2,
            force_outlier={"subtype": "bad"},
        )

        self.assertEqual(int(adata.obs.loc["cell30", "outlier"]), 1)
        self.assertEqual(adata.obs.loc["cell30", "outlier_method"], "forced_outlier")

    def test_small_groups_are_skipped(self):
        adata = self._make_adata()[:3].copy()

        summary = sct.tl.detect_outliers(adata, min_group_size=10)

        self.assertEqual(int(adata.obs["outlier"].sum()), 0)
        self.assertEqual(summary.loc[0, "method"], "skipped")
        self.assertTrue((adata.obs["outlier_method"] == "skipped_small_group").all())


if __name__ == "__main__":
    unittest.main()
