import os
import tempfile
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

    def test_view_writes_results_to_parent(self):
        for backed in (False, True):
            with self.subTest(backed=backed), tempfile.TemporaryDirectory() as tmp:
                adata = self._make_adata()
                adata.obs["keep"] = [True] * 25 + [False] * 5 + [True]
                if backed:
                    path = os.path.join(tmp, "outliers.h5ad")
                    adata.write_h5ad(path)
                    adata = ad.read_h5ad(path, backed="r")
                view = adata[adata.obs["keep"]]

                sct.tl.detect_outliers(view, method="dbscan", dbscan_eps=0.25, dbscan_min_samples=5, min_group_size=2)

                dropped = ~adata.obs["keep"]
                self.assertEqual(int(adata.obs.loc["cell30", "outlier"]), 1)
                self.assertEqual(int(adata.obs["outlier"].sum()), 1)
                self.assertTrue((adata.obs.loc[dropped, "outlier_method"] == "unassigned").all())
                self.assertEqual(adata.uns["outlier_detection"]["n_cells"], view.n_obs)
                self.assertTrue(view.is_view)
                self.assertEqual(int(view.obs.loc["cell30", "outlier"]), 1)
                if backed:
                    adata.file.close()


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for outlier tests")
class NeighborLabelOutlierTests(unittest.TestCase):
    def _make_adata(self):
        # two tight groups, A around (0, 0) and B around (3, 0), and one cell labelled A inside B
        rng = np.random.default_rng(1)
        coords = np.vstack([rng.normal(0, 0.1, size=(40, 2)), rng.normal(0, 0.1, size=(40, 2)) + [3.0, 0.0], [[3.0, 0.02]]])
        obs = pd.DataFrame({"subclass": ["A"] * 40 + ["B"] * 40 + ["A"]}, index=[f"cell{i}" for i in range(81)])
        adata = ad.AnnData(np.zeros((81, 1)), obs=obs)
        adata.obsm["X_pca"] = coords
        return adata

    def _run(self, adata, **kwargs):
        options = {"use_rep": "X_pca", "method": "neighbor_label", "label_n_neighbors": 5, "min_group_size": 2, "backend": "cpu"}
        options.update(kwargs)
        return sct.tl.detect_outliers(adata, **options)

    def test_flags_cell_among_another_label(self):
        adata = self._make_adata()

        summary = self._run(adata)

        self.assertEqual(adata.obs.index[adata.obs["outlier"] == 1].tolist(), ["cell80"])
        self.assertEqual(adata.obs.loc["cell80", "outlier_score"], 1.0)
        self.assertEqual(adata.obs.loc["cell80", "outlier_neighbor_label"], "B")
        self.assertEqual(adata.obs.loc["cell0", "outlier_score"], 0.0)
        self.assertTrue((adata.obs["outlier_method"] == "neighbor_label").all())
        self.assertEqual(set(summary["method"]), {"neighbor_label"})
        self.assertEqual(set(summary["parameter"]), {5})
        self.assertEqual(adata.uns["outlier_detection"]["label_n_neighbors"], 5)

    def test_cut_is_fewer_than(self):
        # cell0's 4 nearest cells are A, B, A, B: half agree, so it is kept at 0.5 and flagged at 0.6
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [3.0, 0.0], [-4.0, 0.0], [100.0, 0.0], [101.0, 0.0]])
        obs = pd.DataFrame({"subclass": ["A", "A", "B", "A", "B", "A", "B"]}, index=[f"cell{i}" for i in range(7)])
        for cut, expected in ((0.5, 0), (0.6, 1)):
            with self.subTest(cut=cut):
                adata = ad.AnnData(np.zeros((7, 1)), obs=obs.copy())
                adata.obsm["X_pca"] = coords
                self._run(adata, label_n_neighbors=4, label_min_agreement=cut, min_group_size=1)
                self.assertEqual(adata.obs.loc["cell0", "outlier_score"], 0.5)
                self.assertEqual(int(adata.obs.loc["cell0", "outlier"]), expected)

    def test_view_uses_only_its_cells_as_neighbours(self):
        for backed in (False, True):
            with self.subTest(backed=backed), tempfile.TemporaryDirectory() as tmp:
                adata = self._make_adata()
                adata.obs["keep"] = [True] * 80 + [False]
                # cell79 (B) moved inside A; cell80 (A inside B) is left out of the view
                adata.obsm["X_pca"][79] = [0.05, 0.05]
                if backed:
                    path = os.path.join(tmp, "outliers.h5ad")
                    adata.write_h5ad(path)
                    adata = ad.read_h5ad(path, backed="r")
                view = adata[adata.obs["keep"]]

                self._run(view)

                self.assertEqual(adata.obs.index[adata.obs["outlier"] == 1].tolist(), ["cell79"])
                self.assertEqual(adata.obs.loc["cell80", "outlier_method"], "unassigned")
                self.assertTrue(pd.isna(adata.obs.loc["cell80", "outlier_neighbor_label"]))
                self.assertEqual(adata.obs.loc["cell79", "outlier_neighbor_label"], "A")
                self.assertTrue(view.is_view)
                self.assertEqual(int(view.obs.loc["cell79", "outlier"]), 1)
                if backed:
                    adata.file.close()

    def test_small_groups_are_neighbours_but_not_scored(self):
        adata = self._make_adata()

        self._run(adata, min_group_size=41)

        # A has 41 cells and is scored; B has 40, so it is skipped but still counts as neighbours of cell80
        self.assertEqual(adata.obs.index[adata.obs["outlier"] == 1].tolist(), ["cell80"])
        self.assertTrue((adata.obs.loc[adata.obs["subclass"] == "B", "outlier_method"] == "skipped_small_group").all())

    def test_other_metric_and_gpu_match_cpu(self):
        reference = self._make_adata()
        self._run(reference)
        runs = {"sklearn": {"metric": "minkowski"}}  # scikit-learn path; p=2 is Euclidean
        try:
            import torch

            if torch.cuda.is_available():
                runs["gpu"] = {"backend": "gpu"}
        except ImportError:
            pass
        for name, kwargs in runs.items():
            with self.subTest(name):
                adata = self._make_adata()
                self._run(adata, **kwargs)
                np.testing.assert_array_equal(adata.obs["outlier"], reference.obs["outlier"])
                np.testing.assert_allclose(adata.obs["outlier_score"], reference.obs["outlier_score"])

    def test_invalid_embedding_rows_are_not_neighbours(self):
        adata = self._make_adata()
        adata.obsm["X_pca"][5] = [np.nan, 0.0]

        self._run(adata)

        self.assertEqual(adata.obs.loc["cell5", "outlier_method"], "invalid_embedding")
        self.assertEqual(sorted(adata.obs.index[adata.obs["outlier"] == 1]), ["cell5", "cell80"])

    def test_requires_groupby(self):
        with self.assertRaises(ValueError):
            sct.tl.detect_outliers(self._make_adata(), groupby=None, use_rep="X_pca", method="neighbor_label")


if __name__ == "__main__":
    unittest.main()
