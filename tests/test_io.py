import tempfile
import unittest
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

import SCTools as sct


def _encoding_type(group):
    encoding = group.attrs["encoding-type"]
    if isinstance(encoding, bytes):
        return encoding.decode()
    return encoding


class IOHelpersTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_adata(self, matrix_format="csr"):
        matrix = np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
                [4.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        )
        if matrix_format == "csc":
            x = sparse.csc_matrix(matrix)
        else:
            x = sparse.csr_matrix(matrix)

        obs = pd.DataFrame({"group": ["a", "b", "a"]}, index=["cell1", "cell2", "cell3"])
        var = pd.DataFrame({"feature_id": ["g1", "g2", "g3"]}, index=["g1", "g2", "g3"])
        adata = ad.AnnData(x, obs=obs, var=var)
        adata.raw = adata.copy()
        return adata

    def test_read_everything_but_x_preserves_annotations(self):
        source = self.tmp / "source.h5ad"
        adata = self._make_adata()
        adata.write_h5ad(source)

        annotations = sct.io.read_everything_but_X(source)

        self.assertEqual(annotations.shape, adata.shape)
        self.assertListEqual(annotations.obs_names.tolist(), adata.obs_names.tolist())
        self.assertListEqual(annotations.var_names.tolist(), adata.var_names.tolist())
        self.assertIsNone(annotations.X)

    def test_csc2csr_on_disk_preserves_annotations(self):
        source = self.tmp / "csc_source.h5ad"
        output = self.tmp / "csr_output.h5ad"
        adata = self._make_adata(matrix_format="csc")
        adata.write_h5ad(source)

        sct.io.csc2csr_on_disk(source, output)

        with h5py.File(output, "r") as handle:
            self.assertEqual(_encoding_type(handle["X"]), "csr_matrix")

        restored = ad.read_h5ad(output)
        self.assertListEqual(restored.obs_names.tolist(), adata.obs_names.tolist())
        self.assertListEqual(restored.var_names.tolist(), adata.var_names.tolist())
        np.testing.assert_array_equal(restored.X.toarray(), adata.X.toarray())

    def test_concat_on_disk_combines_inputs(self):
        first = self.tmp / "first.h5ad"
        second = self.tmp / "second.h5ad"
        output = self.tmp / "concat.h5ad"
        temp = self.tmp / "concat_temp.h5ad"

        adata1 = self._make_adata(matrix_format="csr")[:2].copy()
        adata2 = self._make_adata(matrix_format="csc")[2:].copy()
        adata1.write_h5ad(first)
        adata2.write_h5ad(second)

        sct.io.concat_on_disk([first, second], output, temp_pth=temp)

        merged = ad.read_h5ad(output)
        expected = np.vstack([adata1.X.toarray(), adata2.X.toarray()])
        np.testing.assert_array_equal(merged.X.toarray(), expected)
        self.assertListEqual(merged.obs_names.tolist(), ["cell1", "cell2", "cell3"])
        self.assertFalse(temp.exists())

    def test_write_h5ad_with_new_annotation_copies_raw(self):
        source = self.tmp / "annot_source.h5ad"
        output = self.tmp / "annot_output.h5ad"
        adata = self._make_adata()
        adata.write_h5ad(source)

        annotations = sct.io.read_everything_but_X(source)
        annotations.obs["sample"] = ["s1", "s2", "s3"]

        sct.io.write_h5ad_with_new_annotation(source, annotations, output, raw=True)

        restored = ad.read_h5ad(output)
        self.assertIn("sample", restored.obs.columns)
        self.assertIsNotNone(restored.raw)
        np.testing.assert_array_equal(restored.raw.X.toarray(), adata.raw.X.toarray())

    def test_ondisk_subset_handles_variable_mask_and_raw(self):
        source = self.tmp / "subset_source.h5ad"
        output = self.tmp / "subset_output.h5ad"
        adata = self._make_adata()
        adata.write_h5ad(source)

        subset_obs = np.array([True, False, True])
        subset_var = np.array([True, False, True])

        sct.io.ondisk_subset(source, output, subset_obs=subset_obs, subset_var=subset_var, raw=True)

        restored = ad.read_h5ad(output)
        np.testing.assert_array_equal(
            restored.X.toarray(),
            adata.X.toarray()[subset_obs][:, subset_var],
        )
        np.testing.assert_array_equal(
            restored.raw.X.toarray(),
            adata.raw.X.toarray()[subset_obs][:, subset_var],
        )
        self.assertListEqual(restored.var_names.tolist(), ["g1", "g3"])

    def test_namespace_modules_are_exposed(self):
        self.assertIs(sct.io.read_everything_but_X, sct.read_everything_but_X)
        self.assertTrue(hasattr(sct, "pl"))
        self.assertTrue(hasattr(sct, "pp"))
        self.assertTrue(hasattr(sct, "tl"))


if __name__ == "__main__":
    unittest.main()
