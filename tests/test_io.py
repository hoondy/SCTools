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


def _read_csr_group(group):
    return sparse.csr_matrix(
        (group["data"][:], group["indices"][:], group["indptr"][:]),
        shape=tuple(group.attrs["shape"]),
    )


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

    def _write_pool(self, path, pool):
        obs = pd.DataFrame(
            {
                "SubID_vS": pd.Categorical([f"M{pool}1", f"M{pool}1", f"R{pool}2"]),
                "SubID_cs": [f"M{pool}1", f"H{pool}9", f"R{pool}2"],
                "rep": ["1", "1", "2"],
                "poolID_ref": [f"pool{pool}"] * 3,
                "round_num": ["r1"] * 3,
                "prep": ["p1"] * 3,
                "HTO_n_cs": [1, 1, 2],
                "max_prob": [0.9, 0.8, 0.7],
                "doublet_prob": [0.1, 0.2, 0.3],
                "n_genes": [2, 1, 1],
            },
            index=["AAAC-1", "AAAG-1", "AACT-1"],
        )
        var = pd.DataFrame(index=["g3_index", "g1"])
        x = sparse.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]], dtype=np.float32))
        ad.AnnData(x, obs=obs, var=var).write_h5ad(path)

    def _make_dummy(self):
        return ad.AnnData(
            sparse.csr_matrix((0, 3), dtype=np.float32),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )

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

    def test_csr2csc_on_disk_preserves_annotations(self):
        source = self.tmp / "csr_source.h5ad"
        output = self.tmp / "csc_output.h5ad"
        adata = self._make_adata(matrix_format="csr")
        adata.write_h5ad(source)

        sct.io.csr2csc_on_disk(source, output)

        with h5py.File(output, "r") as handle:
            self.assertEqual(_encoding_type(handle["X"]), "csc_matrix")

        restored = ad.read_h5ad(output)
        self.assertListEqual(restored.obs_names.tolist(), adata.obs_names.tolist())
        self.assertListEqual(restored.var_names.tolist(), adata.var_names.tolist())
        np.testing.assert_array_equal(restored.X.toarray(), adata.X.toarray())

    def test_sparse_conversion_converts_raw_when_requested(self):
        for convert, src_format, dst_encoding in [
            (sct.io.csc2csr_on_disk, "csc", "csr_matrix"),
            (sct.io.csr2csc_on_disk, "csr", "csc_matrix"),
        ]:
            with self.subTest(convert=convert.__name__):
                source = self.tmp / f"{src_format}_raw_source.h5ad"
                output = self.tmp / f"{src_format}_raw_output.h5ad"
                adata = self._make_adata(matrix_format=src_format)
                adata.X = adata.X * 2
                adata.write_h5ad(source)

                convert(source, output, raw=True)

                with h5py.File(output, "r") as handle:
                    self.assertEqual(_encoding_type(handle["X"]), dst_encoding)
                    self.assertEqual(_encoding_type(handle["raw/X"]), dst_encoding)
                restored = ad.read_h5ad(output)
                np.testing.assert_array_equal(restored.X.toarray(), adata.X.toarray())
                np.testing.assert_array_equal(restored.raw.X.toarray(), adata.raw.X.toarray())
                self.assertListEqual(restored.raw.var_names.tolist(), adata.raw.var_names.tolist())

    def test_sparse_conversion_drops_raw_by_default(self):
        for convert, src_format in [(sct.io.csc2csr_on_disk, "csc"), (sct.io.csr2csc_on_disk, "csr")]:
            with self.subTest(convert=convert.__name__):
                source = self.tmp / f"{src_format}_source.h5ad"
                output = self.tmp / f"{src_format}_output.h5ad"
                self._make_adata(matrix_format=src_format).write_h5ad(source)

                convert(source, output)

                self.assertIsNone(ad.read_h5ad(output).raw)

    def test_sparse_conversion_rejects_wrong_input_format(self):
        for convert, wrong_format, expected in [
            (sct.io.csc2csr_on_disk, "csr", "CSC"),
            (sct.io.csr2csc_on_disk, "csc", "CSR"),
        ]:
            with self.subTest(convert=convert.__name__):
                source = self.tmp / f"{wrong_format}_wrong_source.h5ad"
                output = self.tmp / f"{wrong_format}_wrong_output.h5ad"
                self._make_adata(matrix_format=wrong_format).write_h5ad(source)

                with self.assertRaisesRegex(ValueError, expected):
                    convert(source, output, raw=True)
                self.assertFalse(output.exists())

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

    def test_ondisk_subset_handles_raw_x_without_raw_var(self):
        source = self.tmp / "subset_raw_without_var_source.h5ad"
        output = self.tmp / "subset_raw_without_var_output.h5ad"
        adata = self._make_adata()
        adata.write_h5ad(source)

        with h5py.File(source, "a") as handle:
            del handle["raw/var"]
            if "varm" in handle["raw"]:
                del handle["raw/varm"]

        subset_obs = np.array([True, False, True])
        subset_var = np.array([True, False, True])

        sct.io.ondisk_subset(source, output, subset_obs=subset_obs, subset_var=subset_var, raw=True)

        with h5py.File(output, "r") as handle:
            self.assertIn("raw", handle)
            self.assertIn("X", handle["raw"])
            self.assertNotIn("var", handle["raw"])
            raw_x = _read_csr_group(handle["raw/X"])

        np.testing.assert_array_equal(
            raw_x.toarray(),
            adata.raw.X.toarray()[subset_obs][:, subset_var],
        )

    def test_proc_h5ad_harmonizes_obs_and_aligns_to_dummy(self):
        source = self.tmp / "pool1.h5ad"
        self._write_pool(source, 1)
        dummy = self._make_dummy()

        adata = sct.io.proc_h5ad(source, dummy)

        self.assertListEqual(adata.obs_names.tolist(), ["M11-1-AAAC-1", "M11-1-AAAG-1", "R12-2-AACT-1"])
        self.assertListEqual(
            adata.obs.columns.tolist(),
            ["Channel", "SubID", "rep", "poolID", "round_num", "prep", "SubID_cs", "HTO_n_cs", "max_prob", "doublet_prob", "Source"],
        )
        self.assertListEqual(adata.obs["Source"].tolist(), ["M", "M", "R"])
        self.assertListEqual(adata.var_names.tolist(), ["g1", "g2", "g3"])
        np.testing.assert_array_equal(adata.X.toarray(), [[2, 0, 1], [3, 0, 0], [0, 0, 4]])
        self.assertEqual(adata.X.indptr.dtype, np.int64)
        self.assertEqual(adata.X.indices.dtype, np.int64)
        self.assertEqual(dummy.X.indptr.dtype, np.int32)

    def test_proc_h5ad_accepts_custom_columns(self):
        source = self.tmp / "pool1.h5ad"
        self._write_pool(source, 1)

        adata = sct.io.proc_h5ad(source, subject_col="SubID_cs", obs_columns=["SubID_cs", "rep"], source_col=None)

        self.assertListEqual(adata.obs_names.tolist(), ["M11-1-AAAC-1", "H19-1-AAAG-1", "R12-2-AACT-1"])
        self.assertEqual(adata.obs.index.name, "barcodekey")
        self.assertListEqual(adata.obs.columns.tolist(), ["Channel", "SubID_cs", "rep"])
        self.assertListEqual(adata.var_names.tolist(), ["g3", "g1"])

    def test_proc_h5ad_reports_missing_columns(self):
        source = self.tmp / "pool1.h5ad"
        self._write_pool(source, 1)

        with self.assertRaisesRegex(KeyError, "lane"):
            sct.io.proc_h5ad(source, rep_col="lane")

    def test_proc_manifest_prefixes_barcodes_across_pools(self):
        first = self.tmp / "pool1.h5ad"
        second = self.tmp / "pool2.h5ad"
        manifest = self.tmp / "manifest.csv"
        self._write_pool(first, 1)
        self._write_pool(second, 2)
        pd.DataFrame({"Location": [str(first), str(second)]}).to_csv(manifest, index=False)

        parts = sct.io.proc_manifest(manifest, str(self.tmp / "out"), "part", 2, self._make_dummy(), source_col=None)

        merged = ad.read_h5ad(parts[0])
        self.assertEqual(merged.n_obs, 6)
        self.assertTrue(merged.obs_names.is_unique)
        self.assertNotIn("Source", merged.obs.columns)
        self.assertListEqual(merged.var_names.tolist(), ["g1", "g2", "g3"])

    def test_namespace_modules_are_exposed(self):
        self.assertTrue(hasattr(sct, "pl"))
        self.assertTrue(hasattr(sct, "pp"))
        self.assertTrue(hasattr(sct, "tl"))
        self.assertFalse(hasattr(sct, "read_everything_but_X"))
        self.assertFalse(hasattr(sct, "scanpy_hvf"))
        self.assertFalse(hasattr(sct, "corrMat"))
        self.assertFalse(hasattr(sct, "sankey"))


if __name__ == "__main__":
    unittest.main()
