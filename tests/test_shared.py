import unittest
from contextlib import nullcontext

import SCTools._shared as shared


class SparseDatasetCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.public_sparse_dataset = shared._public_sparse_dataset
        self.legacy_sparse_dataset = shared._legacy_sparse_dataset

    def tearDown(self):
        shared._public_sparse_dataset = self.public_sparse_dataset
        shared._legacy_sparse_dataset = self.legacy_sparse_dataset

    def test_sparse_dataset_prefers_public_factory(self):
        def public_factory(group):
            return ("public", group)

        def legacy_factory(group):
            return ("legacy", group)

        shared._public_sparse_dataset = public_factory
        shared._legacy_sparse_dataset = legacy_factory

        self.assertEqual(shared._as_sparse_dataset("X"), ("public", "X"))

    def test_sparse_dataset_falls_back_to_legacy_factory(self):
        def legacy_factory(group):
            return ("legacy", group)

        shared._public_sparse_dataset = None
        shared._legacy_sparse_dataset = legacy_factory

        self.assertEqual(shared._as_sparse_dataset("X"), ("legacy", "X"))

    def test_sparse_dataset_errors_when_no_factory_exists(self):
        shared._public_sparse_dataset = None
        shared._legacy_sparse_dataset = None

        with self.assertRaises(ImportError):
            shared._as_sparse_dataset("X")


class AnnDataIndexCheckOverrideTests(unittest.TestCase):
    _missing = object()

    def setUp(self):
        self.original_settings = getattr(shared.ad, "settings", self._missing)

    def tearDown(self):
        if self.original_settings is self._missing:
            if hasattr(shared.ad, "settings"):
                delattr(shared.ad, "settings")
        else:
            shared.ad.settings = self.original_settings

    def test_skip_anndata_index_checks_uses_local_override(self):
        calls = []

        class Settings:
            def override(self, **overrides):
                calls.append(overrides)
                return nullcontext()

        shared.ad.settings = Settings()

        with shared._skip_anndata_index_checks():
            pass

        self.assertEqual(calls, [{"check_uniqueness": False}])

    def test_skip_anndata_index_checks_is_noop_without_settings(self):
        if hasattr(shared.ad, "settings"):
            delattr(shared.ad, "settings")

        with shared._skip_anndata_index_checks():
            pass


class ScanpyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.original_require_optional_dependency = shared._require_optional_dependency
        self.original_scanpy_configured = shared._SCANPY_CONFIGURED

    def tearDown(self):
        shared._require_optional_dependency = self.original_require_optional_dependency
        shared._SCANPY_CONFIGURED = self.original_scanpy_configured

    def test_require_scanpy_disables_ipython_display_format_setup(self):
        calls = []

        class Settings:
            verbosity = None

        class FakeScanpy:
            settings = Settings()

            def set_figure_params(self, **kwargs):
                calls.append(kwargs)

        fake_scanpy = FakeScanpy()
        shared._SCANPY_CONFIGURED = False
        shared._require_optional_dependency = lambda module_name, install_name=None: fake_scanpy

        self.assertIs(shared._require_scanpy(), fake_scanpy)

        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["ipython_format"])
        self.assertEqual(fake_scanpy.settings.verbosity, 1)
        self.assertTrue(shared._SCANPY_CONFIGURED)

    def test_require_scanpy_supports_scanpy_without_ipython_format_keyword(self):
        calls = []

        class Settings:
            verbosity = None

        class FakeScanpy:
            settings = Settings()

            def set_figure_params(self, **kwargs):
                calls.append(kwargs)
                if "ipython_format" in kwargs:
                    raise TypeError("unexpected keyword argument 'ipython_format'")

        fake_scanpy = FakeScanpy()
        shared._SCANPY_CONFIGURED = False
        shared._require_optional_dependency = lambda module_name, install_name=None: fake_scanpy

        self.assertIs(shared._require_scanpy(), fake_scanpy)

        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["ipython_format"])
        self.assertNotIn("ipython_format", calls[1])
        self.assertEqual(fake_scanpy.settings.verbosity, 1)
        self.assertTrue(shared._SCANPY_CONFIGURED)


if __name__ == "__main__":
    unittest.main()
