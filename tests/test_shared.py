import sys
import types
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


class IPythonDisplayCompatibilityTests(unittest.TestCase):
    _missing = object()

    def setUp(self):
        self.module_names = [
            "IPython",
            "IPython.display",
            "matplotlib_inline",
            "matplotlib_inline.backend_inline",
        ]
        self.original_modules = {
            name: sys.modules.get(name, self._missing)
            for name in self.module_names
        }

    def tearDown(self):
        for name, module in self.original_modules.items():
            if module is self._missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_patch_ipython_display_uses_matplotlib_inline_backend(self):
        ipython = types.ModuleType("IPython")
        ipython_display = types.ModuleType("IPython.display")
        matplotlib_inline = types.ModuleType("matplotlib_inline")
        backend_inline = types.ModuleType("matplotlib_inline.backend_inline")

        def set_matplotlib_formats(*args, **kwargs):
            return None

        ipython.display = ipython_display
        matplotlib_inline.backend_inline = backend_inline
        backend_inline.set_matplotlib_formats = set_matplotlib_formats
        sys.modules["IPython"] = ipython
        sys.modules["IPython.display"] = ipython_display
        sys.modules["matplotlib_inline"] = matplotlib_inline
        sys.modules["matplotlib_inline.backend_inline"] = backend_inline

        shared._patch_ipython_display_for_scanpy()

        self.assertIs(ipython_display.set_matplotlib_formats, set_matplotlib_formats)


class ScanpyCompatibilityTests(unittest.TestCase):
    _missing = object()

    def setUp(self):
        self.original_require_optional_dependency = shared._require_optional_dependency
        self.original_scanpy_configured = shared._SCANPY_CONFIGURED
        self.original_patch_ipython_display = shared._patch_ipython_display_for_scanpy
        self.module_names = [
            "IPython",
            "IPython.display",
            "matplotlib_inline",
            "matplotlib_inline.backend_inline",
        ]
        self.original_modules = {
            name: sys.modules.get(name, self._missing)
            for name in self.module_names
        }

    def tearDown(self):
        shared._require_optional_dependency = self.original_require_optional_dependency
        shared._SCANPY_CONFIGURED = self.original_scanpy_configured
        shared._patch_ipython_display_for_scanpy = self.original_patch_ipython_display
        for name, module in self.original_modules.items():
            if module is self._missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_require_scanpy_patches_ipython_before_scanpy_setup(self):
        events = []

        class Settings:
            verbosity = None

        class FakeScanpy:
            settings = Settings()

            def set_figure_params(self, **kwargs):
                events.append("set_figure_params")

        fake_scanpy = FakeScanpy()
        shared._SCANPY_CONFIGURED = False
        shared._require_optional_dependency = lambda module_name, install_name=None: fake_scanpy
        shared._patch_ipython_display_for_scanpy = lambda: events.append("patch_ipython")

        self.assertIs(shared._require_scanpy(), fake_scanpy)

        self.assertEqual(events, ["patch_ipython", "set_figure_params"])

    def test_require_scanpy_uses_iterable_ipython_display_format(self):
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
        shared._patch_ipython_display_for_scanpy = lambda: None

        self.assertIs(shared._require_scanpy(), fake_scanpy)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["ipython_format"], "png2x")
        self.assertEqual(fake_scanpy.settings.verbosity, 1)
        self.assertTrue(shared._SCANPY_CONFIGURED)

    def test_require_scanpy_supports_scanpy_ipython_format_unpacking(self):
        ipython = types.ModuleType("IPython")
        ipython_display = types.ModuleType("IPython.display")
        matplotlib_inline = types.ModuleType("matplotlib_inline")
        backend_inline = types.ModuleType("matplotlib_inline.backend_inline")
        format_calls = []

        def set_matplotlib_formats(*args, **kwargs):
            format_calls.append((args, kwargs))

        ipython.display = ipython_display
        matplotlib_inline.backend_inline = backend_inline
        backend_inline.set_matplotlib_formats = set_matplotlib_formats
        sys.modules["IPython"] = ipython
        sys.modules["IPython.display"] = ipython_display
        sys.modules["matplotlib_inline"] = matplotlib_inline
        sys.modules["matplotlib_inline.backend_inline"] = backend_inline

        class Settings:
            verbosity = None

        class FakeScanpy:
            settings = Settings()

            def set_figure_params(self, **kwargs):
                ipython_format = kwargs["ipython_format"]
                if isinstance(ipython_format, str):
                    ipython_format = [ipython_format]
                ipython_display.set_matplotlib_formats(*ipython_format)

        fake_scanpy = FakeScanpy()
        shared._SCANPY_CONFIGURED = False
        shared._require_optional_dependency = lambda module_name, install_name=None: fake_scanpy

        self.assertIs(shared._require_scanpy(), fake_scanpy)

        self.assertEqual(format_calls, [(("png2x",), {})])

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
        shared._patch_ipython_display_for_scanpy = lambda: None

        self.assertIs(shared._require_scanpy(), fake_scanpy)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["ipython_format"], "png2x")
        self.assertNotIn("ipython_format", calls[1])
        self.assertEqual(fake_scanpy.settings.verbosity, 1)
        self.assertTrue(shared._SCANPY_CONFIGURED)


if __name__ == "__main__":
    unittest.main()
