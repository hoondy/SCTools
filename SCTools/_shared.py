"""Shared imports and internal utilities for SCTools."""

import gc
import importlib
import logging
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.experimental import read_elem, write_elem
from scipy import sparse, stats

try:
    from anndata.experimental import sparse_dataset as _public_sparse_dataset
except ImportError:
    _public_sparse_dataset = None

_legacy_sparse_dataset = None
if _public_sparse_dataset is None:
    try:
        from anndata._core.sparse_dataset import SparseDataset as _legacy_sparse_dataset
    except ImportError:
        try:
            from anndata._core.sparse_dataset import BaseCompressedSparseDataset as _legacy_sparse_dataset
        except ImportError:
            _legacy_sparse_dataset = None

logger = logging.getLogger("SCTools")
_SCANPY_CONFIGURED = False
_SEABORN_CONFIGURED = False
_LEGACY_MITOCARTA_PATH = Path("/sc/arion/projects/CommonMind/leed62/ref/MitoCarta/Human.MitoCarta3.0.csv")


def _as_sparse_dataset(group):
    """Return an append-capable sparse dataset wrapper across AnnData versions."""
    if _public_sparse_dataset is not None:
        return _public_sparse_dataset(group)
    if _legacy_sparse_dataset is None:
        raise ImportError(
            "This AnnData version does not expose a compatible sparse dataset API. "
            "Upgrade AnnData or install an older SCTools-compatible AnnData release."
        )
    return _legacy_sparse_dataset(group)


def _require_optional_dependency(module_name, install_name=None):
    """Import an optional dependency and raise a user-friendly error if absent."""
    install_target = install_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"This function requires optional dependency '{install_target}'. "
            f"Install it with `pip install {install_target}` or install SCTools with the `analysis` extra."
        ) from exc


def _require_pegasus():
    return _require_optional_dependency("pegasus", "pegasuspy")


def _require_scanpy():
    sc = _require_optional_dependency("scanpy")
    global _SCANPY_CONFIGURED
    if not _SCANPY_CONFIGURED:
        sc.set_figure_params(
            scanpy=True,
            dpi=100,
            dpi_save=300,
            fontsize=12,
            color_map="YlOrRd",
        )
        sc.settings.verbosity = 1
        _SCANPY_CONFIGURED = True
    return sc


def _require_matplotlib():
    plt = _require_optional_dependency("matplotlib.pyplot", "matplotlib")
    rc_context = getattr(plt, "rc_context")
    return plt, rc_context


def _require_seaborn():
    sns = _require_optional_dependency("seaborn")
    global _SEABORN_CONFIGURED
    if not _SEABORN_CONFIGURED:
        sns.set_style("whitegrid", {"axes.grid": False})
        _SEABORN_CONFIGURED = True
    return sns


def _require_adjust_text():
    return _require_optional_dependency("adjustText").adjust_text


def _require_pairwise_metrics():
    metrics = _require_optional_dependency("sklearn.metrics", "scikit-learn")
    pairwise = _require_optional_dependency("sklearn.metrics.pairwise", "scikit-learn")
    return pairwise.cosine_similarity, metrics.pairwise_distances


def _require_pls_regression():
    return _require_optional_dependency("sklearn.cross_decomposition", "scikit-learn").PLSRegression


def _require_group_kfold():
    return _require_optional_dependency("sklearn.model_selection", "scikit-learn").GroupKFold


def _require_numpy_groupies():
    return _require_optional_dependency("numpy_groupies")


def _require_raw(adata, *, context):
    if adata.raw is None:
        raise ValueError(f"{context} requires `adata.raw` to be populated.")


def _get_sample_size(n_obs, sample_size):
    if sample_size <= 0:
        raise ValueError("`sample_size` must be positive.")
    return min(n_obs, sample_size)


def _get_protein_coding_mask(var):
    if "protein_coding" in var.columns:
        return var["protein_coding"].astype(bool).to_numpy()
    if "gene_type" in var.columns:
        return (var["gene_type"] == "protein_coding").to_numpy()
    raise KeyError("Expected `protein_coding` or `gene_type` in `.var`.")


def _get_autosome_mask(var):
    if "gene_chrom" not in var.columns:
        raise KeyError("Expected `gene_chrom` in `.var` when `autosome=True`.")
    return ~var["gene_chrom"].isin(["MT", "X", "Y"]).to_numpy()


def _require_csr_group(group, *, label):
    encoding = group.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding != "csr_matrix":
        raise ValueError(
            f"{label} must be stored as a CSR sparse matrix. "
            f"Found encoding-type={encoding!r}."
        )


def _matrix_shape(group):
    shape = group.attrs.get("shape")
    if shape is None:
        raise ValueError("Sparse matrix group is missing the `shape` attribute.")
    return tuple(shape)


def _empty_csr(n_vars):
    dummy_x = sparse.csr_matrix((0, n_vars), dtype=np.float32)
    dummy_x.indptr = dummy_x.indptr.astype(np.int64)
    dummy_x.indices = dummy_x.indices.astype(np.int64)
    return dummy_x


def _write_raw_annotations(target, src, subset_var=None):
    if "raw" not in src:
        return None

    raw_var = read_elem(src["raw/var"])
    if subset_var is not None:
        raw_var = raw_var.iloc[np.asarray(subset_var)]
    write_elem(target, "raw/var", raw_var)

    if "varm" in src["raw"]:
        raw_varm = read_elem(src["raw/varm"])
        if subset_var is not None:
            raw_varm = {
                key: np.asarray(value)[np.asarray(subset_var)]
                for key, value in raw_varm.items()
            }
        write_elem(target, "raw/varm", raw_varm)

    return raw_var.shape[0]


def clean_unused_categories(data):
    """Remove unused categorical levels from `.obs` and `.var`."""
    for obs_name in data.obs.columns:
        if data.obs[obs_name].dtype == "category":
            print("Removing unused categories from", obs_name)
            data.obs[obs_name] = data.obs[obs_name].cat.remove_unused_categories()
    for var_name in data.var.columns:
        if data.var[var_name].dtype == "category":
            print("Removing unused categories from", var_name)
            data.var[var_name] = data.var[var_name].cat.remove_unused_categories()
    return data
