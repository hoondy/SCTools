"""Preprocessing helpers for SCTools."""

from ._shared import (
    _LEGACY_MITOCARTA_PATH,
    _get_autosome_mask,
    _get_protein_coding_mask,
    _matrix_shape,
    _require_pegasus,
    _require_raw,
    _require_scanpy,
    ad,
    clean_unused_categories,
    gc,
    h5py,
    np,
    pd,
    read_elem,
    sparse,
    stats,
)


def scanpy_hvf(
    data,
    flavor="cell_ranger",
    batch_key=None,
    min_mean=0.0125,
    max_mean=3.0,
    min_disp=0.5,
    n_top_genes=None,
    robust_protein_coding=False,
    protein_coding=False,
    autosome=False,
    plot=False,
):
    """Run Scanpy HVG selection on a Pegasus object and store the result."""
    sc = _require_scanpy()

    adata = data.to_anndata()

    if robust_protein_coding:
        if "robust_protein_coding" not in adata.var:
            raise KeyError("Expected `robust_protein_coding` in `.var`.")
        adata = adata[:, adata.var.robust_protein_coding]

    if protein_coding:
        adata = adata[:, _get_protein_coding_mask(adata.var)]

    if autosome:
        adata = adata[:, _get_autosome_mask(adata.var)]

    if flavor == "seurat_v3":
        _require_raw(adata, context="`scanpy_hvf(..., flavor='seurat_v3')`")
        adata.X = adata.raw.X
        if n_top_genes is None:
            raise ValueError("`n_top_genes` is mandatory if `flavor` is `seurat_v3`.")

    hvg = sc.pp.highly_variable_genes(
        adata,
        flavor=flavor,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        batch_key=batch_key,
        n_top_genes=n_top_genes,
        inplace=False,
        subset=False,
    )
    print(hvg.highly_variable.value_counts())
    if plot:
        sc.pl.highly_variable_genes(hvg)

    data.var["highly_variable_features"] = False
    data.var.loc[hvg[hvg.highly_variable].index, "highly_variable_features"] = True

    if protein_coding:
        data.var.loc[~_get_protein_coding_mask(data.var), "highly_variable_features"] = False
    elif set(data.var[data.var.highly_variable_features].index) != set(hvg[hvg.highly_variable].index):
        raise ValueError("`highly_variable_genes` is not the same as `highly_variable_features`.")

    print(data.var.highly_variable_features.value_counts())


def _drop_scanpy_hvf_h5ad_unused_slots(adata, *, keep_raw):
    if keep_raw:
        if adata.raw is not None:
            adata.raw.varm.clear()
    else:
        adata.raw = None

    missing = object()
    log1p = adata.uns.get("log1p", missing)
    adata.uns.clear()
    if log1p is not missing:
        adata.uns["log1p"] = log1p

    for attr in ("obsm", "varm", "layers", "obsp", "varp"):
        getattr(adata, attr).clear()


def _scanpy_hvf_h5ad_var_mask(adata, *, robust_protein_coding, protein_coding, autosome):
    var_mask = None

    if robust_protein_coding:
        print("subset robust_protein_coding")
        if "robust_protein_coding" not in adata.var:
            raise KeyError("Expected `robust_protein_coding` in `.var`.")
        var_mask = adata.var["robust_protein_coding"].astype(bool).to_numpy()

    if protein_coding:
        protein_coding_mask = _get_protein_coding_mask(adata.var)
        var_mask = protein_coding_mask if var_mask is None else var_mask & protein_coding_mask

    if autosome:
        autosome_mask = _get_autosome_mask(adata.var)
        var_mask = autosome_mask if var_mask is None else var_mask & autosome_mask

    return var_mask


def _h5ad_matrix_encoding(node):
    if isinstance(node, h5py.Dataset):
        return "array"
    encoding = node.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    return encoding


def _read_csr_matrix_subset(group, var_idx, chunk_size=500000):
    n_obs, n_vars = _matrix_shape(group)
    if var_idx is None:
        return sparse.csr_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=(n_obs, n_vars),
        )

    chunks = []
    csr_indptr = group["indptr"][:]
    for row_start in range(0, n_obs, chunk_size):
        row_end = min(row_start + chunk_size, n_obs)
        tmp_indptr = csr_indptr[row_start : row_end + 1]
        tmp_csr = sparse.csr_matrix(
            (
                group["data"][tmp_indptr[0] : tmp_indptr[-1]],
                group["indices"][tmp_indptr[0] : tmp_indptr[-1]],
                tmp_indptr - tmp_indptr[0],
            ),
            shape=(row_end - row_start, n_vars),
        )
        chunks.append(tmp_csr[:, var_idx])

    if not chunks:
        return sparse.csr_matrix((0, len(var_idx)))
    return sparse.vstack(chunks, format="csr")


def _read_csc_matrix_subset(group, var_idx):
    n_obs, n_vars = _matrix_shape(group)
    if var_idx is None:
        return sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=(n_obs, n_vars),
        ).tocsr()

    src_indptr = group["indptr"]
    data_parts = []
    indices_parts = []
    new_indptr = [0]
    for col_idx in var_idx:
        start, end = src_indptr[col_idx], src_indptr[col_idx + 1]
        data_parts.append(group["data"][start:end])
        indices_parts.append(group["indices"][start:end])
        new_indptr.append(new_indptr[-1] + end - start)

    data = np.concatenate(data_parts) if data_parts else np.array([], dtype=group["data"].dtype)
    indices = np.concatenate(indices_parts) if indices_parts else np.array([], dtype=group["indices"].dtype)
    return sparse.csc_matrix(
        (data, indices, np.asarray(new_indptr, dtype=group["indptr"].dtype)),
        shape=(n_obs, len(var_idx)),
    ).tocsr()


def _read_h5ad_matrix_subset(node, var_idx):
    encoding = _h5ad_matrix_encoding(node)
    if encoding == "array":
        return node[:, :] if var_idx is None else node[:, var_idx]
    if encoding == "csr_matrix":
        return _read_csr_matrix_subset(node, var_idx)
    if encoding == "csc_matrix":
        return _read_csc_matrix_subset(node, var_idx)
    raise ValueError(f"Unsupported H5AD matrix encoding for HVG selection: {encoding!r}.")


def _raw_var_index_for_hvf(handle, selected_var_names, fallback_var_idx, n_vars):
    raw = handle["raw"]
    raw_x = raw["X"]
    raw_n_vars = raw_x.shape[1] if isinstance(raw_x, h5py.Dataset) else _matrix_shape(raw_x)[1]

    if "var" in raw:
        raw_var = read_elem(raw["var"])
        raw_idx = raw_var.index.get_indexer(selected_var_names)
        if np.all(raw_idx >= 0):
            return raw_idx

    if fallback_var_idx is None:
        return None
    if raw_n_vars == n_vars:
        return fallback_var_idx
    if raw_n_vars == len(selected_var_names):
        return None
    raise ValueError("Could not align selected `.var` genes with `raw/X` for `flavor='seurat_v3'`.")


def _materialize_scanpy_hvf_h5ad(h5ad_file, backed_adata, var_mask, *, use_raw):
    var_idx = None if var_mask is None else np.flatnonzero(var_mask)
    obs = backed_adata.obs.copy()
    var = backed_adata.var.copy() if var_idx is None else backed_adata.var.iloc[var_idx].copy()
    uns = dict(backed_adata.uns)

    with h5py.File(h5ad_file, "r") as handle:
        if use_raw:
            matrix_var_idx = _raw_var_index_for_hvf(handle, var.index, var_idx, backed_adata.n_vars)
            x_node = handle["raw/X"]
        else:
            matrix_var_idx = var_idx
            x_node = handle["X"]
        x = _read_h5ad_matrix_subset(x_node, matrix_var_idx)

    return ad.AnnData(x, obs=obs, var=var, uns=uns)


def scanpy_hvf_h5ad(
    h5ad_file,
    flavor="cell_ranger",
    batch_key=None,
    min_mean=0.0125,
    max_mean=3.0,
    min_disp=0.5,
    n_top_genes=None,
    robust_protein_coding=False,
    protein_coding=False,
    autosome=False,
    plot=False,
):
    """Return highly variable genes from an H5AD using Scanpy."""
    sc = _require_scanpy()

    backed_adata = None
    adata = None
    hvg = None
    try:
        backed_adata = sc.read_h5ad(h5ad_file, backed="r")
        adata = backed_adata
        print(adata)
        _drop_scanpy_hvf_h5ad_unused_slots(adata, keep_raw=flavor == "seurat_v3")

        var_mask = _scanpy_hvf_h5ad_var_mask(
            adata,
            robust_protein_coding=robust_protein_coding,
            protein_coding=protein_coding,
            autosome=autosome,
        )

        if flavor == "seurat_v3":
            _require_raw(adata, context="`scanpy_hvf_h5ad(..., flavor='seurat_v3')`")

        adata = _materialize_scanpy_hvf_h5ad(h5ad_file, adata, var_mask, use_raw=flavor == "seurat_v3")

        if flavor == "seurat_v3":
            if n_top_genes is None:
                raise ValueError("`n_top_genes` is mandatory if `flavor` is `seurat_v3`.")

        print("scanpy hvg")
        hvg = sc.pp.highly_variable_genes(
            adata,
            flavor=flavor,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            batch_key=batch_key,
            n_top_genes=n_top_genes,
            inplace=False,
            subset=False,
        )
        print(hvg.highly_variable.value_counts())
        if plot:
            sc.pl.highly_variable_genes(hvg)

        return adata.var.index[hvg.highly_variable].tolist()
    finally:
        if adata is not None:
            adata_file = getattr(adata, "file", None)
            close = getattr(adata_file, "close", None)
            if close is not None:
                close()
        if backed_adata is not None and backed_adata is not adata:
            backed_file = getattr(backed_adata, "file", None)
            close = getattr(backed_file, "close", None)
            if close is not None:
                close()
        del hvg
        del adata
        del backed_adata
        gc.collect()


def scanpy_pca(data, n_comps=50, use_highly_variable=True):
    """Run Scanpy PCA on a Pegasus object and copy results back."""
    sc = _require_scanpy()

    adata = data.to_anndata()
    adata.var["highly_variable"] = adata.var.highly_variable_features
    sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=use_highly_variable, svd_solver="arpack")

    data.obsm["X_pca"] = adata.obsm["X_pca"]
    data.uns["PCs"] = adata.varm["PCs"]
    data.uns["pca_variance"] = adata.uns["pca"]["variance"]
    data.uns["pca_variance_ratio"] = adata.uns["pca"]["variance_ratio"]


def qc_boundary(counts, k=3):
    """Return lower and upper QC thresholds using a MAD rule on log1p counts."""
    x = np.log1p(counts)
    mad = stats.median_abs_deviation(x)
    return np.exp(np.median(x) - k * mad), np.exp(np.median(x) + k * mad)


def mad_boundary(x, k=3):
    """Return median +/- k*MAD for a numeric vector."""
    mad = stats.median_abs_deviation(x)
    return (np.median(x) - k * mad), (np.median(x) + k * mad)


def calc_sig_scores(data):
    """Compute several built-in Pegasus signature scores."""
    pg = _require_pegasus()
    pg.calc_signature_score(data, "cell_cycle_human")
    pg.calc_signature_score(data, "gender_human")
    pg.calc_signature_score(data, "mitochondrial_genes_human")
    pg.calc_signature_score(data, "ribosomal_genes_human")
    pg.calc_signature_score(data, "apoptosis_human")
    return data.obs


def mark_MitoCarta(data, mitocarta_path=None, gene_column="Symbol"):
    """Mark MitoCarta genes and set them as non-robust."""
    if mitocarta_path is None:
        if _LEGACY_MITOCARTA_PATH.exists():
            mitocarta_path = _LEGACY_MITOCARTA_PATH
        else:
            raise FileNotFoundError(
                "Could not find a MitoCarta reference file. Pass `mitocarta_path=` "
                "to a Human.MitoCarta3.0.csv file."
            )

    mitocarta = pd.read_csv(mitocarta_path)
    if gene_column not in mitocarta.columns:
        raise KeyError(f"Column `{gene_column}` not found in {mitocarta_path}.")
    mito_genes = set(mitocarta[gene_column].dropna().astype(str))
    data.var["mitocarta_genes"] = [x in mito_genes for x in data.var.index]
    data.var.loc[data.var.mitocarta_genes, "robust"] = False
    return data.var[data.var["robust"]]


def calc_zscore_per_cluster(adata, cluster_labels="subclass", variable="n_genes", zscore_name="zscore_subclass_ngenes"):
    """Calculate per-cluster z-scores for one observation-level variable."""
    adata.obs[zscore_name] = 0
    categories = (
        adata.obs[cluster_labels].cat.categories
        if hasattr(adata.obs[cluster_labels], "cat")
        else sorted(adata.obs[cluster_labels].dropna().unique())
    )
    for s in sorted(categories):
        print(s)
        adata.obs.loc[adata.obs[cluster_labels] == s, zscore_name] = stats.zscore(
            np.array(adata[(adata.obs[cluster_labels] == s)].obs[variable])
        )


__all__ = [
    "calc_sig_scores",
    "calc_zscore_per_cluster",
    "clean_unused_categories",
    "mad_boundary",
    "mark_MitoCarta",
    "qc_boundary",
    "scanpy_hvf",
    "scanpy_hvf_h5ad",
    "scanpy_pca",
]
