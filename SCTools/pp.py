"""Preprocessing helpers for SCTools."""

from ._shared import (
    _LEGACY_MITOCARTA_PATH,
    _get_autosome_mask,
    _get_protein_coding_mask,
    _require_pegasus,
    _require_raw,
    _require_scanpy,
    clean_unused_categories,
    np,
    pd,
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
    sc.pl.highly_variable_genes(hvg)

    data.var["highly_variable_features"] = False
    data.var.loc[hvg[hvg.highly_variable].index, "highly_variable_features"] = True

    if protein_coding:
        data.var.loc[~_get_protein_coding_mask(data.var), "highly_variable_features"] = False
    elif set(data.var[data.var.highly_variable_features].index) != set(hvg[hvg.highly_variable].index):
        raise ValueError("`highly_variable_genes` is not the same as `highly_variable_features`.")

    print(data.var.highly_variable_features.value_counts())


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
):
    """Return highly variable genes from an H5AD using Scanpy."""
    sc = _require_scanpy()

    adata = sc.read_h5ad(h5ad_file)
    print(adata)

    if robust_protein_coding:
        print("subset robust_protein_coding")
        if "robust_protein_coding" not in adata.var:
            raise KeyError("Expected `robust_protein_coding` in `.var`.")
        adata = adata[:, adata.var.robust_protein_coding]

    if protein_coding:
        adata = adata[:, _get_protein_coding_mask(adata.var)]

    if autosome:
        adata = adata[:, _get_autosome_mask(adata.var)]

    if flavor == "seurat_v3":
        _require_raw(adata, context="`scanpy_hvf_h5ad(..., flavor='seurat_v3')`")
        adata.X = adata.raw.X
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
    sc.pl.highly_variable_genes(hvg)

    return adata.var.index[hvg.highly_variable].tolist()


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
