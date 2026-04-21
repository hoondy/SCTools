"""Analysis and tool helpers for SCTools."""

from ._shared import (
    _get_sample_size,
    _require_numpy_groupies,
    _require_pairwise_metrics,
    _require_pegasus,
    _require_pls_regression,
    _require_raw,
    _require_scanpy,
    ad,
    importlib,
    logger,
    np,
    pd,
    stats,
    time,
)


def info():
    """Print the versions of the main SCTools dependencies."""
    print("pegasus extentions - collection of custom functions for pegasus")
    try:
        print("pegasus v%s" % importlib.metadata.version("pegasuspy"))
    except Exception:
        try:
            print("pegasus v%s" % _require_pegasus().__version__)
        except ImportError:
            print("pegasus unavailable")
    try:
        print("scanpy v%s" % importlib.metadata.version("scanpy"))
    except Exception:
        try:
            print("scanpy v%s" % _require_scanpy().__version__)
        except ImportError:
            print("scanpy unavailable")
    try:
        print("anndata v%s" % importlib.metadata.version("anndata"))
    except Exception:
        print("anndata v%s" % ad.__version__)
    try:
        print("numpy v%s" % importlib.metadata.version("numpy"))
    except Exception:
        print("numpy v%s" % np.__version__)
    try:
        print("pandas v%s" % importlib.metadata.version("pandas"))
    except Exception:
        print("pandas v%s" % pd.__version__)


def agg_by_cluster(h5ad_file, cluster_label):
    """Aggregate scaled mean expression by cluster from an H5AD file."""
    sc = _require_scanpy()

    adata = sc.read_h5ad(h5ad_file)
    _require_raw(adata, context="`agg_by_cluster`")
    adata.X = adata.raw.X

    sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    norm_scaled_counts = sc.pp.scale(adata, copy=True).X

    labels = sorted(list(set(adata.obs[cluster_label])))
    mat = []
    for lab in labels:
        mat.append(np.mean(norm_scaled_counts[adata.obs[cluster_label] == lab, :], axis=0).reshape(1, -1))
    mat = np.concatenate(mat, axis=0)

    mat = pd.DataFrame(mat)
    mat.index = labels
    mat.columns = adata.var.index
    return mat


def pb_agg_by_cluster(h5ad_file, cluster_label, robust_var_label=None, log1p=False, PFlog1pPF=False, mat_key="raw.X"):
    """Build pseudobulk expression profiles by cluster."""
    pg = _require_pegasus()
    sc = _require_scanpy()

    data = pg.read_input(h5ad_file)
    pb = pg.pseudobulk(data, sample=cluster_label, mat_key=mat_key)
    pb.uns["modality"] = "rna"

    if robust_var_label:
        pb.var["robust"] = data.var[robust_var_label]
    else:
        pb.var["robust"] = np.array(np.mean(data.X, axis=0)).flatten() > 0

    pb._inplace_subset_var(pb.var.robust)

    if log1p:
        adata = pb.to_anndata()
        adata.layers["log1pPF"] = sc.pp.log1p(sc.pp.normalize_total(adata, target_sum=None, inplace=False)["X"])
        mat = pd.DataFrame(adata.layers["log1pPF"])
    elif PFlog1pPF:
        adata = pb.to_anndata()
        adata.layers["log1pPF"] = sc.pp.log1p(sc.pp.normalize_total(adata, target_sum=None, inplace=False)["X"])
        adata.layers["PFlog1pPF"] = sc.pp.normalize_total(adata, target_sum=None, layer="log1pPF", inplace=False)["X"]
        mat = pd.DataFrame(adata.layers["PFlog1pPF"])
    else:
        pg.log_norm(pb)
        mat = pd.DataFrame(pb.X)

    mat.columns = pb.var.index
    mat.index = pb.obs.index
    return mat


def cos_similarity(mat1, mat2):
    """Compare two matrices with cosine similarity on shared columns."""
    cosine_similarity, _ = _require_pairwise_metrics()

    common_columns = [x for x in mat1.columns if x in mat2.columns]
    print("Shared gene names:", len(common_columns))

    merged = pd.concat([mat1.loc[:, common_columns], mat2.loc[:, common_columns]])

    cos_dist = pd.DataFrame(cosine_similarity(merged))
    cos_dist.index = merged.index
    cos_dist.columns = merged.index

    return cos_dist.iloc[len(mat1.index) :, : len(mat1.index)]


def l2_distance(mat1, mat2):
    """Compare two matrices with Euclidean distance on shared columns."""
    _, pairwise_distances = _require_pairwise_metrics()

    common_columns = [x for x in mat1.columns if x in mat2.columns]
    print("Shared gene names:", len(common_columns))

    merged = pd.concat([mat1.loc[:, common_columns], mat2.loc[:, common_columns]])

    l2_dist = pd.DataFrame(pairwise_distances(merged))
    l2_dist.index = merged.index
    l2_dist.columns = merged.index

    return l2_dist.iloc[len(mat1.index) :, : len(mat1.index)]


def spearman_corr(mat1, mat2):
    """Compare two matrices with Spearman correlation on shared columns."""
    common_genes = sorted(set(mat1.columns).intersection(set(mat2.columns)))
    print("Shared gene names:", len(common_genes))

    corr = np.zeros((len(mat2.index), len(mat1.index)))

    for m1 in range(len(mat1.index)):
        for m2 in range(len(mat2.index)):
            corr[m2, m1] = stats.spearmanr(mat1[common_genes].iloc[m1], mat2[common_genes].iloc[m2]).correlation
    df = pd.DataFrame(corr)
    df.columns = mat1.index
    df.index = mat2.index
    return df


def pearson_corr(mat1, mat2):
    """Compare two matrices with Pearson correlation on shared columns."""
    common_genes = sorted(set(mat1.columns).intersection(set(mat2.columns)))
    print("Shared gene names:", len(common_genes))

    corr = np.zeros((len(mat2.index), len(mat1.index)))

    for m1 in range(len(mat1.index)):
        for m2 in range(len(mat2.index)):
            corr[m2, m1] = stats.pearsonr(mat1[common_genes].iloc[m1], mat2[common_genes].iloc[m2]).statistic
    df = pd.DataFrame(corr)
    df.columns = mat1.index
    df.index = mat2.index
    return df


def corrMat(data, sample_size=1000, corr_method="spearman"):
    """Calculate a gene-gene correlation matrix from a random cell sample."""
    sample_size = _get_sample_size(data.shape[0], sample_size)
    obs_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)

    exp_df = pd.DataFrame(data.X[obs_indices].todense())
    exp_df.columns = data.var_names
    exp_df = exp_df.loc[:, exp_df.var(axis=0) != 0].copy()

    return exp_df.corr(method=corr_method)


def pseudoMetaCellByGroup(
    adata,
    groupby,
    rep="X_pca_regressed_harmony",
    n_pcs=25,
    n_neighbors=15,
    weighted_dist_thres=0.25,
    k=3,
    metric="cosine",
    postfix="pmc",
):
    """Create pseudo-metacell H5ADs for each group in `adata.obs[groupby]`."""
    sc = _require_scanpy()
    npg = _require_numpy_groupies()
    list_group = list(sorted(set(adata.obs[groupby])))
    for g in list_group:
        print("Processing:", g)

        adata_sub = adata[adata.obs[groupby] == g].copy()
        sc.pp.neighbors(
            adata_sub,
            n_neighbors=n_neighbors,
            use_rep=rep,
            n_pcs=n_pcs,
            knn=True,
            method="umap",
            metric=metric,
            key_added="pmc_neighbors",
        )

        pmc = _getPMC(adata_sub, weighted_dist_thres, k)

        keylist = []
        vallist = []
        for key, val in pmc.items():
            vallist = vallist + val
            keylist = keylist + [key] * len(val)
        print("n PMC barcodes:", len(pmc))
        print("average barcodes per PMC:", np.mean([len(value) for value in pmc.values()]))

        _require_raw(adata_sub, context="`pseudoMetaCellByGroup`")
        groups = [dict(zip(vallist, keylist))[x] for x in adata_sub.obs.index.tolist()]
        pmc_ct = npg.aggregate(groups, np.array(adata_sub.raw.X.todense()), func="sum", axis=0, fill_value=0)

        adata_new = ad.AnnData(pmc_ct, dtype=np.float32)
        adata_new.var = adata_sub.var[["featureid"]]
        adata_new.obs["original_barcodekey"] = [",".join(pmc[int(x)]) for x in adata_new.obs.index.tolist()]
        adata_new.obs["barcodekey"] = [g + "_" + "{:05d}".format(int(x) + 1) for x in adata_new.obs.index.tolist()]
        adata_new.obs.index = adata_new.obs.barcodekey
        del adata_new.obs["barcodekey"]

        adata_new.raw = adata_new
        adata_new.write(g + "_" + postfix + ".h5ad")
        print("Saved", g + "_" + postfix + ".h5ad")


def _getPMC(adata, weighted_dist_thres, k):
    donor_graph = np.array(adata.obsp["pmc_neighbors_connectivities"].todense())
    print("n cell barcodes:", donor_graph.shape[0])

    np.fill_diagonal(donor_graph, 0)

    bc_list = adata.obs.index.to_list()
    bc_pool = bc_list.copy()
    pmc = {}
    i = 0

    while len(bc_pool) > 0:
        bc = bc_pool[0]
        bc_idx = bc_list.index(bc)

        sub_graph = donor_graph[bc_idx, :]
        sub_graph_idx = np.where(sub_graph > weighted_dist_thres)[0]
        sub_graph_idx_argsort = np.array(np.argsort(sub_graph[sub_graph_idx])).flatten()[::-1]
        sort_idx = sub_graph_idx[sub_graph_idx_argsort]

        metacell_final = [bc]
        bc_pool.remove(bc)
        for idx in sort_idx:
            b = bc_list[idx]
            if b in bc_pool:
                metacell_final.append(b)
                bc_pool.remove(b)
            if len(metacell_final) >= k:
                break

        pmc[i] = metacell_final
        i += 1

    return pmc


def diff_markers(marker_dict_res, test, ref):
    """Return markers present in `test` but not in `ref`."""
    test_set = marker_dict_res[test]["up"]
    test_set = set(test_set[test_set.log2Mean_other < 1][:50].index)
    ref_set = marker_dict_res[ref]["up"]
    ref_set = set(ref_set[ref_set.log2Mean_other < 1][:100].index)
    return test_set - ref_set


def pls(
    data: ad.AnnData,
    y: str,
    n_components: int = 50,
    features: str = "highly_variable_features",
    standardize: bool = True,
    max_value: float = 10,
) -> None:
    """Perform PLS regression on the data."""
    pg = _require_pegasus()
    PLSRegression = _require_pls_regression()

    keyword = pg.select_features(data, features)
    start = time.perf_counter()
    X = data.uns[keyword]
    assert y in data.obs

    if standardize:
        m1 = X.mean(axis=0)
        psum = np.multiply(X, X).sum(axis=0)
        std = ((psum - X.shape[0] * (m1 ** 2)) / (X.shape[0] - 1.0)) ** 0.5
        std[std == 0] = 1
        X -= m1
        X /= std

    if max_value is not None:
        X[X > max_value] = max_value
        X[X < -max_value] = -max_value

    pls = PLSRegression(n_components=n_components)
    X_pls = pls.fit_transform(X, data.obs[y].values)[0]

    data.obsm["X_pls"] = X_pls
    data.uns["PLS_x_loadings"] = pls.x_loadings_

    end = time.perf_counter()
    logger.info("PLS is done. Time spent = {:.2f}s.".format(end - start))


__all__ = [
    "agg_by_cluster",
    "corrMat",
    "cos_similarity",
    "diff_markers",
    "info",
    "l2_distance",
    "pb_agg_by_cluster",
    "pearson_corr",
    "pls",
    "pseudoMetaCellByGroup",
    "spearman_corr",
]
