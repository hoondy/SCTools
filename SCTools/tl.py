"""Analysis and tool helpers for SCTools."""

from ._shared import (
    _get_sample_size,
    _require_group_kfold,
    _require_numpy_groupies,
    _require_optional_dependency,
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
    sparse,
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


def _require_outlier_estimators():
    neighbors = _require_optional_dependency("sklearn.neighbors", "scikit-learn")
    cluster = _require_optional_dependency("sklearn.cluster", "scikit-learn")
    return neighbors.LocalOutlierFactor, neighbors.NearestNeighbors, cluster.DBSCAN


def _bounded_count(n_obs, explicit, frac, minimum, maximum, *, lower=1, upper=None):
    """Return a model parameter count clipped to the feasible group size."""
    if n_obs < lower:
        return None
    if explicit is None:
        value = int(n_obs * frac)
        value = max(min(value, maximum), minimum)
    else:
        value = int(explicit)
    max_allowed = n_obs if upper is None else min(n_obs, upper)
    return max(lower, min(value, max_allowed))


def _iter_obs_groups(obs, groupby):
    """Yield positional indices for each requested observation group."""
    if groupby is None:
        yield "all", np.arange(obs.shape[0])
        return

    columns = [groupby] if isinstance(groupby, str) else list(groupby)
    missing = [column for column in columns if column not in obs.columns]
    if missing:
        raise KeyError(f"Missing groupby column(s) in `.obs`: {missing}.")

    grouping_frame = obs.loc[:, columns].reset_index(drop=True)
    indices = grouping_frame.groupby(columns, observed=True, dropna=False, sort=True).indices
    for key, positions in indices.items():
        if not isinstance(key, tuple):
            key = (key,)
        label = "|".join(str(value) for value in key)
        yield label, np.asarray(positions, dtype=int)


def _as_obs_mask(adata, selector, *, name):
    """Coerce a selector into a boolean mask over adata.obs."""
    mask = np.zeros(adata.n_obs, dtype=bool)
    if selector is None:
        return mask

    if callable(selector):
        values = selector(adata.obs)
    elif isinstance(selector, dict):
        for column, allowed in selector.items():
            if column not in adata.obs.columns:
                raise KeyError(f"`{name}` references missing `.obs` column {column!r}.")
            if isinstance(allowed, (str, bytes)):
                allowed_values = [allowed]
            else:
                try:
                    allowed_values = list(allowed)
                except TypeError:
                    allowed_values = [allowed]
            mask |= adata.obs[column].isin(allowed_values).to_numpy()
        return mask
    else:
        values = selector

    if isinstance(values, pd.Series):
        if values.index.equals(adata.obs.index):
            arr = values.to_numpy()
        elif len(values) == adata.n_obs:
            arr = values.to_numpy()
        else:
            arr = values.reindex(adata.obs.index).fillna(False).to_numpy()
    else:
        arr = np.asarray(values)

    if arr.shape[0] != adata.n_obs:
        raise ValueError(f"`{name}` must have length {adata.n_obs}; found {arr.shape[0]}.")
    return arr.astype(bool)


def detect_outliers(
    adata: ad.AnnData,
    groupby: str = "subclass",
    use_rep: str = "X_umap",
    outlier_key: str = "outlier",
    method: str = "auto",
    large_group_threshold: int = 300_000,
    min_group_size: int = 100,
    dbscan_eps: float = 1.0,
    dbscan_radius: float = None,
    dbscan_min_samples: int = None,
    dbscan_min_samples_frac: float = 0.02,
    dbscan_min_samples_min: int = 100,
    dbscan_min_samples_max: int = 2_000,
    lof_n_neighbors: int = None,
    lof_n_neighbors_frac: float = 0.04,
    lof_n_neighbors_min: int = 100,
    lof_n_neighbors_max: int = 4_000,
    lof_contamination="auto",
    metric: str = "euclidean",
    n_jobs: int = -1,
    force_outlier=None,
    force_inlier=None,
    invalid_policy: str = "outlier",
    store_details: bool = True,
    diagnostics_key: str = "outlier_detection",
) -> pd.DataFrame:
    """Detect extreme embedding outliers and write ``adata.obs[outlier_key]``.

    The default behavior mirrors the hybrid workflow used in the PsychAD
    notebook: cells are processed within annotation groups, DBSCAN is used for
    ordinary-size groups, and LocalOutlierFactor is used for groups above
    ``large_group_threshold`` to avoid building a very large radius graph.

    ``adata.obs[outlier_key]`` is written as an integer 0/1 label where 1 means
    outlier. A per-group summary DataFrame is returned and, by default, stored
    with parameters in ``adata.uns[diagnostics_key]``.

    Parameters
    ----------
    adata
        AnnData object containing the embedding in ``adata.obsm[use_rep]``.
    groupby
        Observation column used to fit separate outlier models. Set to ``None``
        to fit one model across all cells.
    use_rep
        Key in ``adata.obsm`` to use as the outlier-detection embedding.
    method
        ``"auto"``, ``"dbscan"``, or ``"lof"``. ``"auto"`` switches to LOF only
        for groups larger than ``large_group_threshold``.
    force_outlier, force_inlier
        Optional manual overrides. Pass a boolean mask, callable returning a
        mask, or a dict such as
        ``{"subclass": ["EN_NF"], "subtype": ["Immune_PVM"]}``.
    invalid_policy
        How to label cells with non-finite embedding coordinates: ``"outlier"``
        or ``"inlier"``.
    store_details
        If True, also writes method, score, and DBSCAN cluster labels to
        ``adata.obs`` and stores diagnostics in ``adata.uns``.
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"`{use_rep}` not found in `adata.obsm`.")
    if method not in {"auto", "dbscan", "lof"}:
        raise ValueError("`method` must be one of 'auto', 'dbscan', or 'lof'.")
    if invalid_policy not in {"outlier", "inlier"}:
        raise ValueError("`invalid_policy` must be 'outlier' or 'inlier'.")
    if dbscan_eps <= 0:
        raise ValueError("`dbscan_eps` must be positive.")
    if dbscan_radius is None:
        dbscan_radius = dbscan_eps
    if dbscan_radius < dbscan_eps:
        raise ValueError("`dbscan_radius` must be greater than or equal to `dbscan_eps`.")
    if min_group_size < 1:
        raise ValueError("`min_group_size` must be positive.")

    X = np.asarray(adata.obsm[use_rep])
    if X.ndim != 2:
        raise ValueError(f"`adata.obsm[{use_rep!r}]` must be a 2D array.")
    if X.shape[0] != adata.n_obs:
        raise ValueError(
            f"`adata.obsm[{use_rep!r}]` has {X.shape[0]} rows but adata has {adata.n_obs} observations."
        )

    LocalOutlierFactor, NearestNeighbors, DBSCAN = _require_outlier_estimators()

    outlier = np.zeros(adata.n_obs, dtype=bool)
    method_used = np.full(adata.n_obs, "unassigned", dtype=object)
    score = np.full(adata.n_obs, np.nan, dtype=np.float64)
    dbscan_label = np.full(adata.n_obs, pd.NA, dtype=object)
    rows = []

    for group_name, positions in _iter_obs_groups(adata.obs, groupby):
        group_X = X[positions]
        finite = np.isfinite(group_X).all(axis=1)
        finite_positions = positions[finite]
        invalid_positions = positions[~finite]

        if invalid_positions.size:
            outlier[invalid_positions] = invalid_policy == "outlier"
            method_used[invalid_positions] = "invalid_embedding"

        n_cells = positions.size
        n_scored = finite_positions.size
        row = {
            "group": group_name,
            "n_cells": int(n_cells),
            "n_scored": int(n_scored),
            "n_invalid": int(invalid_positions.size),
            "method": "skipped",
            "parameter": pd.NA,
            "n_outliers": int(outlier[invalid_positions].sum()),
        }

        if n_scored < min_group_size:
            method_used[finite_positions] = "skipped_small_group"
            rows.append(row)
            continue

        chosen = method
        if chosen == "auto":
            chosen = "lof" if n_scored > large_group_threshold else "dbscan"

        scored_X = X[finite_positions]
        if chosen == "lof":
            n_neighbors = _bounded_count(
                n_scored,
                lof_n_neighbors,
                lof_n_neighbors_frac,
                lof_n_neighbors_min,
                lof_n_neighbors_max,
                lower=2,
                upper=n_scored - 1,
            )
            if n_neighbors is None:
                method_used[finite_positions] = "skipped_small_group"
                rows.append(row)
                continue

            clf = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=lof_contamination,
                metric=metric,
                n_jobs=n_jobs,
            )
            labels = clf.fit_predict(scored_X)
            group_outlier = labels == -1
            outlier[finite_positions] = group_outlier
            method_used[finite_positions] = "lof"
            score[finite_positions] = -clf.negative_outlier_factor_
            row.update(
                {
                    "method": "lof",
                    "parameter": int(n_neighbors),
                    "n_outliers": int(group_outlier.sum() + outlier[invalid_positions].sum()),
                }
            )

        elif chosen == "dbscan":
            min_samples = _bounded_count(
                n_scored,
                dbscan_min_samples,
                dbscan_min_samples_frac,
                dbscan_min_samples_min,
                dbscan_min_samples_max,
                lower=2,
            )
            if min_samples is None:
                method_used[finite_positions] = "skipped_small_group"
                rows.append(row)
                continue

            nn = NearestNeighbors(radius=dbscan_radius, metric=metric, n_jobs=n_jobs)
            nn.fit(scored_X)
            graph = nn.radius_neighbors_graph(scored_X, mode="distance", sort_results=True)
            clf = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric="precomputed", n_jobs=n_jobs)
            labels = clf.fit_predict(graph)
            group_outlier = labels == -1
            outlier[finite_positions] = group_outlier
            method_used[finite_positions] = "dbscan"
            score[finite_positions] = min_samples - graph.getnnz(axis=1)
            dbscan_label[finite_positions] = labels.astype(int)
            row.update(
                {
                    "method": "dbscan",
                    "parameter": int(min_samples),
                    "n_outliers": int(group_outlier.sum() + outlier[invalid_positions].sum()),
                }
            )
        else:
            raise ValueError(f"Unexpected outlier method {chosen!r}.")

        rows.append(row)

    force_inlier_mask = _as_obs_mask(adata, force_inlier, name="force_inlier")
    force_outlier_mask = _as_obs_mask(adata, force_outlier, name="force_outlier")
    if force_inlier_mask.any():
        outlier[force_inlier_mask] = False
        method_used[force_inlier_mask] = "forced_inlier"
    if force_outlier_mask.any():
        outlier[force_outlier_mask] = True
        method_used[force_outlier_mask] = "forced_outlier"

    adata.obs[outlier_key] = outlier.astype(int)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.rename(columns={"n_outliers": "n_model_outliers"})
        final_counts = [
            int(outlier[positions].sum())
            for _, positions in _iter_obs_groups(adata.obs, groupby)
        ]
        summary["n_outliers"] = final_counts
        summary["model_outlier_fraction"] = summary["n_model_outliers"] / summary["n_cells"]
        summary["outlier_fraction"] = summary["n_outliers"] / summary["n_cells"]

    if store_details:
        adata.obs[f"{outlier_key}_method"] = pd.Categorical(method_used)
        adata.obs[f"{outlier_key}_score"] = score
        if np.any(pd.notna(dbscan_label)):
            adata.obs[f"{outlier_key}_dbscan_label"] = pd.array(dbscan_label, dtype="Int64")

        adata.uns[diagnostics_key] = {
            "use_rep": use_rep,
            "groupby": groupby,
            "method": method,
            "large_group_threshold": int(large_group_threshold),
            "min_group_size": int(min_group_size),
            "dbscan_eps": float(dbscan_eps),
            "dbscan_radius": float(dbscan_radius),
            "dbscan_min_samples_frac": float(dbscan_min_samples_frac),
            "dbscan_min_samples_min": int(dbscan_min_samples_min),
            "dbscan_min_samples_max": int(dbscan_min_samples_max),
            "lof_n_neighbors_frac": float(lof_n_neighbors_frac),
            "lof_n_neighbors_min": int(lof_n_neighbors_min),
            "lof_n_neighbors_max": int(lof_n_neighbors_max),
            "lof_contamination": lof_contamination,
            "metric": metric,
            "n_cells": int(adata.n_obs),
            "n_outliers": int(outlier.sum()),
            "outlier_fraction": float(outlier.mean()) if adata.n_obs else 0.0,
            "n_forced_outlier": int(force_outlier_mask.sum()),
            "n_forced_inlier": int(force_inlier_mask.sum()),
            "group_summary": summary,
        }

    return summary


def pls(
    data: ad.AnnData,
    y: str,
    n_components: int = 50,
    features: str = "highly_variable_features",
    standardize: bool = True,
    max_value: float = 10,
) -> None:
    """Perform PLS regression on the data.

    Stores the cell embedding in ``data.obsm["X_pls"]`` and the gene-space
    loadings / weights in ``data.uns["PLS_x_loadings"]`` /
    ``data.uns["PLS_x_weights"]``. The input HVG matrix in
    ``data.uns[<features>]`` is not modified.
    """
    pg = _require_pegasus()
    PLSRegression = _require_pls_regression()

    if y not in data.obs.columns:
        raise KeyError(f"`y`={y!r} not in data.obs.")
    y_vec = pd.to_numeric(data.obs[y], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y_vec).all():
        raise ValueError(
            f"`data.obs[{y!r}]` must be numeric and finite; "
            "drop or impute NaN / non-numeric values before calling `pls`."
        )

    keyword = pg.select_features(data, features)
    start = time.perf_counter()

    X = data.uns[keyword]
    if sparse.issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=np.float64, copy=True)  # never mutate data.uns[keyword]

    if standardize:
        m1 = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        X -= m1
        X /= std

    if max_value is not None:
        np.clip(X, -max_value, max_value, out=X)

    model = PLSRegression(n_components=n_components, scale=False)
    X_pls = model.fit_transform(X, y_vec)[0]

    data.obsm["X_pls"] = X_pls
    data.uns["PLS_x_loadings"] = model.x_loadings_
    data.uns["PLS_x_weights"] = model.x_weights_

    end = time.perf_counter()
    logger.info("PLS is done. Time spent = {:.2f}s.".format(end - start))


def _apply_standardize(X, mean, std, max_value):
    """Apply precomputed column mean/std (in place on a fresh copy) and optional clip."""
    X = np.array(X, dtype=np.float64, copy=True)
    X -= mean
    X /= std
    if max_value is not None:
        np.clip(X, -max_value, max_value, out=X)
    return X


def _fit_standardize(X):
    """Compute column mean and std (ddof=1) from X; guard zero-variance columns."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return mean, std


def _residualize(X, C):
    """Return X residualized on covariate matrix C (with intercept). Fits OLS column-wise."""
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    C = np.hstack([np.ones((C.shape[0], 1)), C])
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta, beta


def _build_covariates(obs, covariates):
    """Turn a list of .obs column names into a numeric design matrix (one-hot for categoricals)."""
    if covariates is None or len(covariates) == 0:
        return None
    frame = obs.loc[:, list(covariates)].copy()
    return pd.get_dummies(frame, drop_first=True, dummy_na=False).to_numpy(dtype=np.float64)


def _oof_pls(
    X_valid,
    y_valid,
    donors_valid,
    C_valid,
    n_components,
    n_splits,
    standardize,
    max_value,
    rng,
):
    """Donor-grouped out-of-fold PLS scores.

    Standardization (if enabled) and covariate residualization are both refit on
    each training fold, so no information from the held-out donors leaks into the
    scaling parameters or the covariate betas used to score them.
    """
    PLSRegression = _require_pls_regression()
    GroupKFold = _require_group_kfold()

    n = X_valid.shape[0]
    oof = np.full((n, n_components), np.nan, dtype=np.float64)
    order = rng.permutation(n)
    inv_order = np.argsort(order)
    X_ord = X_valid[order]
    y_ord = y_valid[order]
    g_ord = donors_valid[order]
    C_ord = C_valid[order] if C_valid is not None else None

    for tr, te in GroupKFold(n_splits=n_splits).split(X_ord, y_ord, groups=g_ord):
        X_tr = np.array(X_ord[tr], dtype=np.float64, copy=True)
        X_te = np.array(X_ord[te], dtype=np.float64, copy=True)
        if standardize:
            mean_tr, std_tr = _fit_standardize(X_tr)
            X_tr -= mean_tr
            X_tr /= std_tr
            X_te -= mean_tr
            X_te /= std_tr
        if max_value is not None:
            np.clip(X_tr, -max_value, max_value, out=X_tr)
            np.clip(X_te, -max_value, max_value, out=X_te)
        if C_ord is not None:
            X_tr, beta = _residualize(X_tr, C_ord[tr])
            C_te_aug = np.hstack([np.ones((te.size, 1)), C_ord[te]])
            X_te = X_te - C_te_aug @ beta
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(X_tr, y_ord[tr])
        oof[te] = model.transform(X_te)

    return oof[inv_order]


def pls_score(
    data: ad.AnnData,
    y: str,
    donor_key: str,
    features: str = "highly_variable_features",
    n_components: int = 2,
    n_splits: int = 5,
    standardize: bool = True,
    max_value: float = 10.0,
    covariates=None,
    score_key: str = "pls_disease_score",
    oof_components_key: str = "X_pls_oof",
    loadings_key: str = "PLS_x_loadings_full",
    weights_key: str = "PLS_x_weights_full",
    n_permutations: int = 0,
    random_state: int = 0,
) -> dict:
    """Per-cell disease pseudotime via donor-grouped PLS with out-of-fold scoring.

    Intended to be run on an already-subsetted AnnData (e.g. one cell type). `y` is
    a donor-level quantitative phenotype (Braak, CERAD, PRS, continuous biomarker).
    Returns a dict of diagnostics; writes:
      - data.obs[score_key]              cell score in [0, 1] (rank of PLS1, NaN if excluded)
      - data.obsm[oof_components_key]    (n_cells, n_components) out-of-fold components
      - data.uns[loadings_key]           gene loadings from the full-data fit
      - data.uns[weights_key]            x_weights_ from the full-data fit (for projection)
    """
    PLSRegression = _require_pls_regression()
    _require_group_kfold()  # fail fast if sklearn is missing
    pg = _require_pegasus()

    if y not in data.obs.columns:
        raise KeyError(f"`y`={y!r} not in data.obs.")
    if donor_key not in data.obs.columns:
        raise KeyError(f"`donor_key`={donor_key!r} not in data.obs.")

    keyword = pg.select_features(data, features)
    X_full = data.uns[keyword]
    if sparse.issparse(X_full):
        X_full = X_full.toarray()
    X_full = np.array(X_full, dtype=np.float64, copy=True)  # hard copy; never mutate .uns

    y_vec = pd.to_numeric(data.obs[y], errors="coerce").to_numpy()
    donor_missing = pd.isna(data.obs[donor_key]).to_numpy()
    donors = data.obs[donor_key].astype(str).to_numpy()
    C_full = _build_covariates(data.obs, covariates)

    valid = np.isfinite(y_vec) & ~donor_missing
    if valid.sum() < 2:
        raise ValueError("Not enough cells with finite `y` and valid `donor_key`.")

    unique_donors = pd.unique(donors[valid])
    if len(unique_donors) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} but only {len(unique_donors)} donors with valid `y`. "
            "Lower `n_splits` or check inputs."
        )
    # Warn if y has no within-donor variance (the usual case for Braak etc.)
    dvals = pd.DataFrame({"d": donors[valid], "y": y_vec[valid]}).groupby("d")["y"].nunique()
    if (dvals > 1).any():
        logger.info("`y` varies within donor for some donors; using cell-level values as given.")

    start = time.perf_counter()

    X_valid = X_full[valid]
    y_valid = y_vec[valid].astype(np.float64)
    donors_valid = donors[valid]
    C_valid = C_full[valid] if C_full is not None else None

    rng = np.random.default_rng(random_state)

    # ---- Donor-grouped out-of-fold PLS (per-fold standardization + residualization) ----
    oof = _oof_pls(
        X_valid, y_valid, donors_valid, C_valid,
        n_components, n_splits, standardize, max_value, rng,
    )

    # ---- Full-data fit for exportable loadings/weights ----
    # Standardization here uses global stats (one-shot). Stored alongside the
    # loadings/weights so callers can reproduce the transform on new data.
    if standardize:
        mean_full, std_full = _fit_standardize(X_valid)
        X_fit = _apply_standardize(X_valid, mean_full, std_full, max_value)
    else:
        mean_full = np.zeros(X_valid.shape[1], dtype=np.float64)
        std_full = np.ones(X_valid.shape[1], dtype=np.float64)
        X_fit = np.array(X_valid, dtype=np.float64, copy=True)
        if max_value is not None:
            np.clip(X_fit, -max_value, max_value, out=X_fit)
    if C_valid is not None:
        X_fit, _ = _residualize(X_fit, C_valid)
    full_model = PLSRegression(n_components=n_components, scale=False)
    full_model.fit(X_fit, y_valid)
    full_x_loadings = full_model.x_loadings_.copy()
    full_x_weights = full_model.x_weights_.copy()

    # ---- Orient PLS1 so higher score = higher y, consistently across oof + weights ----
    y_has_variance = np.std(y_valid, ddof=1) > 0
    if y_has_variance:
        in_sample = full_model.transform(X_fit)[:, 0]
        r_full, _ = stats.pearsonr(in_sample, y_valid)
        sign = -1.0 if (np.isfinite(r_full) and r_full < 0) else 1.0
        oof[:, 0] *= sign
        full_x_loadings[:, 0] *= sign
        full_x_weights[:, 0] *= sign
        r1_raw, _ = stats.pearsonr(oof[:, 0], y_valid)
        r1 = float(r1_raw) if np.isfinite(r1_raw) else float("nan")
    else:
        logger.warning("pls_score: `y` has no variance; skipping orientation and permutation test.")
        r1 = float("nan")

    # ---- Sanity diagnostics ----
    df_score = pd.DataFrame({"d": donors_valid, "s": oof[:, 0]})
    between = df_score.groupby("d")["s"].mean().var(ddof=1)
    within = df_score.groupby("d")["s"].var(ddof=1).mean()
    total = df_score["s"].var(ddof=1)
    between_frac = float(between / total) if total and total > 0 else float("nan")
    within_mean = float(within) if np.isfinite(within) else float("nan")

    # ---- Permutation null, using the same donor-grouped OOF procedure ----
    perm_p = None
    if n_permutations and n_permutations > 0 and y_has_variance and np.isfinite(r1):
        donor_y_df = (
            pd.DataFrame({"d": donors_valid, "y": y_valid})
            .drop_duplicates("d")
        )
        donor_ids = donor_y_df["d"].to_numpy()
        donor_ys = donor_y_df["y"].to_numpy()
        null_abs_r = np.empty(n_permutations, dtype=np.float64)
        for i in range(n_permutations):
            shuffled_map = dict(zip(donor_ids, rng.permutation(donor_ys)))
            y_shuf = np.array([shuffled_map[d] for d in donors_valid], dtype=np.float64)
            oof_shuf = _oof_pls(
                X_valid, y_shuf, donors_valid, C_valid,
                1, n_splits, standardize, max_value, rng,
            )
            r_shuf, _ = stats.pearsonr(oof_shuf[:, 0], y_shuf)
            null_abs_r[i] = abs(r_shuf) if np.isfinite(r_shuf) else 0.0
        perm_p = float((np.sum(null_abs_r >= abs(r1)) + 1) / (n_permutations + 1))

    # ---- Write outputs ----
    full_oof = np.full((data.n_obs, n_components), np.nan, dtype=np.float64)
    full_oof[valid] = oof
    data.obsm[oof_components_key] = full_oof

    # Rank-transform PLS1 to [0,1] over valid cells; invalid -> NaN.
    score = np.full(data.n_obs, np.nan, dtype=np.float64)
    r = stats.rankdata(oof[:, 0], method="average")
    score[valid] = (r - 1) / max(len(r) - 1, 1)
    data.obs[score_key] = score

    data.uns[loadings_key] = full_x_loadings
    data.uns[weights_key] = full_x_weights

    diagnostics = {
        "pearson_r_pls1_y": r1,
        "between_donor_variance_fraction": between_frac,
        "within_donor_variance_mean": within_mean,
        "n_cells_scored": int(valid.sum()),
        "n_donors": int(len(unique_donors)),
        "n_components": int(n_components),
        "n_splits": int(n_splits),
        "permutation_p": perm_p,
    }
    logger.info(
        "pls_score: r(PLS1,y)={:.3f}  between-donor var frac={:.2f}  "
        "cells={}  donors={}  time={:.2f}s".format(
            diagnostics["pearson_r_pls1_y"],
            diagnostics["between_donor_variance_fraction"],
            diagnostics["n_cells_scored"],
            diagnostics["n_donors"],
            time.perf_counter() - start,
        )
    )
    if np.isfinite(between_frac) and between_frac > 0.8:
        logger.warning(
            "pls_score: >80%% of PLS1 variance is between-donor (%.2f). "
            "Treat this as a donor discriminator, not a disease axis.",
            between_frac,
        )
    return diagnostics


def _mixing_counts_matrix(adata, layer, use_raw):
    """Return a row-sliceable count matrix and its gene names."""
    if layer is not None and use_raw:
        raise ValueError("Pass either `layer` or `use_raw=True`, not both.")
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"`{layer}` not found in `adata.layers`.")
        return adata.layers[layer], adata.var_names
    if use_raw is None:
        use_raw = adata.raw is not None
    if use_raw:
        _require_raw(adata, context="`detect_lineage_mixing` with `use_raw=True`")
        return adata.raw.X, adata.raw.var_names
    return adata.X, adata.var_names


def _iter_row_chunks(M, chunk_size):
    """Yield (start, end, csr block) over contiguous row chunks of an in-memory or backed matrix."""
    n_obs = M.shape[0]
    for start in range(0, n_obs, chunk_size):
        end = min(n_obs, start + chunk_size)
        block = M[start:end]
        if not sparse.issparse(block):
            block = np.asarray(block)
        yield start, end, sparse.csr_matrix(block)


def _top_two(M):
    """Per-column largest and second-largest values of a 2D array (rows >= 2)."""
    part = np.partition(M, M.shape[0] - 2, axis=0)
    return part[-1], part[-2]


def _specific_genes(means, allowed, n_markers, min_lfc, min_expr):
    """Indices of the top group-specific genes: log2(mean_k / max other mean) >= min_lfc."""
    eps = 1e-7
    if means.shape[0] < 2:
        return set()
    first, second = _top_two(means)
    selected = set()
    for k in range(means.shape[0]):
        other = np.where(means[k] >= first, second, first)
        lfc = np.log2((means[k] + eps) / (other + eps))
        idx = np.where(allowed & (lfc >= min_lfc) & (means[k] >= min_expr))[0]
        selected.update(idx[np.argsort(-lfc[idx])][:n_markers].tolist())
    return selected


def _mixing_profiles(X, codes, n_groups, pseudocount=1e-6):
    """Row-normalized mean per-cell proportion profiles for each group code."""
    totals = np.asarray(X.sum(1)).ravel()
    valid = codes >= 0
    rows = np.where(valid)[0]
    norm = sparse.diags(1.0 / np.maximum(totals[rows], 1.0)) @ X[rows]
    S = sparse.csr_matrix((np.ones(rows.size), (codes[rows], np.arange(rows.size))), shape=(n_groups, rows.size))
    n_per = np.asarray(S.sum(1)).ravel()
    P = np.asarray((S @ norm).todense()) / np.maximum(n_per, 1)[:, None] + pseudocount
    return P / P.sum(1, keepdims=True)


def _fit_mixture_weights(X, P, max_iter=300, tol=1e-4, chunk_size=20000, extra=None, extra_index=None):
    """EM estimate of per-cell mixing weights over fixed expression profiles.

    Each row of ``X`` (counts) is modelled as multinomial with probabilities
    ``sum_k W[i, k] * P[k]``. If ``extra`` is given, each cell gets one more
    component, ``extra[extra_index[i]]``, whose weight is the last column of ``W``.
    Returns ``W`` (rows sum to 1) and the per-cell multinomial log-likelihood
    (without the constant term).
    """
    X = sparse.csr_matrix(X)
    n_obs = X.shape[0]
    P = np.asarray(P, dtype=np.float64)
    n_comp = P.shape[0] + (extra is not None)
    W = np.full((n_obs, n_comp), 1.0 / n_comp)
    loglik = np.zeros(n_obs)
    for start in range(0, n_obs, chunk_size):
        end = min(n_obs, start + chunk_size)
        Xc = X[start:end].astype(np.float64)
        Xc.sort_indices()
        rows = np.repeat(np.arange(end - start), np.diff(Xc.indptr))
        cols = Xc.indices
        totals = np.maximum(np.asarray(Xc.sum(1)).ravel(), 1.0)
        E = None if extra is None else np.asarray(extra, dtype=np.float64)[extra_index[start:end]]
        Wc = W[start:end]

        def mixture_at_nonzeros(Wc):
            dense = Wc[:, : P.shape[0]] @ P
            if E is not None:
                dense = dense + Wc[:, -1:] * E
            return dense[rows, cols]

        for _ in range(max_iter):
            ratio = Xc.data / mixture_at_nonzeros(Wc)
            R = sparse.csr_matrix((ratio, cols, Xc.indptr), shape=Xc.shape)
            grad = R @ P.T
            if E is not None:
                grad = np.column_stack([grad, np.bincount(rows, weights=ratio * E[rows, cols], minlength=end - start)])
            W_new = Wc * grad / totals[:, None]
            converged = np.abs(W_new - Wc).max() < tol
            Wc = W_new
            if converged:
                break
        W[start:end] = Wc
        loglik[start:end] = np.bincount(rows, weights=Xc.data * np.log(mixture_at_nonzeros(Wc)), minlength=end - start)
    return W, loglik


def detect_lineage_mixing(
    adata: ad.AnnData,
    groupby: str = "subclass",
    lineage_key: str = None,
    known_lineage_sets=None,
    layer: str = None,
    use_raw: bool = None,
    library_key: str = None,
    threshold: float = 0.3,
    summary_by=None,
    n_markers: int = 150,
    min_lfc: float = 1.0,
    min_expr_quantile: float = 0.5,
    exclude_genes=None,
    max_ref_cells: int = 5000,
    min_group_size: int = 20,
    core_quantile: float = 0.5,
    n_refine: int = 2,
    max_iter: int = 300,
    tol: float = 1e-4,
    chunk_size: int = 20000,
    key_added: str = "mixing",
    random_state: int = 0,
) -> pd.DataFrame:
    """Flag cells whose transcriptome is a mixture of cell types from different lineages.

    Heterotypic doublets, nuclei carrying attached material from another cell, and
    heavily ambient-contaminated nuclei all look like a mixture of two or more cell
    type profiles. Each cell's counts over an informative marker panel are modelled
    as ``x_i ~ Multinomial(n_i, sum_k w_ik * p_k)``, where ``p_k`` are fixed
    reference profiles for the ``groupby`` labels and ``w_i`` are per-cell mixing
    weights fitted by EM. The foreign score is the total weight on profiles outside
    the cell's own lineage, i.e. the fraction of the cell's marker-panel UMIs
    attributed to other lineages. Clean cells score close to 0; a doublet of two
    cells with similar panel depth scores around 0.5.

    Reference profiles are built from up to ``max_ref_cells`` cells per group and
    refined ``n_refine`` times on the purest ``1 - core_quantile`` of each group,
    so contaminated cells (a minority of their group) do not leak into the
    references. Use a coarse ``groupby`` (e.g. subclass): if a contaminated
    population is its own group, its profile absorbs the contamination.

    With ``library_key``, each flagged cell is also classified by where its
    foreign signal comes from, comparing the log-likelihood of "own lineage + the
    single best foreign profile" with that of "own lineage + the library's pooled
    profile" (a proxy for ambient RNA). ``"doublet"`` means one specific foreign
    cell type explains the signal better, as for a heterotypic doublet or
    attached material from one cell; ``"ambient"`` means a pooled, soup-like
    profile explains it better.

    Parameters
    ----------
    adata
        AnnData with raw counts in ``adata.raw.X`` (default when present),
        ``adata.layers[layer]``, or ``adata.X``. Backed AnnData is supported; the
        matrix is streamed in ``chunk_size`` row blocks.
    groupby
        ``.obs`` column with the cell type labels used as mixture components.
    lineage_key
        ``.obs`` column assigning each ``groupby`` label to a broader lineage
        (e.g. ``"class"``). Mixing between groups of the same lineage is not
        counted as foreign. Defaults to ``groupby``, so every group is its own
        lineage.
    known_lineage_sets
        Sets of lineages known to be connected, each treated as one lineage,
        given as a list of label collections, e.g. ``[("OPC", "Oligo")]``. Use
        this when cells transitioning between lineages are expected (such as
        differentiating OPCs) so they are not flagged; mixtures within a set are
        then no longer detected. Labels may be ``lineage_key`` values or
        ``groupby`` labels.
    library_key
        ``.obs`` column identifying each cell's sequencing library (e.g. 10x
        channel). Cells of one library share one ambient RNA pool, so pass the
        library rather than a coarser processing batch. Optional: it does not
        change the foreign score or the flag, but enables the doublet/ambient
        classification and a per-library summary.
    threshold
        Cells with ``{key_added}_UMI_frac >= threshold`` are flagged.
    summary_by
        ``.obs`` column(s) for the returned summary; defaults to ``groupby``. Use a
        finer label such as ``"subtype"`` to find whole clusters to drop.
    n_markers, min_lfc, min_expr_quantile
        Marker panel: up to ``n_markers`` genes per group and per lineage with
        log2 specificity ``>= min_lfc`` and mean expression above the
        ``min_expr_quantile`` quantile of per-gene maxima.
    exclude_genes
        Genes never used in the panel (list of names or boolean mask). Defaults to
        mitochondrial (``MT-``) and ribosomal protein (``RPL``/``RPS``) genes.
    min_group_size
        Groups with fewer cells get no profile; their cells are left unscored.

    Writes
    ------
    ``adata.obs``:

    - ``{key_added}_UMI_frac``: fraction of the cell's marker-panel UMIs
      attributed to other lineages; the score used for flagging.
    - ``{key_added}_RNA_frac``: the same converted to an estimated fraction of
      the cell's total RNA, using each profile's typical panel share.
    - ``{key_added}_top`` and ``{key_added}_top_weight``: the
      foreign group with the largest weight, and that weight.
    - ``{key_added}_flag``: 1 if flagged, else 0.
    - With ``library_key``, for flagged cells: ``{key_added}_source``
      (``"doublet"`` or ``"ambient"``) and ``{key_added}_doublet_logLR``, the
      log-likelihood ratio behind it (positive favours ``"doublet"``).

    ``adata.obsm[f"{key_added}_weights"]``: per-cell weights over groups.
    ``adata.uns[key_added]``: parameters, groups, lineages, panel genes,
    profiles, and summaries.

    Returns
    -------
    Per-``summary_by`` DataFrame with cell counts, median foreign score, flagged
    fraction, most frequent foreign type among flagged cells, and (with
    ``library_key``) the fraction of flagged cells classified as ``"doublet"``.
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"`groupby`={groupby!r} not in adata.obs.")
    if lineage_key is not None and lineage_key not in adata.obs.columns:
        raise KeyError(f"`lineage_key`={lineage_key!r} not in adata.obs.")
    if library_key is not None and library_key not in adata.obs.columns:
        raise KeyError(f"`library_key`={library_key!r} not in adata.obs.")
    if not 0 < threshold < 1:
        raise ValueError("`threshold` must be between 0 and 1.")
    if not 0 <= core_quantile < 1:
        raise ValueError("`core_quantile` must be in [0, 1).")

    start_time = time.perf_counter()
    rng = np.random.default_rng(random_state)
    M, gene_names = _mixing_counts_matrix(adata, layer, use_raw)
    if M.shape[0] != adata.n_obs:
        raise ValueError("Count matrix rows do not match `adata.n_obs`.")
    n_obs, n_genes = M.shape

    # ---- labels and lineages ----
    labels = adata.obs[groupby].astype("category")
    lineage_values = adata.obs[lineage_key].astype(str) if lineage_key is not None else labels.astype(str)
    lineage_of_group = {}
    for group, lineage in pd.DataFrame({"g": labels, "l": lineage_values}).dropna().drop_duplicates().itertuples(index=False):
        if group in lineage_of_group and lineage_of_group[group] != lineage:
            raise ValueError(f"Group {group!r} maps to more than one `{lineage_key}` value.")
        lineage_of_group[group] = lineage
    for merged in known_lineage_sets or []:
        merged = [str(label) for label in merged]
        known = {str(g) for g in lineage_of_group} | set(lineage_of_group.values())
        unknown = sorted(set(merged) - known)
        if unknown:
            raise KeyError(f"`known_lineage_sets` labels not found as groups or lineages: {unknown}.")
        merged_name = "+".join(merged)
        lineage_of_group = {
            g: merged_name if (str(g) in merged or lin in merged) else lin for g, lin in lineage_of_group.items()
        }

    sizes = labels.value_counts()
    groups = np.array(sorted(g for g in sizes.index if sizes[g] >= min_group_size), dtype=object)
    if len(groups) < 2:
        raise ValueError("Need at least two groups with `min_group_size` cells.")
    skipped = sorted(set(sizes.index[sizes > 0]) - set(groups))
    if skipped:
        logger.warning("detect_lineage_mixing: groups below min_group_size are left unscored: %s", skipped)
    n_groups = len(groups)
    group_lineage = np.array([lineage_of_group[g] for g in groups], dtype=object)
    codes = pd.Categorical(labels, categories=groups).codes.astype(np.int64)
    cell_lineage = labels.astype(object).map(lineage_of_group).to_numpy(dtype=object)

    ref_codes = np.full(n_obs, -1, dtype=np.int64)
    for k in range(n_groups):
        members = np.where(codes == k)[0]
        if members.size > max_ref_cells:
            members = rng.choice(members, size=max_ref_cells, replace=False)
        ref_codes[members] = k

    # ---- gene exclusion ----
    gene_names = pd.Index(gene_names).astype(str)
    if exclude_genes is None:
        excluded = np.asarray(gene_names.str.match(r"^(MT-|mt-|RP[LS]\d|Rp[ls]\d)"), dtype=bool)
    else:
        excluded = np.asarray(exclude_genes)
        if excluded.dtype != bool:
            excluded = np.asarray(gene_names.isin(list(exclude_genes)), dtype=bool)
    allowed = ~excluded

    # ---- pass 1: per-group mean proportions over all genes (reference cells) ----
    sums = np.zeros((n_groups, n_genes))
    totals = np.zeros(n_obs)
    checked_counts = False
    for start, end, block in _iter_row_chunks(M, chunk_size):
        if not checked_counts and block.nnz:
            sample = block.data[: min(block.nnz, 100_000)]
            if not np.allclose(sample, np.round(sample)):
                raise ValueError(
                    "detect_lineage_mixing expects raw counts; the selected matrix has non-integer values. "
                    "Use `use_raw=True` or `layer=` to point at counts."
                )
            checked_counts = True
        row_totals = np.asarray(block.sum(1)).ravel()
        totals[start:end] = row_totals
        chunk_codes = ref_codes[start:end]
        keep = np.where(chunk_codes >= 0)[0]
        if keep.size:
            norm = sparse.diags(1.0 / np.maximum(row_totals[keep], 1.0)) @ block[keep]
            S = sparse.csr_matrix((np.ones(keep.size), (chunk_codes[keep], np.arange(keep.size))), shape=(n_groups, keep.size))
            sums += np.asarray((S @ norm).todense())
    n_ref = np.bincount(ref_codes[ref_codes >= 0], minlength=n_groups)
    group_means = sums / np.maximum(n_ref, 1)[:, None]

    lineages = np.array(sorted(set(group_lineage)), dtype=object)
    lineage_means = np.vstack(
        [np.average(group_means[group_lineage == lin], axis=0, weights=n_ref[group_lineage == lin]) for lin in lineages]
    )
    min_expr = np.quantile(group_means[:, allowed].max(0), min_expr_quantile)
    panel = sorted(
        _specific_genes(group_means, allowed, n_markers, min_lfc, min_expr)
        | _specific_genes(lineage_means, allowed, n_markers, min_lfc, min_expr)
    )
    panel = np.asarray(panel, dtype=np.int64)
    if panel.size == 0:
        raise ValueError("No marker genes passed the panel filters; relax `min_lfc` or `min_expr_quantile`.")
    logger.info("detect_lineage_mixing: %d groups, %d lineages, %d panel genes.", n_groups, len(lineages), panel.size)

    # ---- pass 2: panel counts for every cell ----
    if sparse.issparse(M) and isinstance(M, (sparse.csr_matrix, sparse.csr_array)):
        Xp = sparse.csr_matrix(M[:, panel], dtype=np.float64)
    else:
        Xp = sparse.vstack([block[:, panel] for _, _, block in _iter_row_chunks(M, chunk_size)], format="csr").astype(np.float64)
    panel_fraction = np.asarray(Xp.sum(1)).ravel() / np.maximum(totals, 1.0)

    # ---- reference refinement on the purest cells of each group ----
    same_lineage = group_lineage[:, None] == group_lineage[None, :]
    ref_idx = np.where(ref_codes >= 0)[0]
    ref_group = ref_codes[ref_idx]
    core = ref_group.copy()
    for round_ in range(n_refine + 1):
        P = _mixing_profiles(Xp[ref_idx], core, n_groups)
        if round_ == n_refine:
            break
        W_ref, _ = _fit_mixture_weights(Xp[ref_idx], P, max_iter, tol, chunk_size)
        own_ref = (W_ref * same_lineage[ref_group]).sum(1)
        core = np.full(ref_idx.size, -1, dtype=np.int64)
        for k in range(n_groups):
            members = ref_group == k
            core[members & (own_ref >= np.quantile(own_ref[members], core_quantile))] = k
        logger.info(
            "detect_lineage_mixing: refinement %d, median reference foreign weight %.3f.",
            round_ + 1,
            float(np.median(1 - own_ref)),
        )

    # ---- score every cell ----
    W, _ = _fit_mixture_weights(Xp, P, max_iter, tol, chunk_size)
    scored = codes >= 0
    own_mask = np.zeros((n_obs, n_groups), dtype=bool)
    own_mask[scored] = group_lineage[None, :] == cell_lineage[scored, None]
    umi_frac = np.where(scored, 1 - (W * own_mask).sum(1), np.nan)
    foreign_weights = np.where(own_mask, -1.0, W)
    top_idx = foreign_weights.argmax(1)
    top_weight = np.where(scored, foreign_weights.max(1), np.nan)
    top = pd.Categorical(np.where(scored, groups[top_idx], None), categories=groups)

    # convert panel-UMI weights to total-RNA fractions with each profile's panel share
    core_cells = ref_idx[core >= 0]
    profile_panel_fraction = np.array(
        [np.median(panel_fraction[core_cells[core[core >= 0] == k]]) for k in range(n_groups)]
    )
    rna = W / np.maximum(profile_panel_fraction, 1e-9)
    rna /= rna.sum(1, keepdims=True)
    rna_frac = np.where(scored, 1 - (rna * own_mask).sum(1), np.nan)

    flag = scored & (umi_frac >= threshold)

    # ---- doublet-like (one foreign type) vs ambient-like (library pool) foreign signal ----
    log_lr = np.full(n_obs, np.nan)
    source = np.full(n_obs, None, dtype=object)
    if library_key is not None and flag.any():
        libraries = adata.obs[library_key].astype("category")
        library_codes = libraries.cat.codes.to_numpy()
        in_library = np.where(library_codes >= 0)[0]
        pool = sparse.csr_matrix(
            (np.ones(in_library.size), (library_codes[in_library], in_library)),
            shape=(len(libraries.cat.categories), n_obs),
        )
        ambient = np.asarray((pool @ Xp).todense()) + 1e-3
        ambient /= ambient.sum(1, keepdims=True)
        testable = flag & (library_codes >= 0)
        for lin in np.unique(cell_lineage[testable].astype(str)):
            idx = np.where(testable & (cell_lineage == lin))[0]
            own_profiles = P[group_lineage == lin]
            _, ll_doublet = _fit_mixture_weights(
                Xp[idx], own_profiles, max_iter, tol, chunk_size, extra=P, extra_index=top_idx[idx]
            )
            _, ll_ambient = _fit_mixture_weights(
                Xp[idx], own_profiles, max_iter, tol, chunk_size, extra=ambient, extra_index=library_codes[idx]
            )
            log_lr[idx] = ll_doublet - ll_ambient
        has_log_lr = ~np.isnan(log_lr)
        source[has_log_lr] = np.where(log_lr[has_log_lr] > 0, "doublet", "ambient")

    # ---- write outputs ----
    adata.obs[f"{key_added}_UMI_frac"] = umi_frac.astype(np.float32)
    adata.obs[f"{key_added}_RNA_frac"] = rna_frac.astype(np.float32)
    adata.obs[f"{key_added}_top"] = top
    adata.obs[f"{key_added}_top_weight"] = top_weight.astype(np.float32)
    adata.obs[f"{key_added}_flag"] = flag.astype(int)
    if library_key is not None:
        adata.obs[f"{key_added}_source"] = pd.Categorical(source, categories=["doublet", "ambient"])
        adata.obs[f"{key_added}_doublet_logLR"] = log_lr.astype(np.float32)
    adata.obsm[f"{key_added}_weights"] = W.astype(np.float32)

    summary_by = groupby if summary_by is None else summary_by
    frame = pd.DataFrame(
        {"umi_frac": umi_frac, "flag": flag, "top": top.astype(object), "source": source},
        index=adata.obs.index,
    )
    rows = []
    for group_name, positions in _iter_obs_groups(adata.obs, summary_by):
        sub = frame.iloc[positions]
        flagged = sub[sub["flag"]]
        row = {
            "group": group_name,
            "n_cells": int(len(sub)),
            "median_UMI_frac": float(sub["umi_frac"].median()),
            "n_flagged": int(len(flagged)),
            "flag_fraction": float(sub["flag"].mean()) if len(sub) else np.nan,
            "top": flagged["top"].mode().iloc[0] if len(flagged) else None,
        }
        if library_key is not None:
            row["doublet_fraction"] = float((flagged["source"] == "doublet").mean()) if len(flagged) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["top"] = summary["top"].astype("category")

    library_summary = None
    if library_key is not None:
        library_frame = frame.assign(library=adata.obs[library_key].astype(str).to_numpy())
        library_summary = library_frame.groupby("library").agg(
            n_cells=("flag", "size"),
            flag_fraction=("flag", "mean"),
            median_UMI_frac=("umi_frac", "median"),
        )
        library_summary["median_UMI_frac_unflagged"] = (
            library_frame[~library_frame["flag"]].groupby("library")["umi_frac"].median()
        )
        library_summary = library_summary.reset_index()

    adata.uns[key_added] = {
        "groupby": groupby,
        "threshold": float(threshold),
        "groups": groups.astype(str),
        "group_lineage": group_lineage.astype(str),
        "panel_genes": gene_names[panel].to_numpy().astype(str),
        "profiles": P.astype(np.float32),
        "profile_panel_fraction": profile_panel_fraction,
        "n_flagged": int(flag.sum()),
        "group_summary": summary,
    }
    if lineage_key is not None:
        adata.uns[key_added]["lineage_key"] = lineage_key
    if library_summary is not None:
        adata.uns[key_added]["library_key"] = library_key
        adata.uns[key_added]["library_summary"] = library_summary

    logger.info(
        "detect_lineage_mixing: flagged %d of %d scored cells (%.2f%%) in %.1fs.",
        int(flag.sum()),
        int(scored.sum()),
        100.0 * flag.sum() / max(int(scored.sum()), 1),
        time.perf_counter() - start_time,
    )
    return summary


__all__ = [
    "agg_by_cluster",
    "corrMat",
    "cos_similarity",
    "diff_markers",
    "detect_lineage_mixing",
    "detect_outliers",
    "info",
    "l2_distance",
    "pb_agg_by_cluster",
    "pearson_corr",
    "pls",
    "pls_score",
    "pseudoMetaCellByGroup",
    "spearman_corr",
]
