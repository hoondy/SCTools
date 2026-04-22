"""Analysis and tool helpers for SCTools."""

from ._shared import (
    _get_sample_size,
    _require_group_kfold,
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
    "pls_score",
    "pseudoMetaCellByGroup",
    "spearman_corr",
]
