"""Analysis and tool helpers for SCTools."""

import os
from contextlib import nullcontext

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
    h5py,
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


def _result_target(adata):
    """Return the AnnData that receives per-cell results, and a function placing values at ``adata``'s rows.

    Writing into a view makes anndata copy it, which fails for backed AnnData and otherwise
    leaves the caller's object untouched. Results for a view therefore go into its parent:
    ``expand(values, fill)`` returns parent-length values with ``fill`` outside the view.
    """
    if not adata.is_view:
        return adata, lambda values, fill: values
    parent = adata._adata_ref
    positions = np.arange(parent.n_obs)[adata._oidx]

    def expand(values, fill):
        values = np.asarray(values)
        out = np.full((parent.n_obs,) + values.shape[1:], fill, dtype=values.dtype)
        out[positions] = values
        return out

    return parent, expand


def _refresh_view(adata):
    """Re-slice a view from its parent so it shows results just written to the parent."""
    if adata.is_view:
        adata._init_as_view(adata._adata_ref, adata._oidx, adata._vidx)


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
        Backed AnnData is supported. A view such as ``adata[mask]`` scores only
        its cells but writes the results into its parent ``adata`` (and the
        view); cells outside the view are left unscored (outlier 0, method
        ``"unassigned"``).
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

    target, expand = _result_target(adata)
    target.obs[outlier_key] = expand(outlier.astype(int), 0)

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
        target.obs[f"{outlier_key}_method"] = pd.Categorical(expand(method_used, "unassigned"))
        target.obs[f"{outlier_key}_score"] = expand(score, np.nan)
        if np.any(pd.notna(dbscan_label)):
            target.obs[f"{outlier_key}_dbscan_label"] = pd.array(expand(dbscan_label, pd.NA), dtype="Int64")

        target.uns[diagnostics_key] = {
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

    _refresh_view(adata)
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

    Run this on an AnnData already subset to one cell type. Each cell is scored
    by a model fitted without its donor, so the score is not fitted to the cell
    it describes.

    Parameters
    ----------
    data
        AnnData subset to one cell type.
    y
        ``.obs`` column with a donor-level quantitative phenotype, such as
        Braak stage, CERAD score, a polygenic risk score, or a continuous
        biomarker.
    donor_key
        ``.obs`` column identifying each cell's donor; cross-validation folds
        never split a donor.
    covariates
        ``.obs`` columns regressed out of the expression matrix within each
        training fold. Categorical columns are one-hot encoded.
    n_permutations
        Number of donor-level shuffles of ``y`` for the permutation test.
        0 skips the test.

    Writes
    ------
    - ``data.obs[score_key]``: cell score in [0, 1], the rank of PLS1; NaN for
      cells without a finite ``y`` or a donor.
    - ``data.obsm[oof_components_key]``: ``(n_cells, n_components)``
      out-of-fold components.
    - ``data.uns[loadings_key]``: gene loadings from the full-data fit.
    - ``data.uns[weights_key]``: ``x_weights_`` from the full-data fit, for
      projecting new data.

    Returns
    -------
    Dict of diagnostics:

    - ``pearson_r_pls1_y``: correlation of out-of-fold PLS1 with ``y``.
    - ``between_donor_variance_fraction``: share of PLS1 variance between
      donors. Above 0.8, PLS1 mostly separates donors, not disease states.
    - ``within_donor_variance_mean``: mean PLS1 variance within a donor.
    - ``permutation_p``: permutation p-value, or None if the test did not run.
    - ``n_cells_scored``, ``n_donors``, ``n_components``, ``n_splits``.
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
    """Return a row-sliceable count matrix, its gene names, and the rows of it holding ``adata``'s cells.

    For a view of a backed AnnData this is the parent's matrix with the view's rows, so counts
    stream from the file rather than through anndata's backed view indexing, which loads them
    all and fails with some anndata/scipy combinations. Otherwise rows is None (all rows).
    """
    if layer is not None and use_raw:
        raise ValueError("Pass either `layer` or `use_raw=True`, not both.")
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"`{layer}` not found in `adata.layers`.")
        return adata.layers[layer], adata.var_names, None

    if use_raw is None:
        use_raw = adata.raw is not None
    if use_raw:
        _require_raw(adata, context="`detect_lineage_mixing` with `use_raw=True`")
    source, rows = adata, None
    if adata.is_view and adata.isbacked:
        parent = adata._adata_ref
        positions = np.arange(parent.n_obs)[adata._oidx]
        # raw ignores the view's gene subset; X only matches the parent if the view keeps every gene
        same_genes = use_raw or adata.var_names.equals(parent.var_names)
        if same_genes and np.all(np.diff(positions) > 0):
            source, rows = parent, positions
    M, gene_names = (source.raw.X, source.raw.var_names) if use_raw else (source.X, source.var_names)

    group, encoding = _backed_sparse_group(M)
    if encoding == "csc_matrix":
        # rows of a CSC matrix span every column, so read it into memory once
        M = sparse.csc_matrix((group["data"][:], group["indices"][:], group["indptr"][:]), shape=M.shape).tocsr()
        if rows is not None:
            M, rows = M[rows], None
    return M, gene_names, rows


def _backed_sparse_group(M):
    """Return (HDF5 group, encoding) behind a backed AnnData sparse matrix, else (None, None)."""
    group = getattr(M, "group", None)
    if not isinstance(group, h5py.Group):
        return None, None
    encoding = group.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    return (group, encoding) if encoding in ("csr_matrix", "csc_matrix") else (None, None)


def _iter_row_chunks(M, chunk_size, rows=None):
    """Yield (start, end, csr block) over contiguous row chunks of an in-memory or backed matrix.

    With ``rows`` (increasing row positions), blocks hold ``M[rows[start:end]]`` and at most
    ``chunk_size`` consecutive rows of ``M`` are read at a time.
    """
    n_obs, n_vars = M.shape
    # Backed CSR rows are read straight from HDF5: anndata's backed indexing fails with some
    # anndata/scipy combinations (e.g. anndata 0.10 with scipy >= 1.15).
    group, encoding = _backed_sparse_group(M)
    indptr = group["indptr"][:] if encoding == "csr_matrix" else None
    for first in range(0, n_obs, chunk_size):
        last = min(n_obs, first + chunk_size)
        start, end = (first, last) if rows is None else np.searchsorted(rows, [first, last])
        if start == end:
            continue
        if rows is not None:
            first, last = rows[start], rows[end - 1] + 1
        if indptr is not None:
            lo, hi = indptr[first], indptr[last]
            block = sparse.csr_matrix(
                (group["data"][lo:hi], group["indices"][lo:hi], indptr[first : last + 1] - lo),
                shape=(last - first, n_vars),
            )
        else:
            block = M[first:last]
            if not sparse.issparse(block):
                block = np.asarray(block)
        block = sparse.csr_matrix(block)
        if rows is not None:
            block = block[rows[start:end] - first]
        yield start, end, block


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


def _lineage_nuclei(
    W, n_panel, codes, umi_frac, group_lineage, lineages, library_codes, n_libraries, core_quantile, min_cells,
    max_iter=100, tol=1e-6,
):
    """Per-cell RNA of each lineage in nucleus equivalents.

    A group's typical cells are the ``1 - core_quantile`` of its cells with the lowest
    ``umi_frac``. Their panel UMIs are modelled as (group size) x (library depth) and both
    factors are fitted jointly by alternating medians on the log scale (median polish), so
    a group whose cells sit mostly in deep libraries is not mistaken for a large one.
    Depth is relative to the median library; it is 1 without ``library_codes`` and for
    libraries with fewer than ``min_cells`` typical cells. Group ``k`` adds
    ``W[:, k] * n_panel / (size[k] * depth)`` nuclei to its lineage. Returns the
    (cells x lineages) amounts, the typical panel UMIs per group in the median library, and
    the depth factor per library (None without ``library_codes``).
    """
    n_obs, n_groups = W.shape

    # Typical cells define "one nucleus" of each group: the purest (1 - core_quantile) of the
    # group by foreign share, chosen across all libraries at once. Cells carrying extra material
    # have more UMIs and a higher share, so they drop out. (Choosing the purest cells within each
    # library instead shifts the depth of libraries with unusual ambient RNA, and was less
    # accurate in testing.)
    is_typical = np.zeros(n_obs, dtype=bool)
    for k in range(n_groups):
        members = np.where(codes == k)[0]
        if members.size:
            is_typical[members[umi_frac[members] <= np.quantile(umi_frac[members], 1 - core_quantile)]] = True

    # Model for typical cells, on the log scale:
    #   log(panel UMIs) = log(size of its group) + log(depth of its library) + noise.
    # Cells without a library (code -1) and cells without panel UMIs are left out of the fit.
    use = is_typical & (n_panel > 0)
    if library_codes is not None:
        use &= library_codes >= 0
    y, g = np.log(n_panel[use]), codes[use]

    def group_medians(values):
        # a group without usable typical cells gets the median group size, not 1 UMI
        medians = pd.Series(values).groupby(g).median().reindex(range(n_groups)).to_numpy()
        return np.where(np.isnan(medians), np.nanmedian(medians) if np.isfinite(medians).any() else 0.0, medians)

    # Starting point: each group's median over all libraries pooled. On its own this is the
    # one-pass estimate. It is biased when a group's cells sit mostly in deep (or shallow)
    # libraries, because the pooled median then also measures depth: in brain tissue,
    # glia-rich libraries tend to be sequenced deeper, so glia look larger than they are and
    # attached glial RNA is undercounted.
    log_size = group_medians(y)
    library_depth = None
    depth = np.ones(n_obs)
    if library_codes is not None:
        lib = library_codes[use]
        # Libraries with too few typical cells for a stable median keep depth 1 (the median library).
        fitted = np.bincount(lib, minlength=n_libraries) >= min_cells
        log_depth = np.zeros(n_libraries)
        # Fit size and depth jointly by alternating medians (median polish, i.e. backfitting on
        # absolute deviations):
        #   depth of a library = median over its typical cells of (log UMIs - log size of their group)
        #   size of a group    = median over its typical cells of (log UMIs - log depth of their library)
        # The first round equals the one-pass estimate; each further round removes the depth that
        # the pooled group medians had absorbed. At convergence every group and every fitted
        # library has a median residual of 0. With real (noisy) counts this takes a few dozen
        # rounds at most and lands on the least-absolute-deviation fit.
        for _ in range(max_iter):
            new_depth = pd.Series(y - log_size[g]).groupby(lib).median().reindex(range(n_libraries)).to_numpy()
            new_depth = np.where(fitted, new_depth, np.nan)
            # Size and depth are identified only up to a shared constant. Centring depth on the
            # median fitted library puts sizes in "panel UMIs in the median library" units.
            new_depth = np.nan_to_num(new_depth - np.nanmedian(new_depth), nan=0.0) if fitted.any() else log_depth
            new_size = group_medians(y - new_depth[lib])
            change = max(np.abs(new_depth - log_depth).max(), np.abs(new_size - log_size).max())
            log_depth, log_size = new_depth, new_size
            if change < tol:
                break
        library_depth = np.exp(log_depth)
        in_library = library_codes >= 0
        depth[in_library] = library_depth[library_codes[in_library]]
    # floor at 1 UMI so a group can never divide by ~0
    typical = np.maximum(np.exp(log_size), 1.0)

    # Nucleus equivalents: W[:, k] * n_panel is the cell's panel UMIs from group k. Dividing by
    # the cell's library depth puts them in median-library units, and dividing by typical[k]
    # counts them in typical nuclei of group k. Summed over the groups of each lineage.
    amounts = W * (n_panel / depth)[:, None] / typical[None, :]
    membership = (group_lineage[:, None] == lineages[None, :]).astype(np.float64)
    return amounts @ membership, typical, library_depth


def detect_lineage_mixing(
    adata: ad.AnnData,
    groupby: str = "subclass",
    lineage_key: str = None,
    known_lineage_sets=None,
    layer: str = None,
    use_raw: bool = None,
    library_key: str = None,
    score: str = "UMI_frac",
    threshold: float = None,
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

    With ``score="nuclei"``, cells are flagged on the amount of foreign RNA
    instead of its share: the largest amount from any single other lineage, in
    nucleus equivalents. 1.0 is the panel RNA of one typical nucleus of that
    type in the cell's library, so a heterotypic doublet scores about 1 and
    ambient RNA, spread thinly over all lineages, a few hundredths. The share
    depends on the size of the host nucleus, so small nuclei with ordinary
    ambient RNA can pass a share threshold while large nuclei carrying a whole
    small cell stay below it; the amount does not have that bias.

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
        matrix is streamed in ``chunk_size`` row blocks. A view such as
        ``adata[mask]`` scores only its cells but writes the results into its
        parent ``adata`` (and the view); cells outside the view are left
        unscored.
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
        channel). Cells of one library share one ambient RNA pool and one
        sequencing depth, so pass the library rather than a coarser processing
        batch. It enables the doublet/ambient classification, a per-library
        summary, and the per-library depth factor of ``{key_added}_nuclei``
        (recommended with ``score="nuclei"``). It does not change
        ``{key_added}_UMI_frac``.
    score
        Score to flag on: ``"UMI_frac"`` (``{key_added}_UMI_frac``, the share of
        the cell's marker-panel UMIs from other lineages) or ``"nuclei"``
        (``{key_added}_nuclei``, the largest amount of RNA from any single
        other lineage, in nucleus equivalents).
    threshold
        Cells whose ``score`` is ``>= threshold`` are flagged. Defaults to 0.3
        for ``"UMI_frac"`` and 0.25 for ``"nuclei"``. A quarter of a nucleus is
        far above ambient RNA (a few hundredths); in human cortex, nuclei with
        attached material from another cell carry 0.3-0.6 of a nucleus and
        heterotypic doublets about 1.
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
        Libraries with fewer typical cells (see ``core_quantile``) get a depth
        factor of 1.
    core_quantile
        References are refined on the purest ``1 - core_quantile`` of each
        group's reference cells. The same share of each group's scored cells,
        those with the lowest ``{key_added}_UMI_frac``, are its typical cells.
        Their panel UMIs are modelled as group size times library depth, fitted
        jointly by alternating medians, so a group whose cells sit mostly in
        deep libraries is not mistaken for a large one. One nucleus of a group
        is its typical panel UMIs scaled to the cell's library.

    Writes
    ------
    ``adata.obs``:

    - ``{key_added}_UMI_frac``: fraction of the cell's marker-panel UMIs
      attributed to other lineages; the flagging score with
      ``score="UMI_frac"``.
    - ``{key_added}_RNA_frac``: the same converted to an estimated fraction of
      the cell's total RNA, using each profile's typical panel share.
    - ``{key_added}_nuclei`` and ``{key_added}_nuclei_lineage``: the largest
      amount of RNA from any single other lineage, in nucleus equivalents, and
      that lineage; the flagging score with ``score="nuclei"``. A group's
      weight times the cell's panel UMIs, divided by the panel UMIs of a
      typical cell of the group in the median library and, with
      ``library_key``, by the library's depth factor.
    - ``{key_added}_top`` and ``{key_added}_top_weight``: the
      foreign group with the largest weight, and that weight.
    - ``{key_added}_flag``: 1 if flagged, else 0.
    - With ``library_key``, for flagged cells: ``{key_added}_source``
      (``"doublet"`` or ``"ambient"``) and ``{key_added}_doublet_logLR``, the
      log-likelihood ratio behind it (positive favours ``"doublet"``).

    ``adata.obsm[f"{key_added}_weights"]``: per-cell weights over groups.
    ``adata.obsm[f"{key_added}_lineage_nuclei"]``: per-cell nucleus equivalents
    of every lineage in ``uns[key_added]["lineages"]``, the cell's own included.
    ``adata.uns[key_added]``: parameters, groups, lineages, panel genes,
    profiles, typical panel UMIs per group, and summaries (with
    ``library_key``, the per-library summary holds the depth factors).

    Returns
    -------
    Per-``summary_by`` DataFrame with cell counts, median foreign scores, flagged
    fraction, most frequent foreign type among flagged cells, and (with
    ``library_key``) the fraction of flagged cells classified as ``"doublet"``.
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"`groupby`={groupby!r} not in adata.obs.")
    if lineage_key is not None and lineage_key not in adata.obs.columns:
        raise KeyError(f"`lineage_key`={lineage_key!r} not in adata.obs.")
    if library_key is not None and library_key not in adata.obs.columns:
        raise KeyError(f"`library_key`={library_key!r} not in adata.obs.")
    if score not in ("UMI_frac", "nuclei"):
        raise ValueError("`score` must be 'UMI_frac' or 'nuclei'.")
    if threshold is None:
        # "nuclei": in human cortex the score is bimodal, with ambient-level cells at a few
        # hundredths of a nucleus and cells carrying attached material at about half a nucleus.
        # 0.25 sits in the valley between the two.
        threshold = 0.3 if score == "UMI_frac" else 0.25
    if score == "UMI_frac" and not 0 < threshold < 1:
        raise ValueError("`threshold` must be between 0 and 1 with score='UMI_frac'.")
    if score == "nuclei" and not threshold > 0:
        raise ValueError("`threshold` must be positive with score='nuclei'.")
    if not 0 <= core_quantile < 1:
        raise ValueError("`core_quantile` must be in [0, 1).")
    if score == "nuclei" and library_key is None:
        logger.warning(
            "detect_lineage_mixing: score='nuclei' without `library_key` uses one depth factor for all cells; "
            "pass the sequencing library to adjust for differences in depth between libraries."
        )

    start_time = time.perf_counter()
    rng = np.random.default_rng(random_state)
    M, gene_names, rows = _mixing_counts_matrix(adata, layer, use_raw)
    if (M.shape[0] if rows is None else rows.size) != adata.n_obs:
        raise ValueError("Count matrix rows do not match `adata.n_obs`.")
    n_obs, n_genes = adata.n_obs, M.shape[1]

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
    for start, end, block in _iter_row_chunks(M, chunk_size, rows):
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
    if rows is None and sparse.issparse(M) and isinstance(M, (sparse.csr_matrix, sparse.csr_array)):
        Xp = sparse.csr_matrix(M[:, panel], dtype=np.float64)
    else:
        Xp = sparse.vstack(
            [block[:, panel] for _, _, block in _iter_row_chunks(M, chunk_size, rows)], format="csr"
        ).astype(np.float64)
    n_panel = np.asarray(Xp.sum(1)).ravel()
    panel_fraction = n_panel / np.maximum(totals, 1.0)

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
    top_label = np.where(scored, groups[top_idx], None)
    top = pd.Categorical(top_label, categories=groups)

    # convert panel-UMI weights to total-RNA fractions with each profile's panel share
    core_cells = ref_idx[core >= 0]
    profile_panel_fraction = np.array(
        [np.median(panel_fraction[core_cells[core[core >= 0] == k]]) for k in range(n_groups)]
    )
    rna = W / np.maximum(profile_panel_fraction, 1e-9)
    rna /= rna.sum(1, keepdims=True)
    rna_frac = np.where(scored, 1 - (rna * own_mask).sum(1), np.nan)

    # ---- foreign RNA amount: largest single other lineage, in nucleus equivalents ----
    # umi_frac is a share, so it depends on the host's size. A small host with ordinary ambient
    # RNA can pass a share cutoff, and a large host carrying a whole small cell can stay below
    # it. The amount does not have this problem: each group's RNA is counted in typical nuclei
    # of that group in the cell's library (see _lineage_nuclei) and summed per lineage. The
    # score is the largest lineage other than the cell's own. Ambient RNA is spread thinly over
    # all lineages, so no single lineage gets much of it; material from one attached cell is
    # concentrated in one lineage. Taking the largest single lineage, not the sum of all others,
    # keeps ambient RNA from adding up across lineages.
    libraries = library_codes = None
    if library_key is not None:
        libraries = adata.obs[library_key].astype("category")
        library_codes = libraries.cat.codes.to_numpy()
    lineage_nuclei, typical_panel_umis, library_depth = _lineage_nuclei(
        W,
        n_panel,
        codes,
        umi_frac,
        group_lineage,
        lineages,
        library_codes,
        None if libraries is None else len(libraries.cat.categories),
        core_quantile,
        min_group_size,
    )
    # "own" is the lineage of the cell's label, not the lineage with the most RNA, so a neuron
    # labelled as glia is scored on its neuronal RNA.
    own_lineage = scored[:, None] & (lineages[None, :] == cell_lineage[:, None])
    foreign_nuclei = np.where(own_lineage, 0.0, lineage_nuclei)
    nuclei = np.where(scored, foreign_nuclei.max(1), np.nan)
    nuclei_lineage = np.where(scored & (nuclei > 0), lineages[foreign_nuclei.argmax(1)], None)
    if library_depth is not None:
        logger.info(
            "detect_lineage_mixing: library depth factors %.2f-%.2f (median %.2f).",
            float(library_depth.min()),
            float(library_depth.max()),
            float(np.median(library_depth)),
        )

    flag = scored & ((nuclei if score == "nuclei" else umi_frac) >= threshold)

    # ---- doublet-like (one foreign type) vs ambient-like (library pool) foreign signal ----
    log_lr = np.full(n_obs, np.nan)
    source = np.full(n_obs, None, dtype=object)
    if library_key is not None and flag.any():
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

    # ---- write outputs (into the parent AnnData when `adata` is a view) ----
    target, expand = _result_target(adata)
    target.obs[f"{key_added}_UMI_frac"] = expand(umi_frac.astype(np.float32), np.nan)
    target.obs[f"{key_added}_RNA_frac"] = expand(rna_frac.astype(np.float32), np.nan)
    target.obs[f"{key_added}_nuclei"] = expand(nuclei.astype(np.float32), np.nan)
    target.obs[f"{key_added}_nuclei_lineage"] = pd.Categorical(expand(nuclei_lineage, None), categories=lineages)
    target.obs[f"{key_added}_top"] = pd.Categorical(expand(top_label, None), categories=groups)
    target.obs[f"{key_added}_top_weight"] = expand(top_weight.astype(np.float32), np.nan)
    target.obs[f"{key_added}_flag"] = expand(flag.astype(int), 0)
    if library_key is not None:
        target.obs[f"{key_added}_source"] = pd.Categorical(expand(source, None), categories=["doublet", "ambient"])
        target.obs[f"{key_added}_doublet_logLR"] = expand(log_lr.astype(np.float32), np.nan)
    target.obsm[f"{key_added}_weights"] = expand(W.astype(np.float32), np.nan)
    target.obsm[f"{key_added}_lineage_nuclei"] = expand(lineage_nuclei.astype(np.float32), np.nan)

    summary_by = groupby if summary_by is None else summary_by
    frame = pd.DataFrame(
        {"umi_frac": umi_frac, "nuclei": nuclei, "flag": flag, "top": top.astype(object), "source": source},
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
            "median_nuclei": float(sub["nuclei"].median()),
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
            median_nuclei=("nuclei", "median"),
        )
        library_summary["median_UMI_frac_unflagged"] = (
            library_frame[~library_frame["flag"]].groupby("library")["umi_frac"].median()
        )
        library_summary["depth_factor"] = pd.Series(library_depth, index=libraries.cat.categories.astype(str))
        library_summary = library_summary.reset_index()

    target.uns[key_added] = {
        "groupby": groupby,
        "score": score,
        "threshold": float(threshold),
        "groups": groups.astype(str),
        "group_lineage": group_lineage.astype(str),
        "lineages": lineages.astype(str),
        "panel_genes": gene_names[panel].to_numpy().astype(str),
        "profiles": P.astype(np.float32),
        "profile_panel_fraction": profile_panel_fraction,
        "typical_panel_UMIs": typical_panel_umis,
        "n_flagged": int(flag.sum()),
        "group_summary": summary,
    }
    if lineage_key is not None:
        target.uns[key_added]["lineage_key"] = lineage_key
    if library_summary is not None:
        target.uns[key_added]["library_key"] = library_key
        target.uns[key_added]["library_summary"] = library_summary
    _refresh_view(adata)

    logger.info(
        "detect_lineage_mixing: flagged %d of %d scored cells (%.2f%%) with %s >= %g in %.1fs.",
        int(flag.sum()),
        int(scored.sum()),
        100.0 * flag.sum() / max(int(scored.sum()), 1),
        score,
        threshold,
        time.perf_counter() - start_time,
    )
    return summary


# ---------------------------------------------------------------------------------------------
# kNN doublet score
# ---------------------------------------------------------------------------------------------


def _cell_ranger_hvg(sums, sumsq, n_cells, n_top, allowed):
    """Highly variable genes as Scanpy's ``highly_variable_genes(flavor="cell_ranger")`` on log-normalized data.

    ``sums`` and ``sumsq`` are per-gene sums of the log-normalized values and of their squares over
    ``n_cells`` cells. As Scanpy, the dispersion is var / mean (variance with ddof=1) of those values;
    genes are binned at the 10th, 15th, ..., 100th percentiles of the mean, each dispersion is centred
    on its bin median and scaled by the bin's MAD, and every gene at or above the ``n_top``-th highest
    normalized dispersion is kept. Only ``allowed`` genes detected in at least one cell are considered.
    Returns sorted gene indices.
    """
    candidates = np.flatnonzero(allowed & (sums > 0))
    if candidates.size == 0:
        raise ValueError("No candidate genes for the embedding: check `gene_mask` and the count matrix.")
    mean = sums[candidates] / n_cells
    var = np.maximum(sumsq[candidates] - n_cells * mean**2, 0.0) / max(n_cells - 1, 1)
    mean[mean == 0] = 1e-12
    dispersion = var / mean
    edges = np.r_[-np.inf, np.percentile(mean, np.arange(10, 105, 5)), np.inf]
    bins = pd.cut(mean, np.unique(edges))
    grouped = pd.Series(dispersion).groupby(bins, observed=True)

    def mad(values):
        values = np.asarray(values, dtype=float)
        return np.median(np.abs(values - np.median(values))) / 0.6744897501960817

    median = grouped.transform("median").to_numpy()
    spread = grouped.transform(mad).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (dispersion - median) / spread
    finite = normalized[~np.isnan(normalized)]
    if n_top >= finite.size:
        keep = ~np.isnan(normalized)
    else:
        cutoff = np.sort(finite)[::-1][n_top - 1]
        keep = np.nan_to_num(normalized, nan=-np.inf) >= cutoff
    return np.sort(candidates[keep])


def _log_normalize_rows(X, totals, genes, target_sum):
    """log1p(count / total * target_sum) for the ``genes`` columns of csr ``X`` (all if None), as float32 csr."""
    Xg = sparse.csr_matrix(X if genes is None else X[:, genes], dtype=np.float64)
    rows = np.repeat(np.arange(Xg.shape[0]), np.diff(Xg.indptr))
    Xg.data = np.log1p(Xg.data / np.maximum(totals[rows], 1.0) * target_sum)
    return Xg.astype(np.float32)


def _project_rows(Xlog, model):
    """Coordinates of log-normalized rows in the fitted embedding (standardize, clip, centre, rotate)."""
    Z = Xlog.toarray() if sparse.issparse(Xlog) else np.asarray(Xlog, dtype=np.float32)
    Z -= model["mean"]
    Z /= model["std"]
    np.clip(Z, -model["clip"], model["clip"], out=Z)
    Z -= model["zmean"]
    return Z @ model["loadings"]


def _knn_numpy(queries, reference, k, n_jobs, query_block=256, reference_block=32768):
    """Exact Euclidean k nearest neighbours on the CPU, sorted by distance (blocked, multithreaded NumPy)."""
    from concurrent.futures import ThreadPoolExecutor

    reference = np.ascontiguousarray(reference, dtype=np.float32)
    ref_norm = np.einsum("ij,ij->i", reference, reference, dtype=np.float64).astype(np.float32)
    k = min(k, reference.shape[0])

    def search(start):
        q = np.ascontiguousarray(queries[start : start + query_block], dtype=np.float32)
        best_d = np.full((q.shape[0], k), np.inf, dtype=np.float32)
        best_i = np.zeros((q.shape[0], k), dtype=np.int64)
        for first in range(0, reference.shape[0], reference_block):
            # ||q - r||^2 up to the per-query constant ||q||^2, which does not change the ranking
            d = q @ reference[first : first + reference_block].T
            d *= -2.0
            d += ref_norm[None, first : first + reference_block]
            kk = min(k, d.shape[1])
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
            cand_d = np.concatenate([best_d, np.take_along_axis(d, part, axis=1)], axis=1)
            cand_i = np.concatenate([best_i, part + first], axis=1)
            keep = np.argpartition(cand_d, k - 1, axis=1)[:, :k]
            best_d = np.take_along_axis(cand_d, keep, axis=1)
            best_i = np.take_along_axis(cand_i, keep, axis=1)
        order = np.argsort(best_d, axis=1, kind="stable")
        return np.take_along_axis(best_i, order, axis=1)

    try:  # one BLAS thread per worker; the workers already use every core
        from threadpoolctl import threadpool_limits

        limits = threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        limits = nullcontext()
    with limits, ThreadPoolExecutor(max_workers=n_jobs) as pool:
        parts = list(pool.map(search, range(0, queries.shape[0], query_block)))
    return np.vstack(parts) if parts else np.zeros((0, k), dtype=np.int64)


def _knn_torch(queries, reference, k, device, memory_bytes=1 << 30):
    """Exact Euclidean k nearest neighbours with PyTorch, sorted by distance."""
    import torch

    ref = torch.as_tensor(np.ascontiguousarray(reference, dtype=np.float32), device=device)
    ref_norm = (ref * ref).sum(1)
    k = min(k, ref.shape[0])
    batch = max(64, int(memory_bytes // (4 * max(ref.shape[0], 1))))
    out = []
    with torch.no_grad():
        for start in range(0, queries.shape[0], batch):
            q = torch.as_tensor(np.ascontiguousarray(queries[start : start + batch], dtype=np.float32), device=device)
            d = ref_norm[None, :] - 2.0 * (q @ ref.T)
            out.append(torch.topk(d, k, dim=1, largest=False, sorted=True).indices.cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, k), dtype=np.int64)


def _resolve_knn_backend(backend):
    """Return ("cpu", None) or ("gpu", device) for ``backend`` in {"auto", "cpu", "gpu"}."""
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("`backend` must be 'auto', 'cpu' or 'gpu'.")
    if backend == "cpu":
        return "cpu", None
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        torch, cuda = None, False
    if backend == "gpu" and not cuda:
        raise RuntimeError("backend='gpu' needs PyTorch with a CUDA device; use backend='cpu' or 'auto'.")
    return ("gpu", "cuda") if cuda else ("cpu", None)


def _drop_self(indices, self_index):
    """Remove each query's own reference position from its k+1 sorted neighbours (else the farthest)."""
    is_self = indices == self_index[:, None]
    keep = ~is_self
    keep[~is_self.any(1), -1] = False
    return indices[keep].reshape(indices.shape[0], indices.shape[1] - 1)


def detect_knn_doublets(
    adata: ad.AnnData,
    groupby: str = None,
    library_key: str = None,
    reference=None,
    layer: str = None,
    use_raw: bool = None,
    gene_mask=None,
    n_top_genes: int = 3000,
    n_pcs: int = 30,
    n_neighbors: int = 50,
    n_reference: int = 200_000,
    n_simulated: int = 200_000,
    hvg_cells_per_group: int = 2000,
    pca_cells_per_group: int = 5000,
    target_sum: float = 1e4,
    quantile: float = 0.99,
    threshold: float = None,
    backend: str = "auto",
    n_jobs: int = None,
    chunk_size: int = 20000,
    store_pca: bool = False,
    key_added: str = "knn_doublet",
    random_state: int = 0,
) -> pd.DataFrame:
    """Flag doublets by their neighbourhood among simulated doublets, pooled over all libraries.

    Run this after the per-library doublet step and cleaning (e.g. Scrublet, then
    :func:`detect_lineage_mixing` and :func:`detect_outliers`), on the cells you intend to keep.
    It is the Scrublet idea applied once to the whole dataset:

    1. Embedding: ``n_top_genes`` highly variable genes (Scanpy ``flavor="cell_ranger"`` on
       log1p(count / total * ``target_sum``)) chosen on up to ``hvg_cells_per_group`` reference cells
       per ``groupby`` group, standardized (clipped at 10) and reduced to ``n_pcs`` principal
       components fitted on up to ``pca_cells_per_group`` reference cells per group. Balancing by
       group keeps rare types resolved.
    2. Reference: ``n_reference`` random reference cells as singlets, plus ``n_simulated`` doublets
       made by summing the counts of two random reference cells of the same library
       (``library_key``). Libraries are drawn with probability proportional to the square of their
       number of reference cells, as doublets scale with loading.
    3. Score: the fraction of simulated doublets among each cell's ``n_neighbors`` nearest
       neighbours in the embedding (a cell is never its own neighbour).
    4. Flag: cells scoring at or above ``threshold``; by default the ``quantile`` of the reference
       cells' own scores (cells above 0 when that quantile is 0).

    The reference cells are the singlets, the doublet parents and the cells the embedding is fitted
    on; every cell in ``adata`` is scored. Because both the reference and the simulated doublets come
    from the data, no external annotation is needed.

    Parameters
    ----------
    adata
        AnnData with raw counts; may be backed or a view of a backed AnnData.
    groupby
        ``.obs`` column of cell-type labels used to balance the embedding. Optional; without it the
        embedding cells are a random sample.
    library_key
        ``.obs`` column identifying each cell's library (10x channel); simulated doublets pair cells
        of the same library. Optional; without it, any two reference cells can pair.
    reference
        Cells to use as the reference: a boolean ``.obs`` column name, a boolean array, or any
        selector accepted by :func:`detect_outliers`. Defaults to all cells.
    layer, use_raw
        Where the counts are, as in :func:`detect_lineage_mixing`.
    gene_mask
        Genes allowed in the embedding: a boolean array over the count matrix's genes, a boolean
        column of its ``var``, or a list of gene names. Defaults to all genes except mitochondrial
        (``MT-``) and ribosomal protein (``RPL``/``RPS``) genes.
    n_top_genes, n_pcs
        Embedding size.
    n_neighbors
        Neighbours per cell; the score moves in steps of ``1 / n_neighbors``.
    n_reference, n_simulated
        Numbers of reference singlets and simulated doublets (capped at the number of reference
        cells for singlets). Keep them equal so a score of 0.5 means an even neighbourhood.
    hvg_cells_per_group, pca_cells_per_group
        Reference cells per group used to choose genes and fit the PCA.
    target_sum
        Counts per cell after depth normalization, before log1p.
    quantile
        Quantile of the reference cells' scores used as the threshold when ``threshold`` is None.
    threshold
        Score at or above which a cell is flagged; overrides ``quantile``.
    backend
        ``"cpu"`` (exact search, NumPy, multithreaded), ``"gpu"`` (exact search, PyTorch with CUDA),
        or ``"auto"`` (GPU if available). Both return the same neighbours up to floating-point ties.
    n_jobs
        CPU threads for ``backend="cpu"``; defaults to all cores.
    chunk_size
        Cells per block when streaming counts.
    store_pca
        Also write the embedding to ``.obsm[f"{key_added}_pca"]``.
    key_added
        Prefix of the output keys.
    random_state
        Seed for sampling cells and doublet pairs.

    Writes
    ------
    ``adata.obs``: ``{key_added}_score`` (fraction of simulated doublets among the neighbours) and
    ``{key_added}_flag`` (1 if flagged). ``adata.uns[key_added]``: parameters, threshold, the genes,
    the embedding (``mean``, ``std``, ``zmean``, ``loadings``) and a per-group summary.

    Returns
    -------
    Per-group DataFrame (``groupby``, or one row ``"all"``) with cell counts, median score and the
    flagged fraction.
    """
    if groupby is not None and groupby not in adata.obs.columns:
        raise KeyError(f"`groupby`={groupby!r} not in adata.obs.")
    if library_key is not None and library_key not in adata.obs.columns:
        raise KeyError(f"`library_key`={library_key!r} not in adata.obs.")
    if threshold is None and not 0 < quantile < 1:
        raise ValueError("`quantile` must be between 0 and 1.")
    if n_neighbors < 1 or n_pcs < 1 or n_top_genes < n_pcs:
        raise ValueError("Need n_neighbors >= 1, n_pcs >= 1 and n_top_genes >= n_pcs.")
    if n_reference < n_neighbors or n_simulated < 1:
        raise ValueError("Need n_reference >= n_neighbors and n_simulated >= 1.")
    if target_sum is None or not target_sum > 0:
        raise ValueError("`target_sum` must be positive.")
    backend, device = _resolve_knn_backend(backend)
    n_jobs = (os.cpu_count() or 1) if n_jobs is None else max(1, int(n_jobs))

    start_time = time.perf_counter()
    rng = np.random.default_rng(random_state)
    M, gene_names, rows = _mixing_counts_matrix(adata, layer, use_raw)
    n_obs, n_genes = adata.n_obs, M.shape[1]
    if (M.shape[0] if rows is None else rows.size) != n_obs:
        raise ValueError("Count matrix rows do not match `adata.n_obs`.")
    gene_names = pd.Index(gene_names).astype(str)

    # ---- genes allowed in the embedding ----
    if gene_mask is None:
        allowed = ~np.asarray(gene_names.str.match(r"^(MT-|mt-|RP[LS]\d|Rp[ls]\d)"), dtype=bool)
    elif isinstance(gene_mask, str):
        raw_used = layer is None and (use_raw if use_raw is not None else adata.raw is not None)
        var = adata.raw.var if raw_used else adata.var
        if gene_mask not in var.columns:
            raise KeyError(f"`gene_mask`={gene_mask!r} is not a column of the count matrix's var.")
        allowed = var[gene_mask].to_numpy(dtype=bool)
    else:
        allowed = np.asarray(gene_mask)
        if allowed.dtype != bool:
            allowed = np.asarray(gene_names.isin([str(g) for g in gene_mask]), dtype=bool)
    if allowed.shape[0] != n_genes:
        raise ValueError(f"`gene_mask` must cover the {n_genes} genes of the count matrix.")

    # ---- reference cells, groups, libraries ----
    if reference is None:
        is_ref = np.ones(n_obs, dtype=bool)
    elif isinstance(reference, str):
        if reference not in adata.obs.columns:
            raise KeyError(f"`reference`={reference!r} not in adata.obs.")
        is_ref = adata.obs[reference].to_numpy(dtype=bool)
    else:
        is_ref = _as_obs_mask(adata, reference, name="reference")
    ref_cells = np.flatnonzero(is_ref)
    if ref_cells.size < max(n_neighbors + 1, 3):
        raise ValueError(f"Need more reference cells than n_neighbors; found {ref_cells.size}.")
    group_codes = (
        adata.obs[groupby].astype("category").cat.codes.to_numpy() if groupby is not None else np.zeros(n_obs, dtype=int)
    )
    lib_codes = (
        adata.obs[library_key].astype("category").cat.codes.to_numpy() if library_key is not None else np.zeros(n_obs, dtype=int)
    )

    def balanced(per_group):
        picks = []
        for code in np.unique(group_codes[ref_cells]):
            if code < 0:
                continue
            members = ref_cells[group_codes[ref_cells] == code]
            picks.append(rng.permutation(members)[:per_group])
        return np.sort(np.concatenate(picks)) if picks else np.zeros(0, dtype=int)

    hvg_cells = balanced(hvg_cells_per_group)
    pca_cells = balanced(pca_cells_per_group)
    if hvg_cells.size < 2 or pca_cells.size <= n_pcs:
        raise ValueError("Too few reference cells with a `groupby` label to fit the embedding.")

    # singlets and simulated doublet pairs (within library, libraries weighted by n^2)
    singlets = np.sort(rng.choice(ref_cells, size=min(n_reference, ref_cells.size), replace=False))
    lib_members = pd.Series(ref_cells).groupby(lib_codes[ref_cells]).apply(np.asarray)
    lib_members = lib_members[(lib_members.map(len) >= 2) & (lib_members.index >= 0)]  # -1: no library label
    if lib_members.empty:
        raise ValueError("No library has two reference cells to pair into simulated doublets.")
    weights = lib_members.map(len).to_numpy(dtype=float) ** 2
    per_lib = rng.multinomial(n_simulated, weights / weights.sum())
    parent_a, parent_b = [], []
    for members, count in zip(lib_members.to_numpy(), per_lib):
        if count:
            first = rng.integers(0, len(members), count)
            second = (first + rng.integers(1, len(members), count)) % len(members)
            parent_a.append(members[first])
            parent_b.append(members[second])
    parent_a, parent_b = np.concatenate(parent_a), np.concatenate(parent_b)
    logger.info(
        "detect_knn_doublets: %d reference cells, %d singlets, %d simulated doublets from %d libraries; backend %s.",
        ref_cells.size, singlets.size, parent_a.size, len(lib_members), backend,
    )

    # ---- pass 1: totals of every cell; moments of log-normalized values on the gene-selection cells ----
    totals = np.zeros(n_obs)
    sums, sumsq = np.zeros(n_genes), np.zeros(n_genes)
    in_hvg = np.zeros(n_obs, dtype=bool)
    in_hvg[hvg_cells] = True
    checked = False
    for start, end, block in _iter_row_chunks(M, chunk_size, rows):
        if not checked and block.nnz:
            sample = block.data[: min(block.nnz, 100_000)]
            if not np.allclose(sample, np.round(sample)):
                raise ValueError(
                    "detect_knn_doublets expects raw counts; the selected matrix has non-integer values. "
                    "Use `use_raw=True` or `layer=` to point at counts."
                )
            checked = True
        totals[start:end] = np.asarray(block.sum(1)).ravel()
        picked = np.flatnonzero(in_hvg[start:end])
        if picked.size:
            logged = _log_normalize_rows(block[picked], totals[start:end][picked], None, target_sum).astype(np.float64)
            sums += np.asarray(logged.sum(0)).ravel()
            sumsq += np.asarray(logged.multiply(logged).sum(0)).ravel()
    genes = _cell_ranger_hvg(sums, sumsq, hvg_cells.size, n_top_genes, allowed)
    if genes.size <= n_pcs:
        raise ValueError(f"Only {genes.size} genes available for {n_pcs} principal components.")

    # ---- pass 2: gene-subset counts of the PCA cells and of the doublet parents ----
    needed = np.union1d(pca_cells, np.union1d(parent_a, parent_b))
    is_needed = np.zeros(n_obs, dtype=bool)
    is_needed[needed] = True
    kept = []
    for start, end, block in _iter_row_chunks(M, chunk_size, rows):
        picked = np.flatnonzero(is_needed[start:end])
        if picked.size:
            kept.append(sparse.csr_matrix(block[picked][:, genes], dtype=np.float32))
    H = sparse.vstack(kept, format="csr")
    row_of = np.full(n_obs, -1, dtype=np.int64)
    row_of[needed] = np.arange(needed.size)

    # ---- embedding: standardize (clip 10) and PCA on the PCA cells ----
    pca_rows = row_of[pca_cells]
    Xp = _log_normalize_rows(H[pca_rows], totals[pca_cells], None, target_sum).astype(np.float64)
    n_p = Xp.shape[0]
    mean = np.asarray(Xp.sum(0)).ravel() / n_p
    var = (np.asarray(Xp.multiply(Xp).sum(0)).ravel() - n_p * mean**2) / max(n_p - 1, 1)
    model = {"mean": mean.astype(np.float32), "std": np.maximum(np.sqrt(np.maximum(var, 0)), 1e-6).astype(np.float32), "clip": 10.0}
    zsum = np.zeros(genes.size)
    zcross = np.zeros((genes.size, genes.size))
    for first in range(0, n_p, chunk_size):
        Z = Xp[first : first + chunk_size].toarray().astype(np.float32)
        Z -= model["mean"]
        Z /= model["std"]
        np.clip(Z, -10.0, 10.0, out=Z)
        zsum += Z.sum(0, dtype=np.float64)
        zcross += Z.T.astype(np.float64) @ Z.astype(np.float64)
    zmean = zsum / n_p
    cov = (zcross - n_p * np.outer(zmean, zmean)) / max(n_p - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:n_pcs]
    model["zmean"] = zmean.astype(np.float32)
    model["loadings"] = np.ascontiguousarray(evecs[:, order], dtype=np.float32)
    explained = evals[order] / max(evals.sum(), 1e-12)
    del Xp, zcross, cov

    # ---- pass 3: every cell's coordinates; simulated doublets' coordinates ----
    pcs = np.zeros((n_obs, n_pcs), dtype=np.float32)
    for start, end, block in _iter_row_chunks(M, chunk_size, rows):
        pcs[start:end] = _project_rows(_log_normalize_rows(block, totals[start:end], genes, target_sum), model)
    sim = np.zeros((parent_a.size, n_pcs), dtype=np.float32)
    for first in range(0, parent_a.size, chunk_size):
        a, b = parent_a[first : first + chunk_size], parent_b[first : first + chunk_size]
        D = H[row_of[a]] + H[row_of[b]]
        sim[first : first + chunk_size] = _project_rows(_log_normalize_rows(D, totals[a] + totals[b], None, target_sum), model)
    del H

    # ---- kNN score: fraction of simulated doublets among the neighbours, self excluded ----
    R = np.vstack([pcs[singlets], sim])
    is_sim = np.r_[np.zeros(singlets.size, dtype=bool), np.ones(sim.shape[0], dtype=bool)]
    self_index = np.full(n_obs, -1, dtype=np.int64)
    self_index[singlets] = np.arange(singlets.size)
    k1 = min(n_neighbors + 1, R.shape[0])
    if backend == "gpu":
        neighbours = _knn_torch(pcs, R, k1, device)
    else:
        neighbours = _knn_numpy(pcs, R, k1, n_jobs)
    neighbours = _drop_self(neighbours, self_index)
    score = is_sim[neighbours].mean(1).astype(np.float32)
    if threshold is None:
        threshold = float(np.quantile(score[is_ref], quantile))
    flag = score >= threshold if threshold > 0 else score > 0

    # ---- write results ----
    target, expand = _result_target(adata)
    target.obs[f"{key_added}_score"] = expand(score, np.nan)
    target.obs[f"{key_added}_flag"] = expand(flag.astype(int), 0)
    if store_pca:
        target.obsm[f"{key_added}_pca"] = expand(pcs, np.nan)

    labels = adata.obs[groupby].astype(str).to_numpy() if groupby is not None else np.full(n_obs, "all", dtype=object)
    frame = pd.DataFrame({"group": labels, "score": score, "flag": flag})
    summary = (
        frame.groupby("group", sort=True)
        .agg(n_cells=("flag", "size"), median_score=("score", "median"), n_flagged=("flag", "sum"), flag_fraction=("flag", "mean"))
        .reset_index()
    )
    target.uns[key_added] = {
        "groupby": groupby,
        "library_key": library_key,
        "threshold": float(threshold),
        "quantile": float(quantile),
        "n_neighbors": int(n_neighbors),
        "n_reference_cells": int(ref_cells.size),
        "n_singlets": int(singlets.size),
        "n_simulated": int(parent_a.size),
        "target_sum": float(target_sum),
        "backend": backend,
        "genes": gene_names[genes].to_numpy().astype(str),
        "mean": model["mean"],
        "std": model["std"],
        "zmean": model["zmean"],
        "loadings": model["loadings"],
        "explained_variance_ratio": explained.astype(np.float32),
        "group_summary": summary,
    }
    _refresh_view(adata)
    logger.info(
        "detect_knn_doublets: flagged %d of %d cells (%.2f%%) with score >= %.3f in %.1fs.",
        int(flag.sum()), n_obs, 100.0 * flag.mean(), threshold, time.perf_counter() - start_time,
    )
    return summary


__all__ = [
    "agg_by_cluster",
    "corrMat",
    "cos_similarity",
    "diff_markers",
    "detect_knn_doublets",
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
