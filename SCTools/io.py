"""Input/output helpers for SCTools."""

from ._shared import (
    Path,
    _as_sparse_dataset,
    _empty_csr,
    _matrix_shape,
    _require_csr_group,
    _require_sparse_group,
    _skip_anndata_index_checks,
    _write_raw_annotations,
    ad,
    clean_unused_categories,
    gc,
    h5py,
    np,
    pd,
    read_elem,
    sparse,
    write_elem,
)


# Default `obs` columns kept by `proc_h5ad` (input name -> output name), matching
# vireoSNP-demultiplexed pools that also carry hashtag (`_cs`) calls.
_PROC_H5AD_OBS_COLUMNS = {
    "SubID_vS": "SubID",
    "rep": "rep",
    "poolID_ref": "poolID",
    "round_num": "round_num",
    "prep": "prep",
    "SubID_cs": "SubID_cs",
    "HTO_n_cs": "HTO_n_cs",
    "max_prob": "max_prob",
    "doublet_prob": "doublet_prob",
}


def proc_h5ad(
    filepath,
    dummy_adata=None,
    subject_col="SubID_vS",
    rep_col="rep",
    obs_columns=None,
    source_col="Source",
):
    """Harmonize the annotations of one demultiplexed pool H5AD.

    Adds ``obs["Channel"]`` as ``"<subject>-<rep>"`` and renames each cell to
    ``"<Channel>-<barcode>"``, so cells that share a 10x barcode in different
    pools stay distinct after concatenation. Only the ``obs`` columns in
    ``obs_columns`` are kept, and ``"_index"`` is removed from gene names.

    Parameters
    ----------
    filepath
        H5AD file to read.
    dummy_adata
        Optional AnnData with no cells, whose genes form the gene set of every
        output. When given, the result is outer-joined to these genes, which
        must come out sorted, and ``X`` gets int64 indices so that many
        outputs can be concatenated. ``dummy_adata`` is not modified.
    subject_col, rep_col
        ``obs`` columns joined to form ``Channel``.
    obs_columns
        ``obs`` columns to keep, as a dict of input name to output name, or a
        list to keep the names unchanged. Defaults to ``SubID_vS`` (as
        ``SubID``), ``rep``, ``poolID_ref`` (as ``poolID``), ``round_num``,
        ``prep``, ``SubID_cs``, ``HTO_n_cs``, ``max_prob`` and
        ``doublet_prob``.
    source_col
        Output column for the first character of each subject ID. Set to
        ``None`` to skip it.
    """
    if obs_columns is None:
        obs_columns = _PROC_H5AD_OBS_COLUMNS
    elif not isinstance(obs_columns, dict):
        obs_columns = {col: col for col in obs_columns}

    with _skip_anndata_index_checks():
        adata = ad.read_h5ad(filepath)

    missing = [col for col in dict.fromkeys([subject_col, rep_col, *obs_columns]) if col not in adata.obs]
    if missing:
        raise KeyError(f"`obs` of {filepath} is missing columns: {missing}.")

    subject = adata.obs[subject_col].astype(str)
    channel = subject + "-" + adata.obs[rep_col].astype(str)

    obs = adata.obs[list(obs_columns)].rename(columns=obs_columns)
    obs.insert(0, "Channel", channel)
    if source_col is not None:
        obs[source_col] = subject.str[0]
    obs.index = pd.Index([x[0] + "-" + x[1] for x in zip(channel, adata.obs_names)], name="barcodekey")
    adata.obs = obs

    adata.var.index = [x.replace("_index", "") for x in adata.var.index]

    if dummy_adata is not None:
        with _skip_anndata_index_checks():
            adata = ad.concat([dummy_adata, adata], join="outer", merge="same")
        adata.X.sort_indices()

        if adata.var_names.to_list() != sorted(adata.var_names):
            raise AssertionError("Expected concatenated `var_names` to be sorted.")

        adata.X.indptr = adata.X.indptr.astype(np.int64)
        adata.X.indices = adata.X.indices.astype(np.int64)

    return adata


def proc_manifest(manifest_file, prefix, postfix, chunk_size, dummy_adata, **kwargs):
    """Process manifest rows in chunks and write chunked H5AD outputs.

    Each file is read with `proc_h5ad`; extra keyword arguments are passed to it.
    """
    list_h5ad_parts = []
    df = pd.read_csv(manifest_file)
    chunks = [df.Location[i : i + chunk_size] for i in range(0, len(df.Location), chunk_size)]

    for j, list_filepath in enumerate(chunks):
        list_adata = []
        for filepath in list_filepath:
            print(filepath)
            list_adata.append(proc_h5ad(filepath, dummy_adata, **kwargs))

        outfilename = prefix + "_raw_" + postfix + str((j + 1)) + ".h5ad"

        with _skip_anndata_index_checks():
            adata = ad.concat(list_adata, join="outer", merge="same")
        adata.X.sort_indices()
        adata.X.indptr = adata.X.indptr.astype(np.int64)
        adata.X.indices = adata.X.indices.astype(np.int64)
        adata.write(outfilename)

        list_h5ad_parts.append(outfilename)

        del list_adata
        del adata
        gc.collect()

    return list_h5ad_parts


def read_everything_but_X(pth) -> ad.AnnData:
    """Read an H5AD without loading `X` or `raw/X`."""
    with h5py.File(pth) as f:
        attrs = [key for key in f.keys() if key not in {"X", "raw"}]
        with _skip_anndata_index_checks():
            adata = ad.AnnData(**{k: read_elem(f[k]) for k in attrs})
        print(adata.shape)
    return adata


def _convert_sparse_on_disk(input_pth, output_pth, src_format, raw):
    """Rewrite an H5AD with `X`, and `raw/X` if `raw`, switched between CSC and CSR."""
    keys = ["X"]
    with h5py.File(input_pth, "r") as src:
        if raw and "raw" in src and "X" in src["raw"]:
            keys.append("raw/X")
        for key in keys:
            _require_sparse_group(src[key], src_format, label=f"`input_pth/{key}`")

    annotations = read_everything_but_X(input_pth)
    annotations.write_h5ad(output_pth)
    del annotations

    src_cls, dst_format = (sparse.csc_matrix, "csr") if src_format == "csc" else (sparse.csr_matrix, "csc")
    with h5py.File(output_pth, "a") as target:
        with h5py.File(input_pth, "r") as src:
            if raw:
                _write_raw_annotations(target, src)
            for key in keys:
                group = src[key]
                mat = src_cls(
                    (group["data"], group["indices"], group["indptr"]),
                    shape=_matrix_shape(group),
                )
                dest = target if key == "X" else target.require_group("raw")
                write_elem(dest, "X", mat.asformat(dst_format))
                # Free `X` before loading `raw/X`, so only one matrix is in memory at a time.
                del mat


def csc2csr_on_disk(input_pth, output_pth, raw=False):
    """Convert an H5AD stored in CSC format to CSR on disk.

    Each matrix is converted in memory, which takes about twice its size. With
    ``raw=True``, ``raw/X`` is converted after ``X`` and ``raw/var`` is kept;
    otherwise ``raw`` is dropped.
    """
    _convert_sparse_on_disk(input_pth, output_pth, "csc", raw)


def csr2csc_on_disk(input_pth, output_pth, raw=False):
    """Convert an H5AD stored in CSR format to CSC on disk.

    Each matrix is converted in memory, which takes about twice its size. With
    ``raw=True``, ``raw/X`` is converted after ``X`` and ``raw/var`` is kept;
    otherwise ``raw`` is dropped.
    """
    _convert_sparse_on_disk(input_pth, output_pth, "csr", raw)


def concat_on_disk(input_pths, output_pth, temp_pth="temp.h5ad"):
    """Concatenate H5AD files without materializing the full matrix in memory."""
    with _skip_anndata_index_checks():
        annotations = ad.concat([read_everything_but_X(pth) for pth in input_pths], merge="same")
    annotations.write_h5ad(output_pth)
    n_variables = annotations.shape[1]

    del annotations

    with h5py.File(output_pth, "a") as target:
        write_elem(target, "X", _empty_csr(n_variables))

        mtx = _as_sparse_dataset(target["X"])
        for pth in input_pths:
            with h5py.File(pth, "r") as src:
                if src["X"].attrs["encoding-type"] == "csc_matrix":
                    csc_mat = sparse.csc_matrix(
                        (src["X"]["data"], src["X"]["indices"], src["X"]["indptr"]),
                        shape=_matrix_shape(src["X"]),
                    )
                    csr_mat = csc_mat.tocsr()

                    with h5py.File(temp_pth, "w") as tmp:
                        write_elem(tmp, "X", csr_mat)

                    with h5py.File(temp_pth, "r") as tmp:
                        mtx.append(_as_sparse_dataset(tmp["X"]))
                else:
                    mtx.append(_as_sparse_dataset(src["X"]))

    temp_path = Path(temp_pth)
    if temp_path.exists():
        temp_path.unlink()


def write_h5ad_with_new_annotation(orig_h5ad, adata, new_h5ad, raw=True):
    """Write new annotations while reusing the original on-disk `X` matrix."""
    new_uns = adata.uns if adata.uns else None
    new_obsm = adata.obsm if adata.obsm else None
    new_varm = adata.varm if adata.varm else None
    new_obsp = adata.obsp if adata.obsp else None
    new_varp = adata.varp if adata.varp else None
    new_layers = adata.layers if adata.layers else None

    with _skip_anndata_index_checks():
        ad.AnnData(
            None,
            obs=adata.obs,
            var=adata.var,
            uns=new_uns,
            obsm=new_obsm,
            varm=new_varm,
            obsp=new_obsp,
            varp=new_varp,
            layers=new_layers,
        ).write(new_h5ad)

    with h5py.File(new_h5ad, "a") as target:
        with h5py.File(orig_h5ad, "r") as src:
            write_elem(target, "X", _empty_csr(adata.var.shape[0]))
            _as_sparse_dataset(target["X"]).append(_as_sparse_dataset(src["X"]))

            if raw and "raw" in src:
                target.copy(src["raw"], "raw")


def ondisk_subset(orig_h5ad, new_h5ad, subset_obs, subset_var=None, chunk_size=500000, raw=False, adata=None):
    """Subset a CSR-backed H5AD on disk without loading the full matrix."""
    if adata is None:
        adata = read_everything_but_X(orig_h5ad)

        if subset_obs is not None:
            adata._inplace_subset_obs(subset_obs)
        if subset_var is not None:
            adata._inplace_subset_var(subset_var)

        adata = clean_unused_categories(adata)

    if subset_obs is None:
        subset_obs = np.ones(adata.shape[0], dtype=bool)
    else:
        subset_obs = np.asarray(subset_obs)

    if subset_var is not None:
        subset_var = np.asarray(subset_var)

    new_uns = adata.uns if adata.uns else None
    new_obsm = adata.obsm if adata.obsm else None
    new_varm = adata.varm if adata.varm else None
    new_obsp = adata.obsp if adata.obsp else None
    new_varp = adata.varp if adata.varp else None
    new_layers = adata.layers if adata.layers else None

    with _skip_anndata_index_checks():
        ad.AnnData(
            None,
            obs=adata.obs,
            var=adata.var,
            uns=new_uns,
            obsm=new_obsm,
            varm=new_varm,
            obsp=new_obsp,
            varp=new_varp,
            layers=new_layers,
        ).write(new_h5ad)

    with h5py.File(new_h5ad, "a") as target:
        write_elem(target, "X", _empty_csr(adata.var.shape[0]))
        with h5py.File(orig_h5ad, "r") as src:
            if raw and "raw" in src:
                raw_n_vars = _write_raw_annotations(target, src, subset_var=subset_var)
                if raw_n_vars is not None:
                    write_elem(target.require_group("raw"), "X", _empty_csr(raw_n_vars))

    with h5py.File(orig_h5ad, "r") as f:
        _require_csr_group(f["X"], label="`orig_h5ad/X`")
        csr_indptr = f["X/indptr"][:]
        n_obs, n_vars = _matrix_shape(f["X"])

    for idx in range(0, csr_indptr.shape[0] - 1, chunk_size):
        print("Processing", idx, "to", idx + chunk_size)
        row_start, row_end = idx, min(idx + chunk_size, n_obs)

        if np.sum(subset_obs[row_start:row_end]) > 0:
            with h5py.File(orig_h5ad, "r") as f:
                tmp_indptr = csr_indptr[row_start : row_end + 1]

                new_data = f["X/data"][tmp_indptr[0] : tmp_indptr[-1]]
                new_indices = f["X/indices"][tmp_indptr[0] : tmp_indptr[-1]]
                new_indptr = tmp_indptr - csr_indptr[row_start]

                tmp_csr = sparse.csr_matrix(
                    (new_data, new_indices, new_indptr),
                    shape=(tmp_indptr.shape[0] - 1, n_vars),
                )
                tmp_csr = tmp_csr[subset_obs[row_start:row_end]]
                if subset_var is not None:
                    tmp_csr = tmp_csr[:, subset_var]
                tmp_csr.sort_indices()

            with h5py.File(new_h5ad, "a") as target:
                _as_sparse_dataset(target["X"]).append(tmp_csr)

            if raw:
                with h5py.File(orig_h5ad, "r") as f:
                    if "raw" not in f or "X" not in f["raw"]:
                        continue
                    _require_csr_group(f["raw/X"], label="`orig_h5ad/raw/X`")
                    raw_indptr = f["raw/X/indptr"][:]
                    raw_n_obs, raw_n_vars = _matrix_shape(f["raw/X"])
                    if raw_n_obs != n_obs:
                        raise ValueError("`raw/X` must have the same number of observations as `X`.")

                    tmp_indptr = raw_indptr[row_start : row_end + 1]

                    new_data = f["raw/X/data"][tmp_indptr[0] : tmp_indptr[-1]]
                    new_indices = f["raw/X/indices"][tmp_indptr[0] : tmp_indptr[-1]]
                    new_indptr = tmp_indptr - raw_indptr[row_start]

                    tmp_csr = sparse.csr_matrix(
                        (new_data, new_indices, new_indptr),
                        shape=(tmp_indptr.shape[0] - 1, raw_n_vars),
                    )
                    tmp_csr = tmp_csr[subset_obs[row_start:row_end]]
                    if subset_var is not None:
                        tmp_csr = tmp_csr[:, subset_var]
                    tmp_csr.sort_indices()

                with h5py.File(new_h5ad, "a") as target:
                    _as_sparse_dataset(target["raw/X"]).append(tmp_csr)


def save(data, filename):
    """Convert a Pegasus object to AnnData and write it to disk."""
    if "_tmp_fmat_highly_variable_features" in data.uns:
        del data.uns["_tmp_fmat_highly_variable_features"]
    with _skip_anndata_index_checks():
        data.to_anndata().write(filename)
    print("Saved", filename)


__all__ = [
    "concat_on_disk",
    "csc2csr_on_disk",
    "csr2csc_on_disk",
    "ondisk_subset",
    "proc_h5ad",
    "proc_manifest",
    "read_everything_but_X",
    "save",
    "write_h5ad_with_new_annotation",
]
