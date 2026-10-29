# SCTools

Helper functions for single-cell RNA-seq analysis with
[Pegasus](https://pegasus.readthedocs.io) and
[Scanpy](https://scanpy.readthedocs.io), plus utilities for H5AD files too
large to load into memory.

```python
import SCTools as sct

adata = sct.io.read_everything_but_X("/path/to/file.h5ad")  # annotations only, no expression matrix
```

## Installation

SCTools needs Python 3.8 or later. The base install needs only AnnData, h5py,
NumPy, pandas, and SciPy:

```bash
pip install git+https://github.com/hoondy/SCTools
```

Many `pp`, `tl`, and `pl` functions also need Pegasus, Scanpy, scikit-learn, or
plotting libraries. To install these too, use the `analysis` extra:

```bash
pip install "SCTools[analysis] @ git+https://github.com/hoondy/SCTools"
```

These optional packages are imported only when a function uses them.
`import SCTools` works without them. A function that needs a missing package
raises an `ImportError` that names the package.

## What's included

Functions are grouped into four namespaces. Call them through their namespace,
for example `sct.io.read_everything_but_X(...)`, not
`sct.read_everything_but_X(...)`.

### `sct.io`: reading and writing H5AD files

| Function | What it does |
| --- | --- |
| `read_everything_but_X` | Load a file's annotations (`obs`, `var`, `obsm`, `uns`, ...) without `X` or `raw/X` |
| `ondisk_subset` | Write a subset of cells and genes to a new file, copying `X` in chunks |
| `write_h5ad_with_new_annotation` | Save edited annotations together with the original file's `X` |
| `concat_on_disk` | Concatenate several files, appending each matrix on disk |
| `csc2csr_on_disk` | Rewrite a file whose `X` is stored as CSC with a CSR `X` |
| `save` | Convert a Pegasus object to AnnData and write it |
| `proc_h5ad_v2`, `proc_h5ad_v3`, `proc_manifest` | Import files from one project's pipeline; they expect specific `obs` columns such as `SubID_cs` and `rep` |

See [Working with large H5AD files](#working-with-large-h5ad-files) for examples.

### `sct.pp`: preprocessing

| Function | What it does |
| --- | --- |
| `scanpy_hvf` | Select highly variable genes with Scanpy on a Pegasus object; sets `var["highly_variable_features"]` |
| `scanpy_hvf_h5ad` | Select highly variable genes from an H5AD file in backed mode; returns the gene names |
| `scanpy_pca` | Run Scanpy PCA on a Pegasus object and copy the results back |
| `qc_boundary`, `mad_boundary` | Lower and upper QC cutoffs at the median ± k × MAD (`qc_boundary` works on log1p counts) |
| `calc_zscore_per_cluster` | Z-score one `obs` variable within each cluster |
| `calc_sig_scores` | Compute Pegasus's built-in signature scores: cell cycle, sex, mitochondrial, ribosomal, and apoptosis genes |
| `mark_MitoCarta` | Flag MitoCarta 3.0 genes and mark them as not robust; pass the reference CSV as `mitocarta_path=` |
| `clean_unused_categories` | Drop unused categories from categorical `obs` and `var` columns |

### `sct.tl`: analysis

| Function | What it does |
| --- | --- |
| `detect_outliers` | Flag cells that sit apart from their annotation group in an embedding ([details](#outlier-detection)) |
| `detect_lineage_mixing` | Flag heterotypic doublets and cross-lineage contamination ([details](#lineage-mixing-detection-heterotypic-doublets-and-contamination)) |
| `pls`, `pls_score` | Supervised embedding and per-cell disease score with partial least squares ([details](#pls-supervised-disease-axis)) |
| `agg_by_cluster` | Mean scaled log-CPM expression per cluster, from an H5AD file's `raw.X` |
| `pb_agg_by_cluster` | Pseudobulk expression per cluster, using Pegasus |
| `cos_similarity`, `l2_distance`, `pearson_corr`, `spearman_corr` | Compare every row of one expression matrix with every row of another, using their shared genes |
| `corrMat` | Gene–gene correlation matrix from a random sample of cells |
| `pseudoMetaCellByGroup` | Sum counts over small groups of neighboring cells (pseudo-metacells), writing one H5AD per group |
| `diff_markers` | Up-regulated markers of one cluster that are not markers of another, from Pegasus marker results |
| `info` | Print the installed versions of Pegasus, Scanpy, AnnData, NumPy, and pandas |

### `sct.pl`: plotting

| Function | What it does |
| --- | --- |
| `scree_plot` | Variance ratio of each principal component |
| `corrFeatures`, `corrFeatures2`, `corrPlot` | Heatmaps of the genes that co-vary most with a gene of interest (`corrPlot` takes a precomputed correlation matrix) |
| `plot_correlation_circle` | Correlation of each gene with the first two dimensions of an embedding |
| `sankey` | Sankey diagram between two labelings of the same cells |
| `pal_max` | A 268-color categorical palette, as a comma-separated string |

## Working with large H5AD files

The `sct.io` helpers read and write H5AD files piece by piece with h5py. You can
edit the annotations of a file, or subset it, without loading the expression
matrix.

```python
import SCTools as sct

# Load everything except X and raw/X
meta = sct.io.read_everything_but_X("atlas.h5ad")

# Save new annotations together with the original expression matrix
is_microglia = (meta.obs["subclass"] == "Micro").to_numpy()
meta.obs["is_microglia"] = is_microglia
sct.io.write_h5ad_with_new_annotation("atlas.h5ad", meta, "atlas_annotated.h5ad")

# Write only the microglia to a new file; X and raw/X are copied in chunks
sct.io.ondisk_subset("atlas.h5ad", "microglia.h5ad", subset_obs=is_microglia, raw=True)
```

- `ondisk_subset` takes a boolean mask over cells for `subset_obs`, and requires
  `X` to be stored as CSR. To convert a CSC file, use `csc2csr_on_disk`. It
  loads the whole matrix, so it needs enough memory to hold it.
- `write_h5ad_with_new_annotation` copies `X` unchanged. The AnnData you pass
  must have the same cells and genes, in the same order, as the original file.

## Outlier detection

`sct.tl.detect_outliers` flags cells that sit far from the rest of their
annotation group in an embedding, such as small islands on a UMAP. Each group is
modeled separately. Groups with up to 300,000 cells use DBSCAN. Larger groups
use LocalOutlierFactor, because DBSCAN's radius-neighbor graph would be too
large. You can change this cutoff with `large_group_threshold`. Groups with
fewer than 100 cells (`min_group_size`) are skipped.

```python
summary = sct.tl.detect_outliers(
    adata,
    groupby="subclass",
    use_rep="X_umap",
    force_outlier={"subclass": ["EN_NF"], "subtype": ["Immune_PVM"]},
)

adata = adata[adata.obs["outlier"] == 0]  # remove outliers
```

`force_outlier` and `force_inlier` override the models. The dict above marks
every cell whose `subclass` is `EN_NF` *or* whose `subtype` is `Immune_PVM` as
an outlier. You can also pass a boolean mask, or a function that takes
`adata.obs` and returns one.

| Output | Contents |
| --- | --- |
| `adata.obs["outlier"]` | 1 for outliers, 0 otherwise |
| `adata.obs["outlier_method"]` | How each cell was labeled, such as `dbscan`, `lof`, `skipped_small_group`, or `forced_outlier` |
| `adata.obs["outlier_score"]` | Model score; higher means more outlying |
| `adata.obs["outlier_dbscan_label"]` | DBSCAN cluster label; -1 means noise |
| `adata.uns["outlier_detection"]` | Parameters and the per-group summary |
| Return value | Per-group method, parameter, and outlier counts |

Only `adata.obs["outlier"]` is written when `store_details=False`.

**Choosing an embedding.** UMAP is good at catching visually obvious islands,
but its distances are not comparable across the map. For production filtering,
prefer a biologically meaningful integrated or PCA representation.
`dbscan_eps` is a distance in the units of `use_rep`, so revisit it when you
change the representation.

## Lineage-mixing detection (heterotypic doublets and contamination)

`sct.tl.detect_lineage_mixing` flags cells whose counts mix cell types from
different lineages, for example a glial nucleus carrying neuronal transcripts.
Simulation-based doublet scores can miss these cells when they are common
enough to form their own clusters, and then they look like distinct subtypes.

**How it works.** Each cell's raw counts over a panel of marker genes are
modeled as a mixture of fixed reference profiles, one per `groupby` label, and
the mixture weights are fitted by expectation-maximization (EM). The foreign
score, `mixing_UMI_frac`, is the fraction of the cell's marker-panel UMIs
assigned to profiles outside its own lineage:

- Clean cells score near 0.
- A doublet of two cells with similar panel depth scores around 0.5.
- Mixtures dominated by the foreign cell type score higher.

The reference profiles are refined on the purest cells of each group, so a
contaminated minority does not leak into them. The function needs only the base
install.

```python
import anndata as ad
import SCTools as sct

adata = ad.read_h5ad("data.h5ad", backed="r")   # counts are streamed from raw.X

summary = sct.tl.detect_lineage_mixing(
    adata,
    groupby="subclass",                     # mixture components
    lineage_key="class",                    # mixing within a lineage is not foreign
    known_lineage_sets=[("OPC", "Oligo")],  # optional: allow OPC -> oligodendrocyte transitions
    library_key="library_id",               # optional: sequencing library / 10x channel
    summary_by="subtype",                   # optional: summarize at a finer level
    threshold=0.3,
)

adata = adata[adata.obs["mixing_flag"] == 0]  # remove flagged cells
```

| Output | Contents |
| --- | --- |
| `adata.obs["mixing_UMI_frac"]` | Foreign score: fraction of marker-panel UMIs from other lineages |
| `adata.obs["mixing_RNA_frac"]` | The same, as an estimated fraction of total RNA |
| `adata.obs["mixing_top"]` | Foreign cell type with the largest weight |
| `adata.obs["mixing_top_weight"]` | That cell type's weight |
| `adata.obs["mixing_flag"]` | 1 if `mixing_UMI_frac >= threshold`, else 0 |
| `adata.obs["mixing_source"]` | For flagged cells: `"doublet"` or `"ambient"` (only with `library_key`) |
| `adata.obsm["mixing_weights"]` | Per-cell weights over all groups |
| `adata.uns["mixing"]` | Parameters, panel genes, reference profiles, and summaries |
| Return value | Per-group flagged fraction and most common foreign type, grouped by `summary_by` |

### Choosing the parameters

**`groupby`.** Use a coarse level such as subclass. If a contaminated
population is its own group, its profile absorbs the contamination and its
cells cannot be detected.

**`lineage_key`.** Assigns each `groupby` label to a broader lineage. Weight on
other groups of the same lineage, such as two excitatory neuron subclasses,
does not count as foreign. If omitted, every group is its own lineage.

**`known_lineage_sets`.** Treats the listed lineages as one. Use it when real
cells sit between two lineages, such as differentiating oligodendrocyte
precursors, so they are not flagged. Mixtures across the merged lineages are
then no longer detected. Labels may be lineage or group labels; unknown labels
raise an error.

**`library_key`.** The sequencing library, such as the 10x channel, of each
cell. It does not change the score or the flag. It adds `mixing_source`, which
says where a flagged cell's foreign signal most likely comes from:

- `"doublet"`: one specific foreign cell type explains the signal best, as for
  a heterotypic doublet or material attached from one other cell.
- `"ambient"`: the library's pooled profile, a proxy for ambient RNA, explains
  it best.

To decide, each flagged cell is refitted twice: once as its own lineage plus
the single best foreign profile, and once as its own lineage plus the library's
pooled profile. Pass the library rather than a coarser processing batch,
because ambient RNA is shared within a library.

**`threshold`.** The score distribution is usually bimodal, with clean cells
near 0 and mixtures well above. Check a histogram of `mixing_UMI_frac` per
group before relying on the default of 0.3.

### Notes

- Requires raw counts: `adata.raw.X` when present, otherwise `adata.X`, or a
  layer given with `layer=`. Log-normalized input raises an error.
- Works on backed AnnData. If you pass a view such as `adata[mask]`, only those
  cells are scored, and the results are written into the parent object.
- Review `summary` at a finer level such as subtype. A subcluster with a
  consistent foreign type and a high flagged fraction is often better removed
  as a whole than cell by cell.

## PLS: supervised disease axis

`sct.tl.pls` and `sct.tl.pls_score` use partial least squares (PLS) to find the
direction in gene expression space that best tracks a phenotype, such as Braak
stage. They complement the standard HVG → PCA → UMAP workflow. They do not
replace PCA.

### How PLS differs from PCA

- **PCA is unsupervised.** It finds the directions of greatest variance in the
  expression matrix `X`. In scRNA-seq the leading PCs are dominated by cell
  type, cell cycle, library size, and batch. Subtle disease effects often end
  up in PCs 20–50, or spread across many components.
- **PLS is supervised.** It finds the directions in `X` with the greatest
  covariance with a response `y`, so the first few components are the
  directions most aligned with `y`. That is what you want when looking for a
  disease axis, and not what you want for an unbiased view of cellular
  heterogeneity.

The intended workflow is to cluster and annotate cell types on a PCA embedding,
then run PLS **within one cell type** to find its disease axis. Fitting across
cell types lets the PLS direction pick up cell-type × disease interactions,
which are hard to interpret.

### `sct.tl.pls`: supervised embedding

Fits PLS on all cells and stores an embedding for visualization or
neighbor-graph construction, alongside PCA.

```python
import SCTools as sct

sct.tl.pls(
    adata,
    y="Braak",                       # numeric phenotype in adata.obs
    n_components=50,
    features="highly_variable_features",
)
```

| Output | Contents |
| --- | --- |
| `adata.obsm["X_pls"]` | Cell embedding, `(n_cells, n_components)` |
| `adata.uns["PLS_x_loadings"]` | Gene loadings |
| `adata.uns["PLS_x_weights"]` | Gene weights; use these to project new data |

- `y` must be numeric and finite in every cell. Convert categorical labels to a
  numeric score first.
- The highly variable gene matrix in `adata.uns[<features>]` is read but not
  modified.
- The fit is **in-sample**, so the scores are not a fair test of disease
  separation on the same cells. Use `sct.tl.pls_score` for that.

### `sct.tl.pls_score`: per-cell disease score

Gives each cell a score from 0 to 1, for display on a UMAP or for trajectory
and niche analyses. It guards against the main ways naive PLS misleads:

- **Donor-grouped cross-validation.** Each cell's score comes from a model
  fitted without that cell's donor. Otherwise PLS tends to learn to tell donors
  apart, through ambient RNA, post-mortem interval, dissociation, sex, or age,
  instead of learning a disease axis.
- **Optional covariate removal.** Covariates such as `pct_mito`, `log_counts`,
  `sex`, and `age` are regressed out within each training fold, so nothing
  leaks from the held-out donors.
- **A readable scale.** Scores are oriented so that a higher score means a
  higher `y`, then rank-transformed to [0, 1], like a pseudotime.
- **Diagnostics.** Returned as a dict (see below).

Run it on one cell type at a time:

```python
mg = adata[adata.obs["celltype"] == "Microglia"].copy()

diag = sct.tl.pls_score(
    mg,
    y="Braak",                     # donor-level quantitative phenotype
    donor_key="donor_id",          # grouping variable for cross-validation
    covariates=["pct_mito", "log_counts", "sex", "age"],
    n_components=2,
    n_splits=5,
    n_permutations=200,
)

import scanpy as sc
sc.pl.umap(mg, color="pls_disease_score", cmap="magma")
```

| Output | Contents |
| --- | --- |
| `mg.obs["pls_disease_score"]` | Per-cell score in [0, 1]; NaN for cells with no finite `y` or no donor |
| `mg.obsm["X_pls_oof"]` | Out-of-fold PLS components |
| `mg.uns["PLS_x_loadings_full"]` | Gene loadings from a fit on all cells |
| `mg.uns["PLS_x_weights_full"]` | Gene weights from a fit on all cells |

The returned diagnostics:

| Key | Meaning |
| --- | --- |
| `pearson_r_pls1_y` | Correlation between the out-of-fold PLS1 score and `y` |
| `between_donor_variance_fraction` | Share of PLS1 variance that lies between donors. Above about 0.8, PLS1 mostly separates donors rather than disease states, and a warning is logged. |
| `within_donor_variance_mean` | Mean PLS1 variance within a donor |
| `permutation_p` | Permutation p-value for `pearson_r_pls1_y`, shuffling `y` across donors; `None` when no permutations were run |
| `n_cells_scored`, `n_donors`, `n_components`, `n_splits` | Run summary |

### Caveats

- **Labels belong to donors, not cells.** In case/control designs every cell
  from a donor shares that donor's label. With few donors, the first PLS axis
  can largely separate donors. Check `between_donor_variance_fraction`.
- **In-sample separation is circular.** Separation between cases and controls
  along `X_pls` from `sct.tl.pls` is partly built in. Quantify separation with
  the out-of-fold scores from `sct.tl.pls_score`.
- **PLS is linear.** Thresholded or combinatorial disease effects may show up
  only as a linear shadow. Non-linear supervised models, such as MRVI or
  contrastive VAEs, are alternatives when that matters.
- **The result is a hypothesis.** Validate the resulting gene signature on
  independent data, such as other cohorts, bulk RNA-seq, or spatial data,
  before drawing biological conclusions.

## Changes from the original script

SCTools began as a standalone script. Packaging it fixed these issues:

- `import SCTools` no longer fails when Pegasus, Scanpy, plotting libraries, or
  scikit-learn are missing.
- `csc2csr_on_disk` keeps the annotations instead of overwriting the output
  file when it writes `X`.
- `ondisk_subset` rebuilds chunked CSR matrices with the correct shape before
  subsetting genes.
- `write_h5ad_with_new_annotation` keeps the original `raw` group when asked to.
- `scanpy_hvf` accepts a `gene_type` column when there is no boolean
  `protein_coding` column.
- The correlation helpers cap `sample_size` at the number of cells instead of
  raising an error on small datasets.
- `sankey` no longer crashes when an input has only one label.
- `mark_MitoCarta` takes the reference file as `mitocarta_path=` instead of
  depending on one hard-coded file path. Without the argument it still falls
  back to that path, and raises `FileNotFoundError` if the file is not there.

## Development

Install the package in editable mode and run the tests:

```bash
pip install -e .
python -m unittest discover -s tests
```

Some tests are skipped unless scikit-learn and Scanpy are installed.
