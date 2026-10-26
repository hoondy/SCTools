# SCTools

`SCTools` packages a collection of Pegasus/Scanpy helper functions into an installable Python library.

The project groups helpers into the following namespaces:

- `sct.io` for AnnData/H5AD read-write helpers
- `sct.pl` for plotting helpers
- `sct.pp` for preprocessing helpers
- `sct.tl` for analysis and tool helpers

The project keeps lightweight AnnData/H5AD utilities available from a simple import:

```python
import SCTools as sct

adata = sct.io.read_everything_but_X("/path/to/file.h5ad")
```

## Installation

Base install for the on-disk AnnData helpers:

```bash
pip install git+https://github.com/hoondy/SCTools
```

Some functions depend on optional scientific Python packages that are intentionally not imported at module import time. If you use Pegasus/Scanpy plotting, pseudobulk, or sklearn-based helpers, also install:

```bash
pip install pegasuspy scanpy scikit-learn seaborn matplotlib adjustText numpy-groupies
```

## What is included

`SCTools` currently bundles:

- `sct.io`: `read_everything_but_X`, `csc2csr_on_disk`, `concat_on_disk`, `ondisk_subset`, `write_h5ad_with_new_annotation`, `proc_h5ad_v2`, `proc_h5ad_v3`, `proc_manifest`, and `save`
- `sct.pp`: preprocessing helpers such as HVG selection, PCA, QC boundaries, signature scores, and category cleanup
- `sct.tl`: aggregation, similarity, marker, PLS, outlier detection, lineage-mixing (doublet/contamination) detection, and pseudometacell helpers
- `sct.pl`: scree plots, correlation plots, palettes, and Sankey diagrams

Functions are intentionally namespaced. Use `sct.io.read_everything_but_X(...)` rather than `sct.read_everything_but_X(...)`.

## Audit and fixes applied

This packaging pass also fixed several portability and correctness issues from the original script:

- `import SCTools` no longer hard-fails just because Pegasus, Scanpy, plotting libraries, or scikit-learn are missing
- `csc2csr_on_disk` now preserves annotations instead of overwriting the output file when writing `X`
- `ondisk_subset` now reconstructs chunked CSR matrices with the correct original shape before variable subsetting
- `write_h5ad_with_new_annotation` now preserves the original `raw` group when requested
- `scanpy_hvf` no longer assumes a `protein_coding` boolean column exists if `gene_type` is the available annotation
- Correlation helpers now cap `sample_size` to the number of observations instead of raising on small datasets
- `sankey` now handles single-label inputs without referencing an undefined `topEdge`
- `mark_MitoCarta` now accepts an explicit reference path instead of depending on one hard-coded filesystem location

## PLS: supervised embedding and per-cell disease score

`sct.tl.pls` and `sct.tl.pls_score` provide a **complementary supervised
embedding** to the standard HVG → PCA → UMAP workflow. They are not a drop-in
replacement for PCA.

### How PLS differs from PCA

- **PCA** finds directions that maximize variance in the gene expression
  matrix `X` — it is fully unsupervised. In scRNA-seq the leading PCs are
  dominated by whatever drives the most variance: cell type identity, cell
  cycle, library size, batch. Subtle disease effects often live in PCs 20–50
  or are smeared across many components.
- **PLS** finds directions in `X` that maximize **covariance** with a response
  `Y` — it is supervised. By construction the first few PLS components are the
  directions in gene space most aligned with `Y`. That is exactly what you want
  when the goal is to see the disease axis, and exactly what you don't want if
  the goal is an unbiased representation of cellular heterogeneity.

Cluster and annotate cell types on a PCA embedding, then apply PLS **within a
cell type** to recover the disease axis. Treat PLS output as a focused
analysis tool, not as the general embedding used for clustering.

### `sct.tl.pls` — supervised embedding

Fits PLS on the full dataset and stores a per-cell embedding for downstream
visualization / neighbor-graph construction alongside (not instead of) PCA.

```python
import SCTools as sct

sct.tl.pls(
    adata,
    y="Braak",                       # numeric phenotype in adata.obs
    n_components=50,
    features="highly_variable_features",
)
# adata.obsm["X_pls"]          -> (n_cells, n_components) embedding
# adata.uns["PLS_x_loadings"]  -> gene loadings from the fit
# adata.uns["PLS_x_weights"]   -> gene weights (use these to project new data)
```

Notes:

- `y` must be numeric and finite in every cell; categorical labels should be
  coerced to a numeric score first.
- The HVG matrix in `adata.uns[<features>]` is read but not modified.
- This is an **in-sample** fit. Do not use the resulting scores as a test of
  disease separation on the same cells — use `sct.tl.pls_score` for that.

### `sct.tl.pls_score` — per-cell disease pseudotime

Produces a per-cell score in `[0, 1]` suitable for visualization on UMAP and
for downstream trajectory / niche analyses. Addresses the main failure modes
of naive PLS:

- **Donor-grouped cross-validation.** Each cell's score is an *out-of-fold*
  PLS score — the model that produced it was fit on donors that did not
  include that cell's donor. Without this, PLS preferentially learns a donor
  discriminator (ambient RNA, PMI, dissociation, sex, age), not a disease
  axis.
- **Optional covariate residualization** (e.g. `pct_mito`, `log_counts`,
  `sex`, `age`) fit per training fold to avoid leakage.
- **Sign-oriented** so higher score means higher `y`, then **rank-transformed
  to [0, 1]** to give a proper pseudotime-like readout.
- **Diagnostics** printed and returned: Pearson r(PLS1, y), between-donor
  variance fraction (warns above 0.8 — your PLS1 is then mostly a donor
  discriminator), and an optional permutation-null p-value.

Typical usage, one cell type at a time:

```python
mg = adata[adata.obs["celltype"] == "Microglia"].copy()

diag = sct.tl.pls_score(
    mg,
    y="Braak",                     # donor-level quantitative phenotype
    donor_key="donor_id",          # grouping variable for CV
    covariates=["pct_mito", "log_counts", "sex", "age"],
    n_components=2,
    n_splits=5,
    n_permutations=200,
)

# mg.obs["pls_disease_score"]       -> per-cell score in [0, 1]
# mg.obsm["X_pls_oof"]              -> out-of-fold PLS components
# mg.uns["PLS_x_loadings_full"]     -> loadings from full-data fit
# mg.uns["PLS_x_weights_full"]      -> weights from full-data fit

import scanpy as sc
sc.pl.umap(mg, color="pls_disease_score", cmap="magma")
```

### Caveats worth repeating

- PLS is supervised. Any reported separation between cases and controls along
  `X_pls` computed in-sample is partly circular. Use `sct.tl.pls_score` and
  its out-of-fold scores when you need to quantify separation.
- In typical case/control scRNA-seq designs the label is attached to the
  **donor**, not the cell. With few donors, the first PLS axis can be largely
  a donor-batch axis. Inspect the `between_donor_variance_fraction`
  diagnostic from `pls_score` — values above ~0.8 indicate a donor
  discriminator rather than a disease axis.
- Apply PLS **within a single cell type** after annotation. Fitting across
  cell types invites the PLS direction to capture cell-type × disease
  interactions that are hard to interpret.
- PLS is linear. Thresholded / combinatorial disease effects may only appear
  as a linear shadow. Non-linear supervised models (MRVI, contrastive VAEs)
  are alternatives when that matters.
- Treat the PLS axis as a **hypothesis generator**. Validate the resulting
  gene signature on independent data (other cohorts, bulk RNA-seq, spatial)
  before drawing biological conclusions.

## Embedding outlier detection

`sct.tl.detect_outliers` detects extreme cells within annotation groups and
writes `adata.obs["outlier"]` as a 0/1 label, where 1 means outlier. It uses
DBSCAN by default and switches to LocalOutlierFactor for very large groups to
avoid constructing a large radius-neighbor graph.

```python
summary = sct.tl.detect_outliers(
    adata,
    groupby="subclass",
    use_rep="X_umap",
    force_outlier={"subclass": ["EN_NF"], "subtype": ["Immune_PVM"]},
)

# adata.obs["outlier"] -> 0/1 outlier label
# summary              -> per-group method, parameter, and removal counts
```

For production filtering, prefer a biologically meaningful integrated/PCA
representation when available; UMAP is useful for catching visually obvious
embedding islands, but its local distances are not calibrated across the map.

## Lineage-mixing detection (heterotypic doublets and contamination)

`sct.tl.detect_lineage_mixing` flags cells whose counts are a mixture of cell
types from different lineages, for example a glial nucleus carrying neuronal
transcripts. Such cells can pass simulation-based doublet scores when they are
common enough to form their own clusters, and they then look like distinct
subtypes.

Each cell's raw counts over a marker panel are modelled as a multinomial
mixture of fixed reference profiles (one per `groupby` label), fitted per cell
by EM. The foreign score is the fraction of the cell's marker-panel UMIs
attributed to profiles outside its own lineage. Clean cells score near 0, and
a doublet of two cells with similar panel depth scores around 0.5. Mixtures
dominated by the foreign cell type score higher. References are refined on the
purest cells of each group, so a contaminated minority does not leak into them.

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
    threshold=0.4,
)

# adata.obs["mixing_UMI_frac"]    -> fraction of marker-panel UMIs from other lineages
# adata.obs["mixing_RNA_frac"]    -> the same as an estimated fraction of total RNA
# adata.obs["mixing_top"]         -> foreign cell type with the largest weight
# adata.obs["mixing_top_weight"]  -> that cell type's weight
# adata.obs["mixing_flag"]        -> 1 if mixing_UMI_frac >= threshold, else 0
# adata.obs["mixing_source"]      -> for flagged cells: "doublet" or "ambient"
# adata.obsm["mixing_weights"]    -> per-cell weights over all groups
# summary                         -> per-group flagged fraction and top foreign type

adata = adata[adata.obs["mixing_flag"] == 0]  # remove flagged cells
```

Parameters that need a decision:

- `lineage_key` assigns each `groupby` label to a broader lineage. Weight on
  other groups of the same lineage (e.g. two excitatory neuron subclasses) is
  not counted as foreign. If omitted, every group is its own lineage.
- `known_lineage_sets` treats the listed lineages as one. Use it when real cells
  sit between two lineages, such as differentiating oligodendrocyte precursors,
  so they are not flagged. Mixtures across merged lineages are then no longer
  detected. Labels may be lineage or group labels; unknown labels raise an
  error.
- `library_key` identifies the sequencing library (e.g. 10x channel) of each
  cell. It is optional and does not change the score or the flag. It enables
  `mixing_source`: each flagged cell is refitted as "own lineage + the single
  best foreign profile" and as "own lineage + the library's pooled profile" (a
  proxy for ambient RNA). `"doublet"` means one specific foreign cell type
  explains the signal better, as for a heterotypic doublet or attached material
  from one cell. `"ambient"` means a pooled, soup-like profile explains it
  better. Pass the library, not a coarser processing batch, because ambient RNA
  is shared within a library.

Notes:

- Requires raw counts (`adata.raw.X` by default, or `layer=`); it raises on
  log-normalized input.
- Use a coarse `groupby` such as subclass. If a contaminated population is its
  own group, its profile absorbs the contamination and it cannot be detected.
- The score distribution is usually bimodal, with clean cells near 0 and
  mixtures well above. Check a histogram of `mixing_UMI_frac` per group
  before relying on the default `threshold=0.4`.
- Review `summary` at a finer level such as subtype. A subcluster with a
  consistent foreign type and a high flagged fraction is often better removed
  as a whole than cell by cell.

## Notes

- `read_everything_but_X` is designed for annotation-only access. It reconstructs an `AnnData` object from everything except `X` and `raw/X`.
- `ondisk_subset` expects `X` to be stored in CSR format. Use `csc2csr_on_disk` first if your file is backed by CSC.
- `mark_MitoCarta` now prefers an explicit `mitocarta_path=` argument unless the original lab-specific reference path exists on disk.

## Development

Run the lightweight regression tests for the core H5AD helpers with:

```bash
python -m unittest discover -s tests
```
