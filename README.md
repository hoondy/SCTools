# SCTools

`SCTools` packages a collection of Pegasus/Scanpy helper functions from `pge.py` into an installable Python library.

The project now groups helpers into the following namespaces:

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
- `sct.tl`: aggregation, similarity, marker, PLS, and pseudometacell helpers
- `sct.pl`: scree plots, correlation plots, palettes, and Sankey diagrams

For backward compatibility, the original top-level function names are still available too.

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

## Notes

- `read_everything_but_X` is designed for annotation-only access. It reconstructs an `AnnData` object from everything except `X` and `raw/X`.
- `ondisk_subset` expects `X` to be stored in CSR format. Use `csc2csr_on_disk` first if your file is backed by CSC.
- `mark_MitoCarta` now prefers an explicit `mitocarta_path=` argument unless the original lab-specific reference path exists on disk.

## Development

Run the lightweight regression tests for the core H5AD helpers with:

```bash
python -m unittest discover -s tests
```
