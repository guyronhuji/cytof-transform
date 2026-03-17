# cytof-transform

**CyTOF-transform** is a Python package for normalising mass-cytometry (CyTOF) data by removing per-cell technical variation while preserving biological signal — analogous to `sctransform` for single-cell RNA-seq.

---

## The problem

In CyTOF, each cell's marker intensities are affected by a cell-specific *technical factor*: cell size, permeabilisation efficiency, and other non-biological sources of variation.  This inflates variance and biases downstream clustering and dimensionality reduction.

## The model

CyTOF-transform assumes a multiplicative model:

```
X_{i,m} ≈ Biology_{i,m} × T_i^{γ_m} × noise
```

In arcsinh / log space this becomes additive:

```
y_{i,m} ≈ α_m + γ_m × f_i + biology_{i,m}
```

where:

- `f_i` is a 1-D *technical factor* estimated as PC1 of core-histone and DNA control markers.
- `γ_m` is the marker-specific sensitivity to the technical factor, estimated by OLS regression.

Correction subtracts `γ_m × (f_i − median(f))` from each cell, anchoring at the *median* cell so absolute intensities remain interpretable.

---

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/YOUR_USERNAME/cytof-transform.git
```

### With UMAP support

```bash
pip install "git+https://github.com/YOUR_USERNAME/cytof-transform.git#egg=cytof-transform[umap]"
```

### For development

```bash
git clone https://github.com/YOUR_USERNAME/cytof-transform.git
cd cytof-transform
pip install -e ".[dev]"
```

---

## Quick start

```python
import pandas as pd
import numpy as np
from cytof_transform import (
    CytofTransformConfig,
    cytof_transform_global,
    plot_tech_factor_qc,
    plot_marker_correlations_qc,
    plot_gamma_qc,
)

# asinh_data: cells × markers DataFrame (already arcsinh-transformed)
# Columns include control markers (histones/DNA) and biological markers.

config = CytofTransformConfig(
    control_markers=["H3", "H4", "H2A", "H2B", "DNA1", "DNA2"],
    markers_to_correct=["CD3", "CD4", "CD8a", "CD20", "CD45", "Ki67"],
    anchor_to_median=True,
    zscore=True,
)

result = cytof_transform_global(asinh_data, config)

# result.corrected    — corrected arcsinh values
# result.residuals_z  — z-scored corrected values (use for PCA / UMAP)
# result.tech_factor  — per-cell technical factor (PC1)
# result.gamma        — per-marker γ slopes
```

---

## API reference

### Config

| Class | Description |
|---|---|
| `CytofTransformConfig` | All configuration options |

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `control_markers` | required | Core histones + DNA markers used to define the technical factor |
| `markers_to_correct` | required | Markers to normalise |
| `anchor_to_median` | `True` | Keep the median cell unchanged |
| `zscore` | `True` | Z-score corrected values |
| `line_col` | `None` | Column name identifying batch / cell-line for balanced γ estimation |

### Normalisation functions

| Function | Description |
|---|---|
| `cytof_transform_global(asinh_data, config)` | Single-pass correction over all cells |
| `cytof_transform_by_compartment(asinh_data, compartments, config)` | Per-compartment correction (immune / tumour / stroma etc.) |

### Diagnostics / utilities

| Function | Description |
|---|---|
| `compute_marker_tech_correlations(data, ...)` | Pearson correlations of all markers with the technical factor |
| `evaluate_marker_intensity_regime(asinh_data, markers)` | Flag markers too close to zero in arcsinh space |

### QC plots

| Function | Description |
|---|---|
| `plot_tech_factor_qc(asinh_data, control_markers)` | Histogram, PC1 loadings, scree plot |
| `plot_marker_correlations_qc(pre, post, tech_factor)` | Barplot + scatter of correlations before/after |
| `plot_gamma_qc(gamma)` | Barplot of per-marker γ values |
| `plot_umap_qc(pre, post, tech_factor, ...)` | 2×2 UMAP panel (requires `umap-learn`) |

---

## Multi-batch / multi-line data

If your data spans multiple samples or cell lines you should set `line_col` in the config.  This triggers *balanced sampling*: γ is estimated on a subset with equal cells per group, preventing the largest batch from dominating the slope estimate.

```python
config = CytofTransformConfig(
    control_markers=[...],
    markers_to_correct=[...],
    line_col="sample_id",   # column in asinh_data with batch labels
)
```

---

## Compartment-aware normalisation

When the technical factor differs systematically between cell types (e.g., immune vs. tumour cells have different nuclear content), run the correction separately per compartment:

```python
from cytof_transform import cytof_transform_by_compartment

compartments = cell_metadata["lineage"]   # Series with same index as asinh_data

result = cytof_transform_by_compartment(
    asinh_data=asinh_data,
    compartments=compartments,
    config=config,
)
```

---

## Example notebook

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a full worked example with synthetic data, QC plots, and UMAP visualisation.

---

## Dependencies

| Package | Version |
|---|---|
| numpy | ≥ 1.23 |
| pandas | ≥ 1.5 |
| scikit-learn | ≥ 1.2 |
| matplotlib | ≥ 3.6 |
| seaborn | ≥ 0.12 |
| umap-learn | ≥ 0.5 *(optional, for `plot_umap_qc`)* |

---

## Citation

If you use cytof-transform in your research, please cite this repository.

---

## License

MIT — see [LICENSE](LICENSE).
