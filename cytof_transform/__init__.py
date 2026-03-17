"""
cytof_transform
===============
CyTOF-transform: sctransform-like normalization for CyTOF / mass-cytometry data.

Quick start
-----------
>>> import pandas as pd
>>> from cytof_transform import (
...     CytofTransformConfig,
...     cytof_transform_global,
...     plot_tech_factor_qc,
...     plot_marker_correlations_qc,
... )
>>> config = CytofTransformConfig(
...     control_markers=["H3", "H4", "DNA1", "DNA2"],
...     markers_to_correct=["CD3", "CD4", "CD8", "CD20"],
... )
>>> result = cytof_transform_global(asinh_data, config)
"""

from .core import (
    # Data classes
    CytofTransformConfig,
    CytofTransformResult,
    # Main normalisation functions
    cytof_transform_global,
    cytof_transform_by_compartment,
    # Utility / diagnostic
    compute_marker_tech_correlations,
    evaluate_marker_intensity_regime,
    # QC plots
    plot_tech_factor_qc,
    plot_marker_correlations_qc,
    plot_gamma_qc,
    plot_umap_qc,
)

__all__ = [
    "CytofTransformConfig",
    "CytofTransformResult",
    "cytof_transform_global",
    "cytof_transform_by_compartment",
    "compute_marker_tech_correlations",
    "evaluate_marker_intensity_regime",
    "plot_tech_factor_qc",
    "plot_marker_correlations_qc",
    "plot_gamma_qc",
    "plot_umap_qc",
]

__version__ = "0.1.0"
