"""
core.py — CyTOF-transform implementation.

CyTOF-transform: sctransform-like normalization for CyTOF data.

Core ideas
----------
- Use core histones + DNA as control markers to estimate a per-cell
  technical factor T (via PC1 on arcsinh-transformed data).
- Assume intracellular markers follow a multiplicative model::

      X_{i,m} ≈ Biology_{i,m} * T_i^{γ_m} * noise

  which in arcsinh/log space becomes additive::

      y_{i,m} ≈ α_m + γ_m * f_i + BiologyResidual_{i,m}

  where f_i is a 1D technical factor (PC1).
- Fit γ_m by linear regression of y_m on f.
- Correct by subtracting γ_m * (f - median(f)), i.e. regress out
  the technical factor but anchor at a median "reference" cell.
- Optionally z-score the corrected data for downstream PCA/UMAP.

Supports
--------
- Global transform (all cells together)
- Compartment-aware transform (per lineage / compartment)
- Balanced-line sampling to prevent large batches from biasing γ estimates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CytofTransformResult:
    """Container for CyTOF-transform outputs.

    Attributes
    ----------
    corrected : DataFrame
        Arcsinh-corrected marker intensities, same shape as the input.
    residuals_z : DataFrame
        Z-scored corrected values (useful for PCA / UMAP).
    tech_factor : Series
        1-D technical factor (PC1) per cell.
    gamma : dict
        Per-marker γ — the regression slope against the technical factor.
    alpha : dict
        Per-marker α — regression intercept (diagnostic use).
    pca_model : PCA or None
        Fitted sklearn PCA object used to compute the technical factor.
    """
    corrected: pd.DataFrame
    residuals_z: pd.DataFrame
    tech_factor: pd.Series
    gamma: Dict[str, float]
    alpha: Dict[str, float]
    pca_model: Optional[PCA] = None


@dataclass
class CytofTransformConfig:
    """Configuration for CyTOF-transform.

    Parameters
    ----------
    control_markers : list of str
        Core histones and/or DNA markers used to define the per-cell
        technical factor (e.g. ``["H3", "H4", "DNA1", "DNA2"]``).
    markers_to_correct : list of str
        Markers whose dependence on the technical factor will be removed.
    use_compartments : bool, default False
        If True, ``cytof_transform_global`` will raise; use
        ``cytof_transform_by_compartment`` instead.
    n_pcs_for_T : int, default 1
        Number of PCs to compute. Only PC1 is used as the technical
        factor; keep this at 1 unless you want extra PCs stored in
        the PCA model for inspection.
    anchor_to_median : bool, default True
        If True, corrections are anchored at ``median(f)`` so that the
        "median cell" is unchanged.  If False, anchored at 0.
    zscore : bool, default True
        If True, compute z-scored residuals (``residuals_z``).
    line_col : str or None, default None
        Column name in ``asinh_data`` that identifies the batch / cell
        line. When set, γ is estimated on a balanced subset
        (equal cells per group) to prevent dominant batches from
        biasing the slope estimates.
    """
    control_markers: Sequence[str]
    markers_to_correct: Sequence[str]
    use_compartments: bool = False
    n_pcs_for_T: int = 1
    anchor_to_median: bool = True
    zscore: bool = True
    line_col: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_tech_factor_pc1(
    asinh_data: pd.DataFrame,
    control_markers: Sequence[str],
    n_pcs: int = 1,
    label: Optional[str] = None,
) -> Tuple[pd.Series, PCA]:
    """Compute a PC1-based technical factor from control markers.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    control_markers : list of str
        Names of control markers (core histones + DNA).
    n_pcs : int, default 1
        Number of PCs to compute; only PC1 is used as the technical factor.
    label : str, optional
        Label used in log messages (e.g. compartment name).

    Returns
    -------
    tech_factor : Series
        PC1 scores, index = ``asinh_data.index``.
    pca : PCA
        Fitted sklearn PCA object.
    """
    missing = [m for m in control_markers if m not in asinh_data.columns]
    if missing:
        tag = f" ({label})" if label else ""
        raise ValueError(f"Missing control markers in asinh_data{tag}: {missing}")

    Y = asinh_data[list(control_markers)].copy()
    n_cells = Y.shape[0]
    if n_cells < 2:
        tag = f" for {label}" if label else ""
        raise ValueError(f"Not enough cells ({n_cells}) to compute PCA{tag}.")

    n_pcs_eff = min(n_pcs, Y.shape[1], max(1, n_cells - 1))
    pca = PCA(n_components=n_pcs_eff)
    X_pcs = pca.fit_transform(Y.values)

    tech_factor = pd.Series(X_pcs[:, 0], index=asinh_data.index, name="tech1")

    if label is not None:
        print(
            f"[CyTOF-transform] Computed PC1 technical factor for '{label}' "
            f"(explained var PC1 = {pca.explained_variance_ratio_[0]:.3f})"
        )

    return tech_factor, pca


def _regress_and_correct_1d(
    asinh_data: pd.DataFrame,
    tech_factor: pd.Series,
    markers_to_correct: Sequence[str],
    anchor_to_median: bool = True,
    zscore: bool = True,
    line_labels: Optional[pd.Series] = None,
    max_cells_per_line: Optional[int] = None,
) -> CytofTransformResult:
    """Regress arcsinh intensities on the technical factor and correct.

    For each marker m:
      1. Estimate γ_m via OLS on a (optionally balanced) subset of cells.
      2. Apply ``y_corr = y - γ_m * (f - anchor)`` to all cells.
      3. Optionally z-score the corrected values.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    tech_factor : Series
        1-D technical factor, index matching ``asinh_data``.
    markers_to_correct : list of str
        Markers to correct.
    anchor_to_median : bool
        Anchor correction at ``median(f)`` over all cells.
    zscore : bool
        Z-score the corrected values.
    line_labels : Series or array-like, optional
        Batch / cell-line label per cell.  When provided, γ is estimated
        on a balanced subset (equal cells per group).
    max_cells_per_line : int, optional
        Hard cap on cells taken per group for balanced sampling.
        Defaults to the size of the smallest group.

    Returns
    -------
    CytofTransformResult
    """
    ddf = asinh_data.copy()
    f_all = tech_factor.loc[ddf.index].values

    # ------------------------------------------------------------------
    # Build balanced subset for slope estimation
    # ------------------------------------------------------------------
    if line_labels is not None:
        if isinstance(line_labels, pd.DataFrame):
            if line_labels.shape[1] != 1:
                raise ValueError(
                    "line_labels DataFrame must have exactly one column, "
                    f"got {line_labels.shape[1]}."
                )
            line_labels = line_labels.iloc[:, 0]

        if not isinstance(line_labels, pd.Series):
            line_labels = pd.Series(line_labels, index=ddf.index, name="line")

        if not line_labels.index.equals(ddf.index):
            raise ValueError("line_labels index must match asinh_data.index")

        labels = line_labels.astype("category")
        groups = labels.unique()

        if max_cells_per_line is None:
            target = min((labels == g).sum() for g in groups)
        else:
            target = max_cells_per_line

        balanced_idx: List[int] = []
        for g in groups:
            idx_g = np.where(labels == g)[0]
            chosen = (
                np.random.choice(idx_g, size=target, replace=False)
                if len(idx_g) >= target
                else idx_g
            )
            balanced_idx.extend(chosen.tolist())

        balanced_idx_arr = np.array(balanced_idx)
    else:
        balanced_idx_arr = np.arange(ddf.shape[0])

    # Subset for fitting
    f = f_all[balanced_idx_arr]
    f_centered = f - f.mean()
    denom = np.sum(f_centered ** 2)
    if denom == 0:
        raise ValueError("Technical factor has zero variance in balanced subset.")

    anchor = np.median(f_all) if anchor_to_median else 0.0

    gamma: Dict[str, float] = {}
    alpha: Dict[str, float] = {}

    for m in markers_to_correct:
        if m not in ddf.columns:
            raise ValueError(f"Marker '{m}' not found in asinh_data columns.")

        y_all = ddf[m].values
        y = y_all[balanced_idx_arr]

        y_mean = y.mean()
        y_centered = y - y_mean
        gamma_m = float(np.sum(f_centered * y_centered) / denom)
        alpha_m = float(y_mean - gamma_m * f.mean())

        gamma[m] = gamma_m
        alpha[m] = alpha_m

        # Apply correction to ALL cells
        ddf[m] = y_all - gamma_m * (f_all - anchor)

    if zscore:
        residuals_z = ddf.copy()
        for m in markers_to_correct:
            vals = residuals_z[m].values
            mu, sigma = vals.mean(), vals.std()
            if sigma == 0:
                sigma = 1.0
            residuals_z[m] = (vals - mu) / sigma
    else:
        residuals_z = ddf.copy()

    return CytofTransformResult(
        corrected=ddf,
        residuals_z=residuals_z,
        tech_factor=tech_factor,
        gamma=gamma,
        alpha=alpha,
        pca_model=None,
    )


# ---------------------------------------------------------------------------
# Public API — normalisation
# ---------------------------------------------------------------------------


def cytof_transform_global(
    asinh_data: pd.DataFrame,
    config: CytofTransformConfig,
) -> CytofTransformResult:
    """Run CyTOF-transform on all cells together (no compartments).

    Steps:

    1. Compute PC1 of control markers → technical factor *f*.
    2. For each marker in ``markers_to_correct``, regress *y_m* on *f*.
    3. Subtract ``γ_m * (f - median(f))`` → corrected *y_m*.
    4. Optionally z-score the corrected markers.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.  If ``config.line_col``
        is set the column must be present here (it is used for balanced
        sampling but is not modified).
    config : CytofTransformConfig
        Full configuration object.

    Returns
    -------
    CytofTransformResult
    """
    if config.use_compartments:
        raise ValueError(
            "config.use_compartments=True — use cytof_transform_by_compartment instead."
        )

    tech_factor, pca = _compute_tech_factor_pc1(
        asinh_data=asinh_data,
        control_markers=config.control_markers,
        n_pcs=config.n_pcs_for_T,
        label="global",
    )

    line_labels = asinh_data[config.line_col] if config.line_col is not None else None

    result = _regress_and_correct_1d(
        asinh_data=asinh_data,
        tech_factor=tech_factor,
        markers_to_correct=config.markers_to_correct,
        anchor_to_median=config.anchor_to_median,
        zscore=config.zscore,
        line_labels=line_labels,
    )
    result.pca_model = pca
    return result


def cytof_transform_by_compartment(
    asinh_data: pd.DataFrame,
    compartments: pd.Series,
    config: CytofTransformConfig,
) -> CytofTransformResult:
    """Run CyTOF-transform separately within each compartment.

    Each compartment (e.g. immune / tumour / stroma) gets its own
    technical factor and γ estimates, then results are recombined in
    original cell order.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    compartments : Series
        Compartment label per cell, index matching ``asinh_data``.
    config : CytofTransformConfig
        Configuration object.  ``use_compartments`` is ignored; this
        function is explicitly compartment-aware.

    Returns
    -------
    CytofTransformResult
        ``gamma`` / ``alpha`` are the mean values across compartments
        (for per-compartment values, inspect each sub-result yourself).
    """
    if not asinh_data.index.equals(compartments.index):
        raise ValueError("Index of asinh_data and compartments must match.")

    corrected_list, residuals_list, tech_list = [], [], []
    gamma_all: Dict[str, List[float]] = {m: [] for m in config.markers_to_correct}
    alpha_all: Dict[str, List[float]] = {m: [] for m in config.markers_to_correct}
    last_pca = None

    for comp in compartments.unique():
        idx = compartments == comp
        sub_data = asinh_data.loc[idx]

        print(
            f"[CyTOF-transform] Compartment '{comp}': {sub_data.shape[0]} cells "
            f"({sub_data.shape[1]} markers)"
        )

        tech_factor_c, pca_c = _compute_tech_factor_pc1(
            asinh_data=sub_data,
            control_markers=config.control_markers,
            n_pcs=config.n_pcs_for_T,
            label=str(comp),
        )

        result_c = _regress_and_correct_1d(
            asinh_data=sub_data,
            tech_factor=tech_factor_c,
            markers_to_correct=config.markers_to_correct,
            anchor_to_median=config.anchor_to_median,
            zscore=config.zscore,
        )

        corrected_list.append(result_c.corrected)
        residuals_list.append(result_c.residuals_z)
        tech_list.append(result_c.tech_factor)

        for m in config.markers_to_correct:
            gamma_all[m].append(result_c.gamma[m])
            alpha_all[m].append(result_c.alpha[m])

        last_pca = pca_c

    corrected_all = pd.concat(corrected_list).loc[asinh_data.index]
    residuals_all = pd.concat(residuals_list).loc[asinh_data.index]
    tech_all = pd.concat(tech_list).loc[asinh_data.index]

    return CytofTransformResult(
        corrected=corrected_all,
        residuals_z=residuals_all,
        tech_factor=tech_all,
        gamma={m: float(np.mean(v)) for m, v in gamma_all.items()},
        alpha={m: float(np.mean(v)) for m, v in alpha_all.items()},
        pca_model=last_pca,
    )


# ---------------------------------------------------------------------------
# Public API — diagnostics / utilities
# ---------------------------------------------------------------------------


def compute_marker_tech_correlations(
    data: pd.DataFrame,
    tech_factor: Optional[pd.Series] = None,
    control_markers: Optional[Sequence[str]] = None,
    n_pcs: int = 1,
    tech_name: str = "tech1",
) -> Tuple[pd.Series, pd.Series]:
    """Pearson correlation of every marker with the technical factor.

    Either supply a pre-computed ``tech_factor``, or provide
    ``control_markers`` to let the function compute PC1.

    Parameters
    ----------
    data : DataFrame
        Cells × markers, arcsinh-transformed.
    tech_factor : Series, optional
        Pre-computed technical factor.
    control_markers : list of str, optional
        Used to compute PC1 if ``tech_factor`` is None.
    n_pcs : int, default 1
        Number of PCs when computing PC1.
    tech_name : str, default ``"tech1"``
        Name for the computed Series.

    Returns
    -------
    corr : Series
        Pearson correlations, index = marker names.
    tech_factor : Series
        The technical factor that was used.
    """
    if tech_factor is not None:
        if not tech_factor.index.equals(data.index):
            raise ValueError("Index of tech_factor must match data.index.")
        f = tech_factor.copy()
    else:
        if control_markers is None:
            raise ValueError(
                "Provide either tech_factor or control_markers to compute PC1."
            )
        missing = [m for m in control_markers if m not in data.columns]
        if missing:
            raise ValueError(f"control_markers not found in data: {missing}")

        Y = data[list(control_markers)].copy()
        n_cells = Y.shape[0]
        if n_cells < 2:
            raise ValueError("Not enough cells to compute PCA (need at least 2).")

        n_pcs_eff = min(n_pcs, Y.shape[1], max(1, n_cells - 1))
        pca = PCA(n_components=n_pcs_eff)
        X_pcs = pca.fit_transform(Y.values)
        f = pd.Series(X_pcs[:, 0], index=data.index, name=tech_name)
        print(
            f"[compute_marker_tech_correlations] Computed {tech_name} from PC1. "
            f"Explained var = {pca.explained_variance_ratio_[0]:.3f}"
        )

    corr = data.corrwith(f)
    return corr, f


def evaluate_marker_intensity_regime(
    asinh_data: pd.DataFrame,
    candidate_markers: Sequence[str],
    med_thresh: float = 0.3,
    p90_thresh: float = 0.7,
) -> pd.DataFrame:
    """Flag markers that are mostly in the near-zero (linear) arcsinh regime.

    Markers with very low arcsinh values may not satisfy the log-like
    approximation assumed by CyTOF-transform and should be treated with care.

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    candidate_markers : list of str
        Markers to evaluate.
    med_thresh : float, default 0.3
        If ``median(asinh) < med_thresh`` AND ``p90(asinh) < p90_thresh``
        the marker is flagged as ``too_low``.
    p90_thresh : float, default 0.7
        90th-percentile threshold (see above).

    Returns
    -------
    DataFrame
        Index = marker, columns = ``median``, ``p90``, ``too_low``,
        ``use_for_corr``.
    """
    missing = [m for m in candidate_markers if m not in asinh_data.columns]
    if missing:
        raise ValueError(f"Markers not found in asinh_data: {missing}")

    records = []
    for m in candidate_markers:
        vals = asinh_data[m].values
        med = float(np.median(vals))
        p90 = float(np.quantile(vals, 0.9))
        too_low = (med < med_thresh) and (p90 < p90_thresh)
        records.append(
            {"marker": m, "median": med, "p90": p90, "too_low": too_low, "use_for_corr": not too_low}
        )

    return pd.DataFrame.from_records(records).set_index("marker")


# ---------------------------------------------------------------------------
# Public API — QC plots
# ---------------------------------------------------------------------------


def plot_tech_factor_qc(
    asinh_data: pd.DataFrame,
    control_markers: Sequence[str],
    tech_factor: Optional[pd.Series] = None,
    n_pcs: int = 2,
    tech_name: str = "tech1",
    figsize: tuple = (14, 4),
):
    """QC plot for the technical factor.

    Produces three panels:

    1. Histogram of the technical factor distribution.
    2. PC1 loadings barplot for the control markers.
    3. Explained variance per PC (scree plot).

    Parameters
    ----------
    asinh_data : DataFrame
        Cells × markers, arcsinh-transformed.
    control_markers : list-like
        Control markers used to define the technical factor.
    tech_factor : Series, optional
        Pre-computed technical factor.  If None, PC1 is computed here.
    n_pcs : int, default 2
        Number of PCs shown in the scree plot.
    tech_name : str
        Label used in plot titles.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    tech_factor : Series
    loadings : Series  (PC1 loadings)
    explained : ndarray  (explained variance ratios)
    """
    ctrl = list(control_markers)
    missing = [m for m in ctrl if m not in asinh_data.columns]
    if missing:
        raise ValueError(f"Missing control markers: {missing}")

    Y = asinh_data[ctrl].values
    n_cells = Y.shape[0]
    n_pcs_eff = min(n_pcs, len(ctrl), max(1, n_cells - 1))
    pca = PCA(n_components=n_pcs_eff)
    pca.fit(Y)
    explained = pca.explained_variance_ratio_
    loadings = pd.Series(pca.components_[0], index=ctrl)

    if tech_factor is None:
        X_pcs = pca.transform(Y)
        tech_factor = pd.Series(X_pcs[:, 0], index=asinh_data.index, name=tech_name)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax = axes[0]
    ax.hist(tech_factor.values, bins=50, alpha=1, color="steelblue")
    ax.set_xlabel(tech_name)
    ax.set_ylabel("Cell count")
    ax.set_title(f"{tech_name} distribution")

    ax = axes[1]
    loadings.sort_values(ascending=False).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("PC1 loading")
    ax.set_title("PC1 loadings (control markers)")

    ax = axes[2]
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained, color="steelblue")
    ax.set_xticks(x)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA scree (control markers)")

    plt.tight_layout()
    plt.show()
    return tech_factor, loadings, explained


def plot_marker_correlations_qc(
    asinh_pre: pd.DataFrame,
    asinh_post: pd.DataFrame,
    tech_factor: pd.Series,
    markers_to_highlight: Optional[Sequence[str]] = None,
    top_n: int = 25,
    figsize: tuple = (20, 5),
) -> pd.DataFrame:
    """Compare marker–technical-factor correlations before and after correction.

    Produces two panels:

    1. Barplot (selected / top-N markers) — correlation pre vs. post.
    2. Scatter of all markers: correlation pre (x) vs. post (y).

    Parameters
    ----------
    asinh_pre : DataFrame
        Data *before* normalisation.
    asinh_post : DataFrame
        Data *after* normalisation.
    tech_factor : Series
        Technical factor used for correction.
    markers_to_highlight : list, optional
        Specific markers to show in the barplot.  Defaults to the
        ``top_n`` markers with the largest |corr_pre|.
    top_n : int, default 25
        Number of markers shown when ``markers_to_highlight`` is None.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    corr_df : DataFrame
        ``marker``, ``corr_pre``, ``corr_post``.
    """
    assert asinh_pre.index.equals(asinh_post.index)
    assert asinh_pre.index.equals(tech_factor.index)

    num_cols = asinh_pre.select_dtypes(include=[np.number]).columns.tolist()
    corr_pre = asinh_pre[num_cols].corrwith(tech_factor)
    corr_post = asinh_post[num_cols].corrwith(tech_factor)

    corr_df = pd.DataFrame(
        {"marker": corr_pre.index, "corr_pre": corr_pre.values,
         "corr_post": corr_post.reindex(corr_pre.index).values}
    )

    if markers_to_highlight is not None:
        markers = [m for m in markers_to_highlight if m in corr_df["marker"].values]
    else:
        markers = (
            corr_df
            .assign(abs_pre=lambda d: d["corr_pre"].abs())
            .sort_values("abs_pre", ascending=False)
            .head(top_n)["marker"]
            .tolist()
        )

    long_df = (
        corr_df
        .loc[corr_df["marker"].isin(markers)]
        .melt(id_vars="marker", value_vars=["corr_pre", "corr_post"],
              var_name="state", value_name="corr")
    )
    long_df["state"] = long_df["state"].map(
        {"corr_pre": "Before", "corr_post": "After"}
    )

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    sns.barplot(data=long_df, x="marker", y="corr", hue="state", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Pearson corr(marker, tech)")
    ax.set_xlabel("")
    ax.set_title("Correlations with technical factor (selected markers)")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.scatter(corr_pre, corr_post, s=10, alpha=0.6)
    lim = max(corr_pre.abs().max(), corr_post.abs().max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("corr pre")
    ax.set_ylabel("corr post")
    ax.set_title("All markers: correlation pre vs. post")

    plt.tight_layout()
    plt.show()
    return corr_df


def plot_gamma_qc(
    gamma: Dict[str, float],
    marker_groups: Optional[Dict[str, List[str]]] = None,
    figsize: tuple = (12, 4),
):
    """Barplot of per-marker γ values.

    Parameters
    ----------
    gamma : dict
        ``marker → gamma`` mapping from a :class:`CytofTransformResult`.
    marker_groups : dict, optional
        ``group_name → [marker, ...]`` — used to colour bars by group.
    figsize : tuple
        Matplotlib figure size.
    """
    gamma_series = pd.Series(gamma).sort_values(ascending=False)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    if marker_groups is None:
        gamma_series.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_ylabel("γ (slope vs tech factor)")
        ax.set_xlabel("marker")
        ax.set_title("Marker-specific γ values")
        ax.tick_params(axis="x", rotation=90)
    else:
        df = gamma_series.rename("gamma").reset_index().rename(columns={"index": "marker"})
        group_labels = []
        for m in df["marker"]:
            assigned = next(
                (gname for gname, mlist in marker_groups.items() if m in mlist), "other"
            )
            group_labels.append(assigned)
        df["group"] = group_labels
        sns.barplot(data=df, x="marker", y="gamma", hue="group", dodge=False, ax=ax)
        ax.set_ylabel("γ (slope vs tech factor)")
        ax.set_xlabel("marker")
        ax.set_title("Marker-specific γ values by group")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(frameon=False, title="Group")

    plt.tight_layout()
    plt.show()


def plot_umap_qc(
    asinh_pre: pd.DataFrame,
    asinh_post: pd.DataFrame,
    tech_factor: pd.Series,
    umap_markers: Optional[Sequence[str]] = None,
    bio_marker: Optional[str] = None,
    control_histones: Optional[Sequence[str]] = None,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 0,
    figsize: tuple = (14, 10),
):
    """UMAP QC — compare pre vs. post normalisation in a shared embedding.

    Builds a 2 × 2 figure:

    * Top-left  : UMAP coloured by technical factor.
    * Top-right : UMAP coloured by mean core-histone intensity (pre).
    * Bottom-left / -right : A biological marker before / after correction.

    Requires ``umap-learn`` (``pip install umap-learn``).

    Parameters
    ----------
    asinh_pre : DataFrame
        Data before normalisation.
    asinh_post : DataFrame
        Data after normalisation.
    tech_factor : Series
        Technical factor.
    umap_markers : list, optional
        Features used for the UMAP embedding (default: all columns of
        ``asinh_post``).
    bio_marker : str, optional
        A biological marker to colour bottom panels.
    control_histones : list, optional
        Markers whose mean is shown as "total histone intensity".
    n_neighbors, min_dist, random_state : UMAP parameters.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    umap_coords : ndarray, shape (n_cells, 2)
    """
    if not _UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is required for plot_umap_qc. "
            "Install it with:  pip install umap-learn"
        )

    assert asinh_pre.index.equals(asinh_post.index)
    assert asinh_pre.index.equals(tech_factor.index)

    if umap_markers is None:
        umap_markers = asinh_post.columns.tolist()

    X = asinh_post[list(umap_markers)].values
    X_std = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        verbose=True,
    )
    umap_coords = reducer.fit_transform(X_std)

    sns.set(style="white")
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                    c=tech_factor.values, s=3, cmap="viridis")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("Technical factor")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("tech factor")

    if control_histones is not None:
        total_hist_pre = asinh_pre[list(control_histones)].mean(axis=1)
        ax = axes[0, 1]
        sc2 = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                         c=total_hist_pre.values, s=3, cmap="magma")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title("Total core-histone intensity (pre)")
        plt.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04).set_label("mean(histones)")
    else:
        axes[0, 1].axis("off")

    if bio_marker is not None and bio_marker in asinh_pre.columns:
        for ax, vals, title in zip(
            [axes[1, 0], axes[1, 1]],
            [asinh_pre[bio_marker].values, asinh_post[bio_marker].values],
            [f"{bio_marker} (pre)", f"{bio_marker} (post)"],
        ):
            sc_ = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                             c=vals, s=3, cmap="plasma")
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
            ax.set_title(title)
            plt.colorbar(sc_, ax=ax, fraction=0.046, pad=0.04).set_label(f"{bio_marker} (asinh)")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
    return umap_coords
