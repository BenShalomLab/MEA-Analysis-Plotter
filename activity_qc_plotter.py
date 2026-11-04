"""
activity_qc_plotter.py
Activity-scan QC plotting and genotype-effect testing.
Style borrowed from MEAPlotter (meaplotter.py).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── Shared style identical to MEAPlotter ────────────────────────────────────
_RC = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Liberation Sans", "Arial", "DejaVu Sans", "sans-serif"],
    "svg.fonttype":     "none",
    "pdf.fonttype":     42,
    "axes.labelsize":   9,
    "axes.labelweight": "bold",
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
}

DEFAULT_PALETTE = {
    "MxWT":   "#4C72B0",
    "FxHET":  "#D55E00",
    "MxHEMI": "#A63226",
    "WT":     "#4C72B0",
    "HET":    "#D55E00",
    "HEMI":   "#A63226",
    "KI":     "#A63226",
}


def _resolve(df, group_col, group_order, palette):
    palette = palette if palette is not None else DEFAULT_PALETTE
    if group_order is None:
        if isinstance(palette, dict):
            group_order = [g for g in palette if g in df[group_col].unique()]
            group_order += [g for g in df[group_col].unique() if g not in group_order]
        else:
            group_order = sorted(df[group_col].unique())
    color_map = (
        palette if isinstance(palette, dict)
        else {g: c for g, c in zip(group_order, sns.color_palette(palette, len(group_order)))}
    )
    return group_order, color_map


# ── QC trajectory plot ───────────────────────────────────────────────────────

def plot_active_area_trajectory(
    df,
    group_col="NeuronType",
    div_col="DIV",
    value_col="Active_area",
    group_order=None,
    palette=None,
    figsize=(5.5, 3.5),
    scatter_alpha=0.25,
    scatter_size=18,
    scatter_jitter=0.25,
    sem_alpha=0.20,
    line_markersize=3,
    show_expected=True,
    show_threshold=None,
    threshold_label="threshold",
    title=None,
    ylabel="Active Area (%)",
    save_path=None,
    seed=0,
):
    """
    Publication-style active-area QC trajectory.

    Valid wells  : jittered scatter + mean ± SEM ribbon per genotype.
    MAD outliers : × marker in group colour  (requires `is_outlier` column).
    Residual dips: ◆ marker in group colour  (requires `is_bad_residual` column).
    """
    rng = np.random.default_rng(seed)
    group_order, color_map = _resolve(df, group_col, group_order, palette)

    has_exclude = "to_exclude"      in df.columns
    has_outlier = "is_outlier"      in df.columns
    has_resid   = "is_bad_residual" in df.columns

    valid   = df[~df["to_exclude"]].copy() if has_exclude else df.copy()
    mad_bad = df[df["is_outlier"]].copy()  if has_outlier else df.iloc[:0].copy()
    res_bad = df[df["is_bad_residual"]].copy() if has_resid else df.iloc[:0].copy()

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)

        for g in group_order:
            sub = valid[valid[group_col] == g]
            if sub.empty:
                continue
            jitter = rng.uniform(-scatter_jitter, scatter_jitter, size=len(sub))
            ax.scatter(
                sub[div_col].to_numpy(dtype=float) + jitter, sub[value_col],
                color=color_map.get(g, "grey"),
                alpha=scatter_alpha, s=scatter_size, edgecolor="none", zorder=2,
            )

        if not mad_bad.empty:
            for g in group_order:
                sub = mad_bad[mad_bad[group_col] == g]
                if not sub.empty:
                    ax.scatter(sub[div_col], sub[value_col],
                               color=color_map.get(g, "grey"),
                               marker="x", s=60, linewidths=1.4, zorder=5)
            ax.scatter([], [], color="dimgrey", marker="x", s=60,
                       linewidths=1.4, label="MAD outlier")

        if not res_bad.empty:
            for g in group_order:
                sub = res_bad[res_bad[group_col] == g]
                if not sub.empty:
                    ax.scatter(sub[div_col], sub[value_col],
                               color=color_map.get(g, "grey"),
                               marker="D", s=40, zorder=5)
            ax.scatter([], [], color="dimgrey", marker="D", s=40, label="Residual dip")

        for g in group_order:
            sub = valid[valid[group_col] == g]
            if sub.empty:
                continue
            sg = sub.groupby(div_col)[value_col].agg(["mean", "sem"]).dropna()
            divs  = sg.index.to_numpy(dtype=float)
            means = sg["mean"].to_numpy()
            sems  = sg["sem"].to_numpy()
            col   = color_map.get(g, "black")
            ax.plot(divs, means, "-o", lw=1.8, markersize=line_markersize,
                    color=col, zorder=4, label=g)
            ax.fill_between(divs, means - sems, means + sems,
                            color=col, alpha=sem_alpha, zorder=3)

        if show_expected and "expected_y" in df.columns:
            for g in group_order:
                sub = df[(df[group_col] == g) & df["expected_y"].notna()]
                if sub.empty:
                    continue
                exp = sub.groupby(div_col)["expected_y"].median()
                ax.plot(exp.index.to_numpy(dtype=float), exp.to_numpy(),
                        "--", lw=1.2, color=color_map.get(g, "black"), alpha=0.50, zorder=3)

        if show_threshold is not None:
            ax.axhline(show_threshold, color="crimson", lw=1.0, ls="--",
                       label=f"{threshold_label} = {show_threshold}")

        sns.despine(ax=ax)
        ax.set_xlabel(div_col, fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", format="svg")

    return fig, ax


# ── Genotype-effect test: LMM ────────────────────────────────────────────────

def _fit_lmm_ml(formula, data, unit_col, methods=("powell", "nm", "lbfgs")):
    """Random-intercept LMM by maximum likelihood (needed for likelihood-ratio tests).

    Tries several optimizers and returns the fit with the highest FINITE log-likelihood among those that converged.
    Raises RuntimeError if none gives a usable fit. (A fit that returns llf = +/-inf is discarded.)
    """
    import warnings
    import statsmodels.formula.api as smf
    best, errors = None, []
    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(formula, data=data, groups=data[unit_col]).fit(reml=False, method=method)
        except Exception as exc:                      # singular matrix etc.: try the next optimizer
            errors.append(f"{method}: {type(exc).__name__}")
            continue
        llf = float(res.llf)
        if not np.isfinite(llf) or not bool(getattr(res, "converged", True)):
            errors.append(f"{method}: llf={llf}, converged={getattr(res, 'converged', None)}")
            continue
        if best is None or llf > best.llf:
            best = res
    if best is None:
        raise RuntimeError(f"No usable LMM fit for '{formula}': " + "; ".join(errors))
    return best


def test_genotype_lmm(
    df,
    group_col="NeuronType",
    div_col="DIV",
    value_col="Active_area",
    unit_col="CHIP_WELL",
    group_order=None,
    palette=None,
    figsize=(5.5, 5.0),
    height_ratios=(3, 2),
    alpha=0.05,
    show_fit_panel=False,
    plot=True,
    save_path=None,
    transform=None,
    center_div=True,
):
    """
    Linear mixed model to test whether Active Area differs by genotype.

        value ~ Genotype * DIV + (1 | unit)

    Two likelihood-ratio tests via model comparison (REML=False):
      1. Genotype main effect  — is one group persistently higher/lower?
      2. Genotype × DIV        — do trajectories diverge over time?

    Panel A: raw data mean ± SEM + LRT badge.
    Panel B: LMM population-level fitted trajectories (only when show_fit_panel=True).

    Fitting (changed 2026-10-19, see activity_qc_plotter.py.bak_20261019 for the previous version)
    ---------------------------------------------------------------------------------------------
    * Each model is fitted with several optimizers and the best FINITE, converged fit is kept. The previous
      version took the first optimizer that did not raise; for some data L-BFGS returned a log-likelihood of +inf
      for the null model, giving chi2 = -inf and a spurious p = 1.
    * If a nested comparison is still not monotone (larger model with lower likelihood) chi2 is set to 0 and a note
      is added to lrt_results["notes"].
    * transform : None or "log1p" (or a callable) applied to value_col before fitting, e.g. for counts.
    * center_div : if True (default) DIV is centred on its mean before fitting, so the genotype coefficients and
      the pairwise Wald tests are differences at the mean DIV rather than at DIV = 0. LRTs are unaffected.

    Returns
    -------
    fig, ax_data, ax_fit (None if show_fit_panel=False), lrt_results dict
    """
    try:
        import statsmodels.formula.api as smf
        from scipy.stats import chi2 as chi2_dist
    except ImportError:
        raise ImportError("pip install statsmodels scipy")

    group_order, color_map = _resolve(df, group_col, group_order, palette)
    ref = group_order[0]

    df_w = df[[group_col, div_col, value_col, unit_col]].dropna().copy()
    df_w[group_col] = df_w[group_col].astype(str)

    # ── Optional transform and DIV centring (fitting copy only; df_w is kept for plotting) ──────────
    df_fit = df_w.copy()
    if transform == "log1p":
        df_fit[value_col] = np.log1p(df_fit[value_col].astype(float))
    elif callable(transform):
        df_fit[value_col] = transform(df_fit[value_col].astype(float))
    elif transform is not None:
        raise ValueError("transform must be None, 'log1p' or a callable")
    if center_div:
        df_fit[div_col] = df_fit[div_col].astype(float) - df_fit[div_col].astype(float).mean()

    # ── Fit three nested models (REML=False required for LRT) ───────────────
    ref_str      = ref.replace("'", "\\'")
    f_null       = f"{value_col} ~ {div_col}"
    f_main       = f"{value_col} ~ C({group_col}, Treatment('{ref_str}')) + {div_col}"
    f_full       = f"{value_col} ~ C({group_col}, Treatment('{ref_str}')) * {div_col}"

    notes = []

    def _fit(formula):
        return _fit_lmm_ml(formula, df_fit, unit_col)

    fit_null = _fit(f_null)
    fit_main = _fit(f_main)
    fit_full = _fit(f_full)

    # ── Likelihood-ratio tests ───────────────────────────────────────────────
    n_extra = len(group_order) - 1   # params added per step (n_groups - 1)

    def _chi2(ll_big, ll_small, label):
        c = 2 * (ll_big - ll_small)
        if c < 0:                                    # nested ML fits cannot decrease: optimizer failure
            notes.append(f"{label}: negative chi2 ({c:.3f}) set to 0 (optimizer did not reach the nested optimum)")
            c = 0.0
        return c

    lrt_main = {"chi2": _chi2(fit_main.llf, fit_null.llf, "genotype main effect"), "df": n_extra}
    lrt_int  = {"chi2": _chi2(fit_full.llf, fit_main.llf, "genotype x DIV"), "df": n_extra}
    lrt_main["p"] = chi2_dist.sf(lrt_main["chi2"], df=lrt_main["df"])
    lrt_int["p"]  = chi2_dist.sf(lrt_int["chi2"],  df=lrt_int["df"])

    def _stars(p):
        for thresh, label in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
            if p < thresh:
                return label
        return "ns"

    # ── All pairwise Wald tests (refit with each non-first group as reference) ─
    def _wald(fit, g_col, ref_s, g):
        key = f"C({g_col}, Treatment('{ref_s}'))[T.{g}]"
        if key in fit.pvalues.index:
            return float(fit.tvalues[key]), float(fit.pvalues[key])
        return np.nan, np.nan

    from itertools import combinations as _comb
    pairwise_wald = {}
    for g1, g2 in _comb(group_order, 2):
        r_s = g1.replace("'", "\\'")
        f_pw = f"{value_col} ~ C({group_col}, Treatment('{r_s}')) * {div_col}"
        fit_pw = _fit(f_pw)
        z, p = _wald(fit_pw, group_col, r_s, g2)
        pairwise_wald[(g1, g2)] = {"z": z, "p": p}

    # ── Print ────────────────────────────────────────────────────────────────
    w = 62
    print("=" * w)
    print(f"LMM: {value_col} ~ {group_col} * {div_col} + (1 | {unit_col})")
    print(f"Reference: {ref}   |   n wells: {df_w[unit_col].nunique()}")
    print("=" * w)
    print(f"\n  {'Effect':<32} {'χ²':>7}  {'df':>3}  {'p':>9}  {'':>5}")
    print(f"  {'-'*32} {'-'*7}  {'-'*3}  {'-'*9}  {'-'*5}")
    print(f"  {'Genotype (main effect)':<32} {lrt_main['chi2']:>7.3f}  "
          f"{lrt_main['df']:>3}  {lrt_main['p']:>9.4f}  {_stars(lrt_main['p']):>5}")
    print(f"  {'Genotype × DIV (trajectory)':<32} {lrt_int['chi2']:>7.3f}  "
          f"{lrt_int['df']:>3}  {lrt_int['p']:>9.4f}  {_stars(lrt_int['p']):>5}")

    print(f"\n  ── Interpretation {'─'*43}")
    if lrt_int["p"] < alpha:
        print(f"  ✦ Trajectories DIVERGE over time (interaction p = {lrt_int['p']:.4f}).")
        print("    Active Area differences are DIV-dependent.")
        print("    → Do NOT use an absolute threshold filter.")
        print("      Use within-genotype (MAD) QC only.")
    elif lrt_main["p"] < alpha:
        print(f"  ✦ Persistent genotype shift (main p = {lrt_main['p']:.4f}); slopes parallel.")
        print("    Active Area is offset between groups but tracks together over time.")
        print("    → Absolute threshold inappropriate; use per-genotype cutoffs.")
    else:
        print("  ✓ No significant genotype effect on Active Area.")
        print("    → Absolute-threshold QC is defensible.")

    print(f"\n  ── Pairwise Wald tests (full model) {'─'*25}")
    print(f"  {'Comparison':<28} {'z':>7}  {'p':>9}  {'':>5}")
    print(f"  {'-'*28} {'-'*7}  {'-'*9}  {'-'*5}")
    for (g1, g2), res in pairwise_wald.items():
        label = f"{g1} vs {g2}"
        z_str = f"{res['z']:>7.3f}" if np.isfinite(res['z']) else "    nan"
        p_str = f"{res['p']:>9.4f}" if np.isfinite(res['p']) else "      nan"
        print(f"  {label:<28} {z_str}  {p_str}  {_stars(res['p']) if np.isfinite(res['p']) else '':>5}")

    print(f"\n  ── Fixed-effect coefficients (full model, ref={ref}) {'─'*8}")
    fe_table = fit_full.summary().tables[1]
    print(fe_table.as_text() if hasattr(fe_table, "as_text") else fe_table.to_string())

    for n_ in notes:
        print(f"  [note] {n_}")
    lrt_results = {"main": lrt_main, "interaction": lrt_int, "fit_full": fit_full,
                   "pairwise_wald": pairwise_wald, "notes": notes,
                   "transform": transform, "center_div": center_div}

    if not plot:
        return None, None, None, lrt_results

    # ── Population-level fitted values (only needed for Panel B) ─────────────
    fitted_agg = None
    if show_fit_panel:
        re_vals = {k: float(v.iloc[0]) for k, v in fit_full.random_effects.items()}
        df_w["_re"]        = df_w[unit_col].map(re_vals).fillna(0.0)
        df_w["_fe_fitted"] = fit_full.fittedvalues - df_w["_re"]
        fitted_agg = df_w.groupby([group_col, div_col])["_fe_fitted"].mean().reset_index()

    # ── LRT badge ────────────────────────────────────────────────────────────
    badge = (
        f"Genotype main:   χ²({lrt_main['df']}) = {lrt_main['chi2']:.2f},  "
        f"p = {lrt_main['p']:.3f}  {_stars(lrt_main['p'])}\n"
        f"Genotype × DIV:  χ²({lrt_int['df']}) = {lrt_int['chi2']:.2f},  "
        f"p = {lrt_int['p']:.3f}  {_stars(lrt_int['p'])}"
    )

    # ── Figure ───────────────────────────────────────────────────────────────
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=figsize)
        if show_fit_panel:
            gs = GridSpec(2, 1, figure=fig,
                          height_ratios=list(height_ratios),
                          hspace=0.10,
                          left=0.12, right=0.82, top=0.93, bottom=0.10)
            ax_data = fig.add_subplot(gs[0])
            ax_fit  = fig.add_subplot(gs[1], sharex=ax_data)
        else:
            gs = GridSpec(1, 1, figure=fig,
                          left=0.12, right=0.82, top=0.93, bottom=0.10)
            ax_data = fig.add_subplot(gs[0])
            ax_fit  = None

        # Panel A: raw data + LRT badge
        for g in group_order:
            sub = df_w[df_w[group_col] == g]
            sg  = sub.groupby(div_col)[value_col].agg(["mean", "sem"]).dropna()
            col = color_map.get(g, "black")
            ax_data.plot(sg.index.to_numpy(dtype=float), sg["mean"].to_numpy(),
                         "-o", lw=1.8, markersize=3, color=col, label=g, zorder=4)
            ax_data.fill_between(
                sg.index.to_numpy(dtype=float),
                sg["mean"].to_numpy() - sg["sem"].to_numpy(),
                sg["mean"].to_numpy() + sg["sem"].to_numpy(),
                color=col, alpha=0.20, zorder=3,
            )

        ax_data.text(0.99, 0.03, badge, transform=ax_data.transAxes,
                     fontsize=6, va="bottom", ha="right", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

        sns.despine(ax=ax_data)
        ax_data.set_ylabel(f"{value_col.replace('_', ' ')} (%)", fontsize=9, fontweight="bold")
        if not show_fit_panel:
            ax_data.set_xlabel(div_col, fontsize=9, fontweight="bold")
        else:
            plt.setp(ax_data.get_xticklabels(), visible=False)
        ax_data.legend(frameon=False, fontsize=7,
                       bbox_to_anchor=(1.02, 1), loc="upper left")

        # Panel B: LMM fitted (population level) — optional
        if show_fit_panel and fitted_agg is not None:
            for g in group_order:
                sub_fit = fitted_agg[fitted_agg[group_col] == g].sort_values(div_col)
                col     = color_map.get(g, "black")
                ax_fit.plot(sub_fit[div_col].to_numpy(dtype=float),
                            sub_fit["_fe_fitted"].to_numpy(),
                            "-o", lw=1.8, markersize=3, color=col, zorder=4)

            sns.despine(ax=ax_fit)
            ax_fit.set_xlabel(div_col, fontsize=9, fontweight="bold")
            ax_fit.set_ylabel(f"LMM fitted\n{value_col.replace('_', ' ')} (%)",
                              fontsize=9, fontweight="bold")
            ax_fit.text(0.01, 1.02, "B  LMM population-level fit",
                        transform=ax_fit.transAxes,
                        fontsize=8, fontweight="bold", va="bottom", ha="left")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", format="svg")

    return fig, ax_data, ax_fit, lrt_results
