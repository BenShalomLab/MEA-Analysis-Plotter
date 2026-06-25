import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import kruskal, ranksums
from itertools import combinations
from matplotlib.backends.backend_pdf import PdfPages

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


class MEAPlotter:
    def __init__(self, group_order=None, palette=None, div_order=None):
        """
        Publication-ish defaults + centralized config.
        """
        # --- FONT CONFIGURATION ---
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['pdf.fonttype'] = 42

        plt.rcParams['axes.titlesize'] = 9
        plt.rcParams['axes.labelsize'] = 8
        plt.rcParams['xtick.labelsize'] = 6
        plt.rcParams['ytick.labelsize'] = 6
        plt.rcParams['legend.fontsize'] = 7
        plt.rcParams['legend.title_fontsize'] = 8

        # --- TICK APPEARANCE ---
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['xtick.major.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 0.8

        # --- GLOBAL FONT WEIGHT ---
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['font.weight'] = 'bold'

        # --- DEFAULTS ---
        self.group_order_default = list(group_order) if group_order is not None else None
        self.div_order_default = list(div_order) if div_order is not None else None
        self.palette_default = palette  # "Set2" OR {"WT":"#..",...}

    # --------------------------
    # helpers to resolve defaults
    # --------------------------
    def _resolve_group_order(self, df, col, group_order):
        if group_order is not None:
            return list(group_order)
        if self.group_order_default is not None:
            return list(self.group_order_default)
        return sorted(df[col].dropna().unique())

    def _resolve_div_order(self, df, col, div_order):
        if div_order is not None:
            return list(div_order)
        if self.div_order_default is not None:
            return list(self.div_order_default)
        vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique()
        return sorted(vals)

    def _resolve_palette(self, palette):
        if palette is not None:
            return palette
        return self.palette_default

    def _prepare_data(self, df, cols, div_col=None):
        """
        Common cleaning helper.
        """
        d = df.dropna(subset=cols).copy()

        if div_col is not None:
            d[div_col] = pd.to_numeric(d[div_col], errors="coerce")
            d = d.dropna(subset=[div_col])
            d[div_col] = d[div_col].astype(int)

        return d

    # --------------------------
    # stats helpers
    # --------------------------
    @staticmethod
    def _stars_from_p(p):
        if not np.isfinite(p):
            return "ns"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    @staticmethod
    def _safe_sem(x):
        x = pd.Series(x).dropna()
        return x.sem() if len(x) > 1 else np.nan

    @staticmethod
    def _safe_cohens_d(d1, d2):
        d1 = pd.Series(d1).dropna()
        d2 = pd.Series(d2).dropna()
        if len(d1) < 2 or len(d2) < 2:
            return np.nan
        sd1 = d1.std(ddof=1)
        sd2 = d2.std(ddof=1)
        pooled_std = np.sqrt((sd1**2 + sd2**2) / 2)
        if not np.isfinite(pooled_std) or pooled_std <= 0:
            return np.nan
        return (d1.mean() - d2.mean()) / pooled_std

    @staticmethod
    def _fmt_grp(d):
        """Format a group's data as 'mean ± SEM (n=N)' or 'NA (n=N)'."""
        d = pd.Series(d).dropna()
        if len(d) == 0:
            return "NA (n=0)"
        sem = d.sem() if len(d) > 1 else np.nan
        sem_str = f"{sem:.2f}" if np.isfinite(sem) else "NA"
        return f"{d.mean():.2f} ± {sem_str} (n={len(d)})"

    @staticmethod
    def _build_color_map(group_order, palette):
        """Build {group: color} from a dict palette or seaborn palette name."""
        if isinstance(palette, dict):
            return {g: palette[g] for g in group_order}
        colors = sns.color_palette(palette, len(group_order))
        return {g: colors[i] for i, g in enumerate(group_order)}

    @staticmethod
    def _compute_clip_cap(yvals, method="quantile", q=0.98, k=1.5):
        """Return upper clip ceiling or None if clip disabled / no data."""
        yvals = pd.Series(yvals).dropna()
        if len(yvals) == 0:
            return None
        if method == "iqr":
            q1, q3 = float(yvals.quantile(0.25)), float(yvals.quantile(0.75))
            iqr = q3 - q1
            return float(q3 + k * iqr) if np.isfinite(iqr) and iqr > 0 else float(yvals.quantile(q))
        return float(yvals.quantile(q))

    def _draw_upper_outliers(self, ax, d, y, div_col, group_col,
                             div_order, group_order, div_to_x, offsets, color_map, cap,
                             stem_frac=0.04, marker_size=26):
        """Lollipop markers for values above cap (shared by line and bar plots)."""
        upper_out = d[d[y] > cap].copy()
        if upper_out.empty:
            return
        ymin_ax, ymax_ax = ax.get_ylim()
        yr = ymax_ax - ymin_ax
        y_tip   = ymax_ax - 0.01 * yr
        y_stem0 = y_tip   - stem_frac * yr
        for _, r in upper_out.iterrows():
            div_val, g = r[div_col], r[group_col]
            if div_val not in div_order or g not in group_order:
                continue
            x   = float(div_to_x[div_val]) + offsets[g]
            col = color_map[g]
            ax.plot([x, x], [y_stem0, y_tip], color=col, lw=0.9, zorder=5)
            ax.scatter([x], [y_tip], marker="^", s=marker_size,
                       color=col, edgecolor="black", linewidth=0.25, zorder=6)

    def get_style(
        self,
        figsize=(6, 4),
        title="Title",
        xlabel="X label",
        ylabel="Y label",
        xticks=None,
        xticklabels=None
    ):
        """
        Returns a styled (fig, ax) with the MEAPlotter defaults applied.
        No data is plotted. Intended for quick notebook experimentation.
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=8, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")

        ax.tick_params(axis="x", labelsize=6, width=0.8, length=3)
        ax.tick_params(axis="y", labelsize=6, width=0.8, length=3)

        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)

        sns.despine(ax=ax)
        return fig, ax

    # --------------------------
    # centralized stats dispatcher
    # --------------------------
    def _run_group_stats(self, df, group_col, y, group_order=None, alpha=0.05):
        """
        Central inferential logic.

        Rules:
        - <2 groups with data -> no test
        - exactly 2 groups -> independent t-test
        - >=3 groups -> one-way ANOVA, then Tukey HSD if omnibus is significant

        Returns dict with:
            test_type
            omnibus_stat
            omnibus_p
            pairwise_df
        """
        d = self._prepare_data(df, [group_col, y])

        if d.empty:
            return {
                "test_type": None,
                "omnibus_stat": np.nan,
                "omnibus_p": np.nan,
                "pairwise_df": pd.DataFrame()
            }

        group_order = self._resolve_group_order(d, group_col, group_order)
        present_groups = []
        for g in group_order:
            n = d.loc[d[group_col] == g, y].dropna().shape[0]
            if n > 0:
                present_groups.append(g)

        if len(present_groups) < 2:
            return {
                "test_type": None,
                "omnibus_stat": np.nan,
                "omnibus_p": np.nan,
                "pairwise_df": pd.DataFrame()
            }

        # ---------------- 2 groups -> t-test ----------------
        if len(present_groups) == 2:
            g1, g2 = present_groups
            d1 = d.loc[d[group_col] == g1, y].dropna()
            d2 = d.loc[d[group_col] == g2, y].dropna()

            if len(d1) <= 1 or len(d2) <= 1:
                pairwise_df = pd.DataFrame([{
                    "group1": g1, "group2": g2, "comparison": f"{g1} vs {g2}",
                    "p_adj": np.nan, "raw_p": np.nan, "reject": False, "stat": np.nan,
                    "Grp1_Stats": self._fmt_grp(d1), "Grp2_Stats": self._fmt_grp(d2),
                    "Cohen's d": np.nan,
                }])
                return {"test_type": "t-test", "omnibus_stat": np.nan,
                        "omnibus_p": np.nan, "pairwise_df": pairwise_df}

            # Standard 2-group independent t-test
            t_stat, p_val = stats.ttest_ind(d1, d2, equal_var=True, nan_policy="omit")
            cohens_d = self._safe_cohens_d(d1, d2)

            pairwise_df = pd.DataFrame([{
                "group1": g1, "group2": g2, "comparison": f"{g1} vs {g2}",
                "p_adj": p_val, "raw_p": p_val,
                "reject": bool(p_val < alpha) if np.isfinite(p_val) else False,
                "stat": t_stat,
                "Grp1_Stats": self._fmt_grp(d1), "Grp2_Stats": self._fmt_grp(d2),
                "Cohen's d": cohens_d,
            }])

            return {
                "test_type": "t-test",
                "omnibus_stat": t_stat,
                "omnibus_p": p_val,
                "pairwise_df": pairwise_df
            }

        # ---------------- 3+ groups -> one-way ANOVA ----------------
        # safer formula quoting
        yq = f'Q("{y}")'
        gq = f'Q("{group_col}")'
        formula = f"{yq} ~ C({gq})"

        try:
            model = ols(formula, data=d).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            row_name = f"C({gq})"
            omnibus_stat = anova_table.loc[row_name, "F"]
            omnibus_p = anova_table.loc[row_name, "PR(>F)"]
        except Exception:
            omnibus_stat = np.nan
            omnibus_p = np.nan

        pairwise_rows = []

        if np.isfinite(omnibus_p) and omnibus_p < alpha:
            tukey = pairwise_tukeyhsd(endog=d[y], groups=d[group_col], alpha=alpha)
            tukey_df = pd.DataFrame(
                tukey._results_table.data[1:],
                columns=tukey._results_table.data[0]
            )

            tukey_df = tukey_df.rename(columns={
                "p-adj": "p_adj",
                "meandiff": "stat"
            })

            for _, row in tukey_df.iterrows():
                g1 = row["group1"]
                g2 = row["group2"]

                d1 = d.loc[d[group_col] == g1, y].dropna()
                d2 = d.loc[d[group_col] == g2, y].dropna()

                pairwise_rows.append({
                    "group1": g1, "group2": g2, "comparison": f"{g1} vs {g2}",
                    "p_adj": float(row["p_adj"]) if pd.notna(row["p_adj"]) else np.nan,
                    "raw_p": np.nan, "reject": bool(row["reject"]), "stat": row["stat"],
                    "Grp1_Stats": self._fmt_grp(d1), "Grp2_Stats": self._fmt_grp(d2),
                    "Cohen's d": self._safe_cohens_d(d1, d2),
                })
        else:
            for g1, g2 in combinations(present_groups, 2):
                d1 = d.loc[d[group_col] == g1, y].dropna()
                d2 = d.loc[d[group_col] == g2, y].dropna()
                pairwise_rows.append({
                    "group1": g1, "group2": g2, "comparison": f"{g1} vs {g2}",
                    "p_adj": np.nan, "raw_p": np.nan, "reject": False, "stat": np.nan,
                    "Grp1_Stats": self._fmt_grp(d1), "Grp2_Stats": self._fmt_grp(d2),
                    "Cohen's d": self._safe_cohens_d(d1, d2),
                })

        pairwise_df = pd.DataFrame(pairwise_rows)

        return {
            "test_type": "one-way ANOVA",
            "omnibus_stat": omnibus_stat,
            "omnibus_p": omnibus_p,
            "pairwise_df": pairwise_df
        }

    def calculate_stats(self, df, x, y, order=None, alpha=0.05):
        """
        Single-panel public stats API.
        Centralized version:
        - 2 groups -> t-test
        - 3+ groups -> one-way ANOVA + Tukey
        """
        d = self._prepare_data(df, [x, y])
        order = self._resolve_group_order(d, x, order)

        result = self._run_group_stats(
            d,
            group_col=x,
            y=y,
            group_order=order,
            alpha=alpha
        )

        pairwise_df = result["pairwise_df"].copy()
        if pairwise_df.empty:
            return pd.DataFrame(columns=[
                "Comparison", "Grp1_Stats", "Grp2_Stats",
                "Stat", "p-val", "Sig", "Cohen's d",
                "Test", "Omnibus Stat", "Omnibus p-val"
            ])

        out_rows = []
        for _, row in pairwise_df.iterrows():
            p = row.get("p_adj", np.nan)
            out_rows.append({
                "Comparison": row["comparison"],
                "Grp1_Stats": row["Grp1_Stats"],
                "Grp2_Stats": row["Grp2_Stats"],
                "Stat": row["stat"],
                "p-val": p,
                "Sig": self._stars_from_p(p),
                "Cohen's d": row["Cohen's d"],
                "Test": result["test_type"],
                "Omnibus Stat": result["omnibus_stat"],
                "Omnibus p-val": result["omnibus_p"]
            })

        return pd.DataFrame(out_rows)

    def calculate_stats_by_div(self, df, div_col, group_col, y, div_order=None, group_order=None, alpha=0.05):
        """
        Per-DIV public stats API.
        Centralized version:
        - 2 groups -> t-test
        - 3+ groups -> one-way ANOVA + Tukey
        """
        d = self._prepare_data(df, [div_col, group_col, y], div_col=div_col)
        div_order = self._resolve_div_order(d, div_col, div_order)
        group_order = self._resolve_group_order(d, group_col, group_order)

        rows = []

        print(f"\n{'='*20} CENTRALIZED STATS: {y} by {div_col} {'='*20}")

        for div_val in div_order:
            div_data = d[d[div_col] == div_val].copy()
            if div_data.empty:
                continue

            result = self._run_group_stats(
                div_data,
                group_col=group_col,
                y=y,
                group_order=group_order,
                alpha=alpha
            )

            print(f"\n--- DIV {div_val} ---")
            print(f"Test: {result['test_type']}, omnibus p = {result['omnibus_p']}")

            pairwise_df = result["pairwise_df"]

            if pairwise_df.empty:
                rows.append({
                    "DIV": div_val,
                    "Comparison": "None",
                    "Grp1_Stats": "",
                    "Grp2_Stats": "",
                    "Stat": np.nan,
                    "p-val": np.nan,
                    "Sig": "ns",
                    "Cohen's d": np.nan,
                    "Test": result["test_type"],
                    "Omnibus Stat": result["omnibus_stat"],
                    "Omnibus p-val": result["omnibus_p"],
                    "group1": np.nan,
                    "group2": np.nan,
                    "reject": False
                })
                continue

            for _, row in pairwise_df.iterrows():
                p = row.get("p_adj", np.nan)
                stars = self._stars_from_p(p)

                print(f"  {row['comparison']}: p={p}, {stars}")

                rows.append({
                    "DIV": div_val,
                    "Comparison": row["comparison"],
                    "Grp1_Stats": row["Grp1_Stats"],
                    "Grp2_Stats": row["Grp2_Stats"],
                    "Stat": row["stat"],
                    "p-val": p,
                    "Sig": stars,
                    "Cohen's d": row["Cohen's d"],
                    "Test": result["test_type"],
                    "Omnibus Stat": result["omnibus_stat"],
                    "Omnibus p-val": result["omnibus_p"],
                    "group1": row["group1"],
                    "group2": row["group2"],
                    "reject": bool(row["reject"])
                })

        return pd.DataFrame(rows)

    def calculate_stats_at_timepoints(
        self,
        df,
        div_col,
        group_col,
        y,
        selected_divs,
        group_order=None,
        min_n=3,
        alpha=0.05,
    ):
        """
        Non-parametric stats at pre-selected timepoints only.

        Per timepoint: Kruskal-Wallis omnibus.
        If KW significant: pairwise Wilcoxon rank-sum post-hoc.
        BH-FDR correction applied across ALL pairwise tests in the family
        (all significant-KW timepoints × all pairs for this metric).
        Groups with n < min_n are excluded before testing.

        Parameters
        ----------
        selected_divs : list of int/float — e.g. [13, 20, 27]
        min_n : int — minimum n per group to enter any test (default 3)

        Returns DataFrame with columns:
            DIV, group1, group2, comparison,
            KW_stat, KW_p,
            W_stat, raw_p, q_val, reject, Sig,
            Grp1_Stats, Grp2_Stats, Cohen's d
        """
        d = self._prepare_data(df, [div_col, group_col, y], div_col=div_col)
        group_order = self._resolve_group_order(d, group_col, group_order)
        selected_divs = [int(v) for v in selected_divs]

        print(f"\n{'='*20} STATS (KW + Wilcoxon, BH-FDR): {y} {'='*20}")
        print(f"Selected timepoints: {selected_divs}  min_n={min_n}")

        omnibus_cache = {}    # div_val -> (kw_stat, kw_p, present_groups)
        pending_pairwise = []  # rows where pairwise Wilcoxon was run (need BH-FDR)

        for div_val in selected_divs:
            div_data = d[d[div_col] == div_val]
            present_groups = []
            group_data = {}
            for g in group_order:
                vals = div_data.loc[div_data[group_col] == g, y].dropna()
                if len(vals) >= min_n:
                    present_groups.append(g)
                    group_data[g] = vals

            if len(present_groups) < 2:
                print(f"  DIV {div_val}: skipped — fewer than 2 groups with n ≥ {min_n}")
                omnibus_cache[div_val] = (np.nan, np.nan, present_groups)
                continue

            try:
                kw_stat, kw_p = kruskal(*[group_data[g].values for g in present_groups])
            except Exception:
                kw_stat, kw_p = np.nan, np.nan

            omnibus_cache[div_val] = (kw_stat, kw_p, present_groups)
            print(f"  DIV {div_val}: KW H={kw_stat:.3f} p={kw_p:.4g}", end="")

            if np.isfinite(kw_p) and kw_p < alpha:
                print(" → pairwise tests")
                for g1, g2 in combinations(present_groups, 2):
                    d1, d2 = group_data[g1], group_data[g2]
                    try:
                        W_stat, raw_p = ranksums(d1.values, d2.values)
                    except Exception:
                        W_stat, raw_p = np.nan, np.nan
                    pending_pairwise.append({
                        "DIV":        div_val,
                        "group1":     g1,
                        "group2":     g2,
                        "comparison": f"{g1} vs {g2}",
                        "KW_stat":    kw_stat,
                        "KW_p":       kw_p,
                        "W_stat":     W_stat,
                        "raw_p":      raw_p,
                        "Grp1_Stats": f"{d1.mean():.2f} ± {self._safe_sem(d1):.2f} (n={len(d1)})",
                        "Grp2_Stats": f"{d2.mean():.2f} ± {self._safe_sem(d2):.2f} (n={len(d2)})",
                        "Cohen's d":  self._safe_cohens_d(d1, d2),
                    })
            else:
                print(" → ns (no pairwise tests)")

        # BH-FDR correction across all collected pairwise Wilcoxon tests
        if pending_pairwise:
            raw_ps = [r["raw_p"] if np.isfinite(r["raw_p"]) else 1.0 for r in pending_pairwise]
            reject_arr, q_vals, _, _ = multipletests(raw_ps, method="fdr_bh", alpha=alpha)
            for i, row in enumerate(pending_pairwise):
                row["q_val"]  = float(q_vals[i])
                row["reject"] = bool(reject_arr[i])
                row["Sig"]    = self._stars_from_p(q_vals[i])
            print(f"\n  BH-FDR: {int(sum(reject_arr))}/{len(reject_arr)} pairs significant (q < {alpha})")

        # Build complete output — include ns rows for non-tested pairs
        rows = list(pending_pairwise)
        tested_keys = {(r["DIV"], r["group1"], r["group2"]) for r in pending_pairwise}

        for div_val in selected_divs:
            kw_stat, kw_p, present_groups = omnibus_cache.get(div_val, (np.nan, np.nan, []))
            div_data = d[d[div_col] == div_val]
            for g1, g2 in combinations(group_order, 2):
                if (div_val, g1, g2) in tested_keys:
                    continue
                d1 = div_data.loc[div_data[group_col] == g1, y].dropna()
                d2 = div_data.loc[div_data[group_col] == g2, y].dropna()
                rows.append({
                    "DIV":        div_val,
                    "group1":     g1,
                    "group2":     g2,
                    "comparison": f"{g1} vs {g2}",
                    "KW_stat":    kw_stat,
                    "KW_p":       kw_p,
                    "W_stat":     np.nan,
                    "raw_p":      np.nan,
                    "q_val":      np.nan,
                    "reject":     False,
                    "Sig":        "ns",
                    "Grp1_Stats": f"{d1.mean():.2f} ± {self._safe_sem(d1):.2f} (n={len(d1)})" if len(d1) > 0 else "NA",
                    "Grp2_Stats": f"{d2.mean():.2f} ± {self._safe_sem(d2):.2f} (n={len(d2)})" if len(d2) > 0 else "NA",
                    "Cohen's d":  self._safe_cohens_d(d1, d2),
                })

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["DIV", "group1", "group2"]).reset_index(drop=True)
        return out

    # --------------------------
    # backward-compatible Welch APIs
    # --------------------------
    def calculate_stats_welch(self, df, group_col, y, group_order=None):
        """Pairwise Welch t-tests (uncorrected) across groups, no DIV grouping."""
        return self.calculate_stats_by_div_welch(
            df, div_col=None, group_col=group_col, y=y, group_order=group_order
        )

    def calculate_stats_by_div_welch(self, df, div_col, group_col, y,
                                     div_order=None, group_order=None):
        """
        Pairwise Welch t-tests (uncorrected) per DIV.
        Pass div_col=None to test across all data without DIV grouping.
        """
        drop_cols = [c for c in [div_col, group_col, y] if c is not None]
        data = df.dropna(subset=drop_cols).copy()
        if div_col is not None:
            data[div_col] = pd.to_numeric(data[div_col], errors="coerce")
            data = data.dropna(subset=[div_col])
            data[div_col] = data[div_col].astype(int)
            div_vals = self._resolve_div_order(data, div_col, div_order)
        else:
            div_vals = [None]

        group_order = self._resolve_group_order(data, group_col, group_order)
        pairs       = list(combinations(group_order, 2))
        rows        = []

        label = f"{y} by {div_col}" if div_col else y
        print(f"\n{'='*20} STATS (Welch, uncorrected): {label} {'='*20}")

        for div_val in div_vals:
            sub   = data[data[div_col] == div_val] if div_col is not None else data
            if sub.empty:
                continue
            if div_val is not None:
                print(f"\n--- DIV {div_val} ---")
            indent = "  " if div_val is not None else ""

            for g1, g2 in pairs:
                d1 = sub.loc[sub[group_col] == g1, y].dropna()
                d2 = sub.loc[sub[group_col] == g2, y].dropna()
                n1, n2 = len(d1), len(d2)
                base = {"Comparison": f"{g1} vs {g2}",
                        "Grp1_Stats": self._fmt_grp(d1),
                        "Grp2_Stats": self._fmt_grp(d2),
                        "Cohen's d":  np.nan}
                if div_val is not None:
                    base["DIV"] = div_val
                if n1 <= 1 or n2 <= 1:
                    print(f"{indent}{g1} vs {g2}: not enough data (n1={n1}, n2={n2})")
                    rows.append({**base, "t-stat": np.nan, "p-val": np.nan, "Sig": "ns"})
                    continue
                t_stat, p_val = stats.ttest_ind(d1, d2, equal_var=False, nan_policy="omit")
                cohens_d = self._safe_cohens_d(d1, d2)
                stars    = self._stars_from_p(p_val)
                print(f"{indent}{g1} (n={n1}) vs {g2} (n={n2}): "
                      f"p={p_val:.4e} ({stars}), d={cohens_d:.3f}")
                rows.append({**base, "t-stat": t_stat, "p-val": p_val,
                             "Sig": stars, "Cohen's d": cohens_d})

        return pd.DataFrame(rows)

    # --------------------------
    # annotation helper
    # --------------------------
    def _annotate_stats_brackets(self, ax, stats_df, div_order, div_to_x, offsets, y_top_reserved_frac=0.20):
        """
        Draw significance brackets from centralized stats output.
        Expects stats_df with columns:
            DIV, group1, group2, reject, p-val, Sig
        """
        if stats_df is None or len(stats_df) == 0:
            return ax

        ymin_ax, ymax_ax = ax.get_ylim()
        yr = ymax_ax - ymin_ax
        base_top = ymax_ax - y_top_reserved_frac * yr
        bracket_step = 0.06 * yr
        max_annotation_y = base_top

        for div_val in div_order:
            div_tests = stats_df[
                (stats_df["DIV"] == div_val) &
                (stats_df["reject"] == True)
            ].copy()

            if div_tests.empty:
                continue

            level = 0
            for _, t in div_tests.iterrows():
                g1 = t["group1"]
                g2 = t["group2"]
                stars = t["Sig"]

                if stars == "ns":
                    continue
                if pd.isna(g1) or pd.isna(g2):
                    continue
                if g1 not in offsets or g2 not in offsets:
                    continue

                x1 = div_to_x[div_val] + offsets[g1]
                x2 = div_to_x[div_val] + offsets[g2]
                bracket_y = base_top + level * bracket_step

                ax.plot(
                    [x1, x1, x2, x2],
                    [
                        bracket_y,
                        bracket_y + 0.25 * bracket_step,
                        bracket_y + 0.25 * bracket_step,
                        bracket_y
                    ],
                    c="black",
                    lw=1.0,
                    zorder=7
                )

                ax.text(
                    (x1 + x2) / 2,
                    bracket_y + 0.30 * bracket_step,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    zorder=8
                )

                max_annotation_y = max(max_annotation_y, bracket_y + 0.55 * bracket_step)
                level += 1

        if max_annotation_y > ax.get_ylim()[1]:
            ax.set_ylim(ax.get_ylim()[0], max_annotation_y + 0.04 * yr)

        return ax

    def _annotate_dotplot_sig(self, ax, div_stats, group_order):
        """Significance brackets on a categorical dot-plot axis (group positions = integers)."""
        if div_stats is None or div_stats.empty:
            return

        pos_map  = {g: i for i, g in enumerate(group_order)}
        ymin, ymax = ax.get_ylim()
        yr       = ymax - ymin
        base_y   = ymax + 0.03 * yr
        step     = 0.14 * yr
        level    = 0

        for _, row in div_stats.iterrows():
            g1    = row.get("group1")
            g2    = row.get("group2")
            stars = row.get("Sig", "ns")
            if stars == "ns" or g1 not in pos_map or g2 not in pos_map:
                continue
            x1, x2   = pos_map[g1], pos_map[g2]
            yb        = base_y + level * step
            ax.plot(
                [x1, x1, x2, x2],
                [yb, yb + 0.3 * step, yb + 0.3 * step, yb],
                c="black", lw=0.8, zorder=8,
            )
            ax.text(
                (x1 + x2) / 2, yb + 0.35 * step, stars,
                ha="center", va="bottom", fontsize=7, zorder=9,
            )
            level += 1

        if level > 0:
            ax.set_ylim(ymin, base_y + (level + 0.7) * step)

    def _compute_auc_stats(self, auc_df, group_col, auc_col, group_order, min_n=3, alpha=0.05):
        """KW + pairwise Wilcoxon + BH-FDR on per-unit AUC values."""
        from scipy.stats import kruskal, ranksums
        from statsmodels.stats.multitest import multipletests

        groups = [auc_df.loc[auc_df[group_col] == g, auc_col].dropna().values
                  for g in group_order if len(auc_df.loc[auc_df[group_col] == g, auc_col].dropna()) >= min_n]
        if len(groups) < 2:
            return pd.DataFrame()
        kw_stat, kw_p = kruskal(*groups)
        if kw_p >= alpha:
            return pd.DataFrame()

        pairs = [(g1, g2) for idx, g1 in enumerate(group_order) for g2 in group_order[idx + 1:]]
        ws, raw_ps, valid_pairs = [], [], []
        for g1, g2 in pairs:
            v1 = auc_df.loc[auc_df[group_col] == g1, auc_col].dropna().values
            v2 = auc_df.loc[auc_df[group_col] == g2, auc_col].dropna().values
            if len(v1) >= min_n and len(v2) >= min_n:
                w, p = ranksums(v1, v2)
                ws.append(w); raw_ps.append(p); valid_pairs.append((g1, g2))
        if not raw_ps:
            return pd.DataFrame()

        rejects, q_vals, _, _ = multipletests(raw_ps, alpha=alpha, method="fdr_bh")
        sig_map = {0: "ns", 1: "*", 2: "**", 3: "***", 4: "****"}
        rows = []
        for (g1, g2), w, rp, q, rej in zip(valid_pairs, ws, raw_ps, q_vals, rejects):
            n_stars = sum(q < t for t in [0.05, 0.01, 0.001, 0.0001])
            rows.append({
                "group1": g1, "group2": g2,
                "W_stat": w, "raw_p": rp, "q_val": q,
                "reject": bool(rej),
                "Sig": sig_map.get(n_stars, "ns") if rej else "ns",
            })
        return pd.DataFrame(rows)

    # --------------------------
    # plots
    # --------------------------
    def plot_bars(self, df, x, y, order=None, palette=None, title=None, ax=None, annotate=False):
        """
        Simple grouped bar + scatter (single panel).
        annotate=False by default.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        data = df.dropna(subset=[x, y]).copy()
        order = self._resolve_group_order(data, x, order)
        palette = self._resolve_palette(palette)

        sns.barplot(
            data=data, x=x, y=y, order=order, palette=palette,
            errorbar='se', capsize=0.12, alpha=0.75,
            edgecolor='black', linewidth=0.8, ax=ax
        )

        sns.stripplot(
            data=data, x=x, y=y, order=order,
            color='black', alpha=0.7, jitter=0.18, size=5,
            edgecolor='black', linewidth=0.3, ax=ax
        )

        ax.set_title(title if title else y, fontweight='bold', pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(y.replace('_', ' '), fontweight='bold')
        sns.despine(ax=ax)

        # Optional single-panel annotation using centralized stats
        if annotate:
            stats_df = self.calculate_stats(df, x=x, y=y, order=order, alpha=0.05)

            if len(stats_df) > 0:
                ymin_ax, ymax_ax = ax.get_ylim()
                yr = ymax_ax - ymin_ax
                base_y = ymax_ax + 0.02 * yr
                step = 0.08 * yr

                pos_map = {g: i for i, g in enumerate(order)}
                level = 0

                for _, row in stats_df.iterrows():
                    p = row["p-val"]
                    stars = row["Sig"]

                    if stars == "ns" or not np.isfinite(p):
                        continue

                    g1, g2 = row["Comparison"].split(" vs ")
                    x1, x2 = pos_map[g1], pos_map[g2]
                    yb = base_y + level * step

                    ax.plot([x1, x1, x2, x2], [yb, yb + 0.2 * step, yb + 0.2 * step, yb], c="black", lw=1.1)
                    ax.text((x1 + x2) / 2, yb + 0.25 * step, stars, ha="center", va="bottom", fontsize=9)
                    level += 1

                ax.set_ylim(ymin_ax, base_y + (level + 1) * step)

        return ax

    def plot_line_sem_by_div(
        self,
        df,
        div_col,
        group_col,
        y,
        div_order=None,
        group_order=None,
        palette=None,
        title=None,
        ylabel=None,
        xlabel=None,
        ax=None,
        fig_width=6,
        aspect=1.0,
        scatter=True,
        sem_fill=True,
        sem_alpha=0.25,
        line_marker="o",
        line_markersize=2,
        annotate=True,
        stats_df=None,          # pre-computed stats (e.g. from calculate_stats_at_timepoints)
        # x-position control
        group_offset_width=0.25,
        scatter_jitter=0.04,
        # clipping / outlier display
        clip_upper=True,
        clip_method="quantile",
        clip_q=0.98,
        clip_k=1.5,
        show_upper_outliers=True,
        # legend
        show_legend=True,
        legend_outside=True,
        legend_loc="upper right"
    ):
        """
        Developmental trajectory plot:
        - mean line
        - SEM fill
        - individual well scatter
        - genotype-specific x-offset within each DIV
        - upper clipped outliers shown as lollipops/triangles
        - centralized stats:
            * 2 groups -> t-test
            * 3+ groups -> one-way ANOVA + Tukey
        """
        d = self._prepare_data(df, [div_col, group_col, y], div_col=div_col)

        div_order = self._resolve_div_order(d, div_col, div_order)
        group_order = self._resolve_group_order(d, group_col, group_order)
        palette = self._resolve_palette(palette)
        div_to_x = {div: i for i, div in enumerate(div_order)}

        if ax is None:
            fig_height = fig_width * aspect
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # ---------------- Colors ----------------
        color_map = self._build_color_map(group_order, palette)

        # ---------------- X offsets ----------------
        n_groups = len(group_order)
        if n_groups == 1:
            offsets = {group_order[0]: 0.0}
        else:
            raw_offsets = np.linspace(-group_offset_width, group_offset_width, n_groups)
            offsets = {g: raw_offsets[i] for i, g in enumerate(group_order)}

        # ---------------- Robust upper clip ----------------
        cap = self._compute_clip_cap(d[y], clip_method, clip_q, clip_k) if clip_upper else None

        # ---------------- Plot each group ----------------
        for g in group_order:
            gdata = d[d[group_col] == g].copy()
            color = color_map[g]
            xoff = offsets[g]

            means = []
            sems = []
            xline = []

            for div in div_order:
                vals = gdata.loc[gdata[div_col] == div, y].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                sems.append(vals.sem() if len(vals) > 1 else 0.0)
                xline.append(div_to_x[div] + xoff)

            means = np.array(means, dtype=float)
            sems = np.array(sems, dtype=float)
            xline = np.array(xline, dtype=float)

            if sem_fill:
                ax.fill_between(
                    xline,
                    means - sems,
                    means + sems,
                    color=color,
                    alpha=sem_alpha,
                    linewidth=0,
                    zorder=1
                )

            ax.plot(
                xline,
                means,
                color=color,
                linewidth=1.0,
                marker=line_marker if line_marker is not None else 'None',
                markersize=line_markersize,
                label=g,
                zorder=3
            )

            if scatter and len(gdata) > 0:
                xvals = gdata[div_col].map(div_to_x).astype(float).values + xoff
                xvals = xvals + np.random.uniform(-scatter_jitter, scatter_jitter, size=len(xvals))

                if cap is not None:
                    main_mask = gdata[y].values <= cap
                else:
                    main_mask = np.ones(len(gdata), dtype=bool)

                ax.scatter(
                    xvals[main_mask],
                    gdata[y].values[main_mask],
                    color=color,
                    alpha=0.6,
                    s=10,
                    edgecolor="black",
                    linewidth=0.25,
                    zorder=2
                )

        # ---------------- Apply clipped y-limit ----------------
        if cap is not None:
            ymin = float(d[y].min())
            pad = 0.08 * max(cap - ymin, 1e-9)
            ax.set_ylim(ymin, cap + pad)

        # ---------------- Upper outliers ----------------
        if show_upper_outliers and cap is not None:
            self._draw_upper_outliers(ax, d, y, div_col, group_col,
                                      div_order, group_order, div_to_x,
                                      offsets, color_map, cap)

        # ---------------- Centralized stats + annotation ----------------
        if annotate:
            if stats_df is None:
                stats_df = self.calculate_stats_by_div(
                    df=d,
                    div_col=div_col,
                    group_col=group_col,
                    y=y,
                    div_order=div_order,
                    group_order=group_order,
                    alpha=0.05
                )
            self._annotate_stats_brackets(ax, stats_df, div_order, div_to_x, offsets)

        # ---------------- Axes / ticks / labels ----------------
        ax.set_xticks(range(len(div_order)))
        ax.set_xticklabels([str(v) for v in div_order])
        ax.set_xlim(-0.5, len(div_order) - 0.5)

        ax.set_xlabel(xlabel if xlabel else div_col, fontweight="bold", fontsize=8)
        ax.set_ylabel(ylabel if ylabel else y.replace("_", " "), fontweight="bold", fontsize=8)
        ax.set_title(title if title else f"{y} trajectory", fontweight="bold", pad=6, fontsize=9)

        ax.tick_params(axis="x", labelsize=6, width=0.8, length=3)
        ax.tick_params(axis="y", labelsize=6, width=0.8, length=3)

        if show_legend:
            if legend_outside:
                ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title_fontsize=8)
            else:
                ax.legend(frameon=False, loc=legend_loc, fontsize=7, title_fontsize=8)

        sns.despine(ax=ax)
        return ax

    def plot_trajectory_with_dotplots(
        self,
        df,
        div_col,
        group_col,
        y,
        selected_divs,
        stats_df=None,
        group_order=None,
        palette=None,
        title=None,
        ylabel=None,
        figsize=(11.69, 6.5),    # A4 landscape default; use ~(2.24, 3.5) for 1/3-column
        height_ratios=(2, 1.5),  # trajectory ~57 %, dot plots ~43 %
        clip_upper=True,
        clip_method="quantile",
        clip_q=0.98,
        clip_k=1.5,
        scatter_jitter=0.12,
        dot_size=22,
        legend_outside=True,
        show_legend=False,       # show legend on trajectory panel
        traj_markers=True,       # show mean data-points on trajectory line
        font_sizes=(6, 9),       # (min, max): min→ticks, mid→labels/titles, max→panel letters
        annotate_traj=False,     # show sig brackets on the trajectory line
        show_xtick_labels=False, # show group name labels on dot-plot x-axes
        show_auc=False,          # append an AUC dot-plot panel after the DIV panels
        unit_cols=None,          # col(s) identifying one trajectory, e.g. ['Chip_ID','Well']
        lmm_results=None,        # dict from test_genotype_lmm(); stamps LRT badge on Panel A
    ):
        """
        A4-landscape two-row publication figure.

        Row 0 (A) — trajectory: line + SEM ribbon (alpha 0.2), no scatter, no markers.
                    Grey bands 0.5 DIV wide at selected_divs; sig brackets above them.
        Row 1 (B/C/D) — one dot-plot per selected DIV.
                    Scatter + median line + IQR bar (Q1–Q3, coloured).
                    Shared y-axis; single y-label centred on the row.
                    Ticks/labels only on leftmost panel.

        Parameters
        ----------
        selected_divs : list of int — e.g. [13, 20, 27]
        stats_df      : from calculate_stats_at_timepoints(); computed internally if None
        figsize       : (w, h) in inches — default A4 landscape (11.69 × 6.5)

        Returns
        -------
        fig, ax_traj, list[ax_dot]
        """
        from matplotlib.gridspec import GridSpec

        d = self._prepare_data(df, [div_col, group_col, y], div_col=div_col)
        group_order   = self._resolve_group_order(d, group_col, group_order)
        palette       = self._resolve_palette(palette)
        selected_divs = [int(v) for v in selected_divs]
        n_sel         = len(selected_divs)

        color_map = self._build_color_map(group_order, palette)

        fs_tick  = font_sizes[0]
        fs_label = font_sizes[0] + (font_sizes[1] - font_sizes[0]) * 2 // 3  # e.g. 8 for (6,9)
        fs_panel = font_sizes[1]

        rc_overrides = {
            "font.family":      "sans-serif",
            "font.sans-serif":  ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "svg.fonttype":     "none",
            "pdf.fonttype":     42,
            "axes.labelsize":   fs_label,
            "axes.labelweight": "bold",
            "xtick.labelsize":  fs_tick,
            "ytick.labelsize":  fs_tick,
        }

        with plt.rc_context(rc_overrides):
            # ── Figure / GridSpec ──────────────────────────────────────────
            _has_auc = show_auc and unit_cols is not None
            n_cols   = n_sel + (1 if _has_auc else 0)
            fig = plt.figure(figsize=figsize)
            _right = 0.87 if (show_legend and legend_outside) else 0.97
            gs  = GridSpec(
                2, n_cols,
                figure=fig,
                height_ratios=list(height_ratios),
                hspace=0.35,
                wspace=0.05,      # panels share y → nearly zero gap
                left=0.08, right=_right,
                top=0.93, bottom=0.10,
            )
            ax_traj  = fig.add_subplot(gs[0, :])
            dot_axes = [fig.add_subplot(gs[1, i]) for i in range(n_sel)]

            # y-range enforced explicitly after all panels are drawn (see below)

            # ── Row 0: trajectory (panel A) ────────────────────────────────
            self.plot_line_sem_by_div(
                df=df,
                div_col=div_col,
                group_col=group_col,
                y=y,
                group_order=group_order,
                palette=palette,
                title="",          # panel label added below
                ylabel=ylabel or y.replace("_", " "),
                ax=ax_traj,
                scatter=False,     # no individual well dots on trajectory
                sem_alpha=0.2,
                line_marker="o" if traj_markers else None,
                line_markersize=3 if traj_markers else 0,
                show_upper_outliers=False,
                annotate=annotate_traj and (stats_df is not None),
                stats_df=stats_df if annotate_traj else None,
                clip_upper=clip_upper,
                clip_method=clip_method,
                clip_q=clip_q,
                clip_k=clip_k,
                legend_outside=legend_outside,
                show_legend=show_legend,
            )

            # Clear y-axis label — placed via fig.text below to align with dot-plot row
            ax_traj.set_ylabel("")

            # Legend: remove border; hide entirely if show_legend=False
            leg = ax_traj.get_legend()
            if leg is not None:
                if show_legend:
                    leg.set_frame_on(False)
                else:
                    leg.remove()

            # Panel label A — top-left corner, anchored
            ax_traj.text(
                0.01, 1.02, "A",
                transform=ax_traj.transAxes,
                fontsize=fs_panel, fontweight="bold", va="bottom", ha="left",
            )

            # LRT badge from test_genotype_lmm() — bottom-right of trajectory panel
            if lmm_results is not None:
                def _s(p):
                    for t, l in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
                        if p < t: return l
                    return "ns"
                m, i_ = lmm_results["main"], lmm_results["interaction"]
                _badge = (
                    f"Genotype main:   χ²({m['df']}) = {m['chi2']:.2f},  p = {m['p']:.3f}  {_s(m['p'])}\n"
                    f"Genotype × DIV:  χ²({i_['df']}) = {i_['chi2']:.2f},  p = {i_['p']:.3f}  {_s(i_['p'])}"
                )
                ax_traj.text(
                    0.99, 0.03, _badge, transform=ax_traj.transAxes,
                    fontsize=6, va="bottom", ha="right", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
                )

            # Narrow grey bands at selected timepoints (0.5 DIV wide)
            div_order_all = self._resolve_div_order(d, div_col, None)
            div_to_x = {dv: i for i, dv in enumerate(div_order_all)}
            for dv in selected_divs:
                if dv in div_to_x:
                    ax_traj.axvspan(
                        div_to_x[dv] - 0.25, div_to_x[dv] + 0.25,
                        color="grey", alpha=0.10, zorder=0, lw=0,
                    )

            # ── Shared y-range for dot plots (from selected timepoints only) ─
            sel_vals = d[d[div_col].isin(selected_divs)][y].dropna()
            y_dot_min = float(sel_vals.min()) if len(sel_vals) > 0 else 0.0
            y_dot_cap = float(sel_vals.quantile(clip_q)) if clip_upper and len(sel_vals) > 0 else float(sel_vals.max())
            y_dot_pad = 0.05 * max(y_dot_cap - y_dot_min, 1e-9)
            y_dot_max = y_dot_cap + y_dot_pad

            # ── Row 1: dot plots (panels B / C / D …) ─────────────────────
            panel_labels = list("BCDE") + [chr(ord("F") + i) for i in range(max(0, n_sel - 4))]

            for i, div_val in enumerate(selected_divs):
                ax       = dot_axes[i]
                div_data = d[d[div_col] == div_val]
                label    = panel_labels[i]

                for j, g in enumerate(group_order):
                    vals  = div_data.loc[div_data[group_col] == g, y].dropna()
                    color = color_map[g]
                    plot_vals = vals[vals <= y_dot_cap] if clip_upper else vals

                    # Jittered scatter
                    if len(plot_vals) > 0:
                        xs = np.full(len(plot_vals), j) + np.random.uniform(
                            -scatter_jitter, scatter_jitter, size=len(plot_vals),
                        )
                        ax.scatter(
                            xs, plot_vals,
                            color=color, alpha=0.70, s=dot_size,
                            edgecolor="black", linewidth=0.25, zorder=3,
                        )

                    # Median line + IQR bar (Q1–Q3)
                    if len(vals) >= 3:
                        q1, med, q3 = np.percentile(vals, [25, 50, 75])
                        # IQR vertical bar in group colour
                        ax.plot([j, j], [q1, q3],
                                color=color, linewidth=2.5, alpha=0.75,
                                solid_capstyle="butt", zorder=4)
                        # Median horizontal bar in black, wider
                        ax.plot([j - 0.22, j + 0.22], [med, med],
                                color="black", linewidth=2.0,
                                solid_capstyle="round", zorder=5)
                    elif len(vals) > 0:
                        ax.plot([j - 0.22, j + 0.22], [float(vals.mean()), float(vals.mean())],
                                color="black", linewidth=2.0,
                                solid_capstyle="round", zorder=5)

                # Panel label B/C/D — top-left corner, anchored
                ax.text(
                    0.01, 1.02, label,
                    transform=ax.transAxes,
                    fontsize=fs_panel, fontweight="bold", va="bottom", ha="left",
                )

                # Title: DIV subtitle, one step below panel letter
                ax.set_title(f"DIV {div_val}", fontsize=fs_label, fontweight="normal", pad=4)

                # x-axis: horizontal labels, no rotation
                ax.set_xticks(range(len(group_order)))
                ax.set_xticklabels(
                    group_order if show_xtick_labels else [""] * len(group_order),
                    fontsize=fs_tick, rotation=0, ha="center",
                )
                ax.set_xlim(-0.6, len(group_order) - 0.4)
                ax.tick_params(axis="x", labelsize=fs_tick, width=0.8, length=3)

                # y-axis: ticks and label only on leftmost panel
                ax.set_ylim(y_dot_min, y_dot_max)
                ax.tick_params(axis="y", labelsize=fs_tick, width=0.8, length=3)
                if i > 0:
                    ax.tick_params(axis="y", left=False, labelleft=False)

                sns.despine(ax=ax)

                # Significance brackets
                if stats_df is not None and not stats_df.empty:
                    div_sig = stats_df[
                        (stats_df["DIV"] == div_val) & (stats_df["reject"] == True)
                    ]
                    self._annotate_dotplot_sig(ax, div_sig, group_order)

            # Enforce shared y-range across all DIV panels post-annotation
            _final_ymax = max(ax.get_ylim()[1] for ax in dot_axes)
            for ax in dot_axes:
                ax.set_ylim(y_dot_min, _final_ymax)

            # y-labels for both rows via fig.text at same x (aligns left edges)
            _ylabel_str = ylabel or y.replace("_", " ")
            pos_t  = ax_traj.get_position()
            pos_b  = dot_axes[0].get_position()
            _x_lbl = pos_b.x0 - 0.055
            fig.text(_x_lbl, (pos_t.y0 + pos_t.y1) / 2, _ylabel_str,
                     ha="right", va="center", rotation=90,
                     fontsize=fs_label, fontweight="bold")
            fig.text(_x_lbl, (pos_b.y0 + pos_b.y1) / 2, _ylabel_str,
                     ha="right", va="center", rotation=90,
                     fontsize=fs_label, fontweight="bold")

            # ── AUC panel (optional last panel) ───────────────────────────
            ax_auc = None
            if _has_auc:
                unit_key_cols = [unit_cols] if isinstance(unit_cols, str) else list(unit_cols)
                auc_rows = []
                for _key, grp in d.groupby(unit_key_cols):
                    g_val = grp[group_col].iloc[0]
                    sg    = grp.sort_values(div_col)
                    auc_rows.append({
                        group_col: g_val,
                        "AUC": float(np.trapz(sg[y].values, sg[div_col].values)),
                    })
                auc_df_plot = pd.DataFrame(auc_rows)
                auc_stats   = self._compute_auc_stats(auc_df_plot, group_col, "AUC", group_order)

                ax_auc      = fig.add_subplot(gs[1, n_sel])
                auc_label   = (panel_labels[n_sel]
                               if n_sel < len(panel_labels)
                               else chr(ord("B") + n_sel))

                auc_all = auc_df_plot["AUC"].dropna()
                auc_cap = (float(auc_all.quantile(clip_q))
                           if clip_upper and len(auc_all) > 0
                           else float(auc_all.max()))
                auc_ymin = float(auc_all.min())
                auc_pad  = 0.10 * max(auc_cap - auc_ymin, 1e-9)

                for j, g in enumerate(group_order):
                    vals  = auc_df_plot.loc[auc_df_plot[group_col] == g, "AUC"].dropna()
                    color = color_map[g]
                    pvals = vals[vals <= auc_cap] if clip_upper else vals
                    if len(pvals) > 0:
                        xs = np.full(len(pvals), j) + np.random.uniform(
                            -scatter_jitter, scatter_jitter, size=len(pvals),
                        )
                        ax_auc.scatter(xs, pvals, color=color, alpha=0.70, s=dot_size,
                                       edgecolor="black", linewidth=0.25, zorder=3)
                    if len(vals) >= 3:
                        q1, med, q3 = np.percentile(vals, [25, 50, 75])
                        ax_auc.plot([j, j], [q1, q3], color=color, linewidth=2.5,
                                    alpha=0.75, solid_capstyle="butt", zorder=4)
                        ax_auc.plot([j - 0.22, j + 0.22], [med, med], color="black",
                                    linewidth=2.0, solid_capstyle="round", zorder=5)
                    elif len(vals) > 0:
                        ax_auc.plot([j - 0.22, j + 0.22], [float(vals.mean())] * 2,
                                    color="black", linewidth=2.0, solid_capstyle="round", zorder=5)

                ax_auc.text(-0.08, 1.08, auc_label, transform=ax_auc.transAxes,
                            fontsize=fs_panel, fontweight="bold",
                            va="top", ha="right")
                ax_auc.set_title("AUC", fontsize=fs_label,
                                 fontweight="normal", pad=4)
                ax_auc.set_xticks(range(len(group_order)))
                ax_auc.set_xticklabels(
                    group_order if show_xtick_labels else [""] * len(group_order),
                    fontsize=fs_tick, rotation=0, ha="center",
                )
                ax_auc.set_xlim(-0.6, len(group_order) - 0.4)
                ax_auc.tick_params(axis="x", labelsize=fs_tick, width=0.8, length=3)
                ax_auc.set_ylim(auc_ymin - auc_pad, auc_cap + auc_pad)
                ax_auc.tick_params(axis="y", labelsize=fs_tick, width=0.8, length=3)
                ax_auc.set_ylabel("AUC", fontsize=fs_label, fontweight="bold")
                sns.despine(ax=ax_auc)

                if not auc_stats.empty:
                    self._annotate_dotplot_sig(
                        ax_auc, auc_stats[auc_stats["reject"]], group_order,
                    )

                dot_axes = dot_axes + [ax_auc]

        return fig, ax_traj, dot_axes

    def plot_bars_by_div(
        self,
        df,
        div_col,
        group_col,
        y,
        div_order=None,
        group_order=None,
        palette=None,
        title=None,
        ax=None,
        fig_width=6,
        aspect=1.0,
        annotate=True,
        # clipping / outliers
        clip_upper=True,
        clip_method="quantile",
        clip_k=1.5,
        clip_q=0.98,
        show_upper_outliers=True,
        # legend control
        legend_loc="upper right",
        legend_outside=False
    ):
        """
        Grouped bars by DIV with scatter.
        Centralized stats:
            * 2 groups -> t-test
            * 3+ groups -> one-way ANOVA + Tukey
        """
        data = self._prepare_data(df, [div_col, group_col, y], div_col=div_col)

        div_order = self._resolve_div_order(data, div_col, div_order)
        group_order = self._resolve_group_order(data, group_col, group_order)
        palette = self._resolve_palette(palette)

        if ax is None:
            fig_height = fig_width * aspect
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.barplot(
            data=data,
            x=div_col, y=y,
            hue=group_col,
            order=div_order,
            hue_order=group_order,
            palette=palette,
            errorbar="se",
            capsize=0.10,
            alpha=0.75,
            edgecolor='black',
            linewidth=0.8,
            ax=ax
        )

        sns.stripplot(
            data=data,
            x=div_col, y=y,
            hue=group_col,
            order=div_order,
            hue_order=group_order,
            dodge=True,
            jitter=0.15,
            alpha=0.75,
            size=5,
            edgecolor='black',
            linewidth=0.3,
            ax=ax
        )

        # ---- legend cleanup ----
        handles, labels = ax.get_legend_handles_labels()
        wanted = [str(g) for g in group_order]
        kept_handles, kept_labels, seen = [], [], set()
        for h, lab in zip(handles, labels):
            if lab in wanted and lab not in seen:
                kept_handles.append(h)
                kept_labels.append(lab)
                seen.add(lab)

        if legend_outside:
            ax.legend(
                kept_handles, kept_labels, title=group_col,
                frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left"
            )
        else:
            ax.legend(kept_handles, kept_labels, title=group_col, frameon=False, loc=legend_loc)

        # ---- clip cap ----
        cap = self._compute_clip_cap(data[y], clip_method, clip_q, clip_k) if clip_upper else None
        if cap is not None:
            ymin = float(data[y].min())
            pad  = 0.06 * max(cap - ymin, 1e-9)
            ax.set_ylim(ymin, cap + pad)

        # ---- upper outliers (bar-plot variant: bar-position offsets, larger marker) ----
        if show_upper_outliers and cap is not None:
            n_groups  = len(group_order)
            bar_width = 0.8 / n_groups
            bar_offsets = {g: (i - n_groups / 2 + 0.5) * bar_width
                           for i, g in enumerate(group_order)}
            bar_div_to_x = {div: i for i, div in enumerate(div_order)}
            # use a black color_map for bars (bar outliers use black, not group colour)
            black_map = {g: "black" for g in group_order}
            self._draw_upper_outliers(ax, data, y, div_col, group_col,
                                      div_order, group_order, bar_div_to_x,
                                      bar_offsets, black_map, cap,
                                      stem_frac=0.06, marker_size=42)

        # ---- centralized stats annotations ----
        if annotate:
            stats_df = self.calculate_stats_by_div(
                df=data,
                div_col=div_col,
                group_col=group_col,
                y=y,
                div_order=div_order,
                group_order=group_order,
                alpha=0.05
            )

            # build offsets to match grouped bars
            n_groups = len(group_order)
            bar_width = 0.8 / n_groups
            offsets = {
                g: (i - n_groups / 2 + 0.5) * bar_width
                for i, g in enumerate(group_order)
            }
            div_to_x = {div: i for i, div in enumerate(div_order)}

            self._annotate_stats_brackets(ax, stats_df, div_order, div_to_x, offsets)

        ax.set_title(title if title else f"{y} by {div_col}", fontweight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(y.replace("_", " "), fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        sns.despine(ax=ax)

        return ax

    # --------------------------
    # wrapper: save pdf
    # --------------------------
    def save_pdf(
        self,
        df,
        y,
        mode="by_div",               # "by_div", "by_group", or "at_timepoints"
        div_col="DIV",
        group_col="NeuronType",
        x=None,                      # only for mode="by_group"
        selected_divs=None,          # only for mode="at_timepoints"; e.g. [13, 20, 27]
        div_order=None,
        group_order=None,
        palette=None,
        title=None,
        filename="Report.pdf",
        pdf_obj=None,
        # plot options forwarded
        annotate=True,
        clip_upper=True,
        clip_method="quantile",
        clip_q=0.98,
        clip_k=1.5,
        show_upper_outliers=True,
        legend_outside=False
    ):
        """
        Orchestrator:
        - runs centralized stats
        - makes plot
        - adds stats table
        - saves a single PDF page

        Stats rule:
        - 2 groups -> t-test
        - 3+ groups -> one-way ANOVA + Tukey
        """
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        palette = self._resolve_palette(palette)

        # ---- compute stats ----
        if mode == "by_div":
            stats_df = self.calculate_stats_by_div(
                df=df,
                div_col=div_col,
                group_col=group_col,
                y=y,
                div_order=div_order,
                group_order=group_order,
                alpha=0.05
            )
        elif mode == "by_group":
            if x is None:
                raise ValueError("For mode='by_group', you must pass x=<group column> (e.g., 'NeuronType').")
            stats_df = self.calculate_stats(
                df=df,
                x=x,
                y=y,
                order=group_order,
                alpha=0.05
            )
        elif mode == "at_timepoints":
            if selected_divs is None:
                raise ValueError(
                    "mode='at_timepoints' requires selected_divs=[13, 20, 27] "
                    "(or your biologically motivated DIVs)."
                )
            stats_df = self.calculate_stats_at_timepoints(
                df=df,
                div_col=div_col,
                group_col=group_col,
                y=y,
                selected_divs=selected_divs,
                group_order=group_order,
                alpha=0.05
            )
        else:
            raise ValueError("mode must be 'by_div', 'by_group', or 'at_timepoints'")

        # ---- build fig ----
        fig = plt.figure(figsize=(12, 11))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0])
        ax_plot = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis("off")

        # ---- plot ----
        if mode == "by_div":
            self.plot_bars_by_div(
                df=df,
                div_col=div_col,
                group_col=group_col,
                y=y,
                div_order=div_order,
                group_order=group_order,
                palette=palette,
                title=title,
                ax=ax_plot,
                annotate=annotate,
                clip_upper=clip_upper,
                clip_method=clip_method,
                clip_q=clip_q,
                clip_k=clip_k,
                show_upper_outliers=show_upper_outliers,
                legend_outside=legend_outside
            )
        elif mode == "at_timepoints":
            self.plot_line_sem_by_div(
                df=df,
                div_col=div_col,
                group_col=group_col,
                y=y,
                group_order=group_order,
                palette=palette,
                title=title,
                ax=ax_plot,
                annotate=annotate,
                stats_df=stats_df,
                clip_upper=clip_upper,
                clip_method=clip_method,
                clip_q=clip_q,
                clip_k=clip_k,
                show_upper_outliers=show_upper_outliers,
                legend_outside=legend_outside,
            )
        else:
            self.plot_bars(
                df=df,
                x=x,
                y=y,
                order=group_order,
                palette=palette,
                title=title,
                ax=ax_plot,
                annotate=annotate
            )

        # ---- table ----
        if stats_df is None or len(stats_df) == 0:
            ax_table.text(0.5, 0.5, "No stats to display", ha="center", va="center")
        else:
            # choose columns
            if mode == "at_timepoints":
                cols = [
                    "DIV", "KW_stat", "KW_p",
                    "comparison", "Grp1_Stats", "Grp2_Stats",
                    "W_stat", "raw_p", "q_val", "Sig", "Cohen's d"
                ]
            elif mode == "by_div" and "DIV" in stats_df.columns:
                cols = [
                    "DIV", "Test", "Omnibus Stat", "Omnibus p-val",
                    "Comparison", "Grp1_Stats", "Grp2_Stats",
                    "Stat", "p-val", "Sig", "Cohen's d"
                ]
            else:
                cols = [
                    "Test", "Omnibus Stat", "Omnibus p-val",
                    "Comparison", "Grp1_Stats", "Grp2_Stats",
                    "Stat", "p-val", "Sig", "Cohen's d"
                ]

            # format
            float2_cols = {"Stat", "Omnibus Stat", "KW_stat", "W_stat"}
            sci_cols    = {"p-val", "Omnibus p-val", "KW_p", "raw_p", "q_val"}

            cell_text = []
            for _, row in stats_df.iterrows():
                row_vals = []
                for c in cols:
                    v = row.get(c, "")
                    if isinstance(v, float) and np.isnan(v):
                        row_vals.append("NA")
                    elif c in float2_cols and isinstance(v, (float, np.floating)):
                        row_vals.append(f"{v:.2f}" if np.isfinite(v) else "NA")
                    elif c in sci_cols and isinstance(v, (float, np.floating)):
                        row_vals.append(f"{v:.3e}" if np.isfinite(v) else "NA")
                    elif c == "Cohen's d" and isinstance(v, (float, np.floating)):
                        row_vals.append(f"{v:.2f}" if np.isfinite(v) else "NA")
                    else:
                        row_vals.append(str(v))
                cell_text.append(row_vals)

            table = ax_table.table(
                cellText=cell_text,
                colLabels=cols,
                loc="center",
                cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.05, 1.55)

            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_text_props(weight="bold")

        # ---- layout control ----
        right = 0.82 if legend_outside else 0.95
        fig.subplots_adjust(top=0.92, bottom=0.05, left=0.07, right=right, hspace=0.15)

        # ---- save ----
        if pdf_obj is not None:
            pdf_obj.savefig(fig)
        else:
            with PdfPages(filename) as pdf:
                pdf.savefig(fig)
            print(f"✅ PDF saved to: {filename}")

        plt.close(fig)