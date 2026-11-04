"""Active-area QC functions extracted from activity_outlier_filter.ipynb.

Flags are review candidates; they do not establish human inclusion.
"""
import numpy as np
import pandas as pd


def filter_below_floor(
    df: pd.DataFrame,
    value_col: str = 'Active_area',
    floor: float = 30.0,
) -> pd.DataFrame:
    df = df.copy()
    well_max = df.groupby('CHIP_WELL')[value_col].transform('max')
    df['below_floor'] = well_max < floor
    n_wells = df[df['below_floor']]['CHIP_WELL'].nunique()
    print(f'Floor filter (<{floor}%): {n_wells} wells excluded')
    return df


def filter_insufficient_divs(
    df: pd.DataFrame,
    min_divs: int = 4,
) -> pd.DataFrame:
    df = df.copy()
    well_ndivs = df.groupby('CHIP_WELL')['DIV'].transform('nunique')
    df['insufficient_divs'] = well_ndivs < min_divs
    n_wells = df[df['insufficient_divs']]['CHIP_WELL'].nunique()
    print(f'DIV coverage filter (<{min_divs} timepoints): {n_wells} wells excluded')
    return df


def detect_outliers_within_groups(
    df: pd.DataFrame,
    value_col: str = 'Active_area',
    min_n: int = 5,
    z_thresh: float = 3.5,
) -> pd.DataFrame:
    df = df.copy()
    df['is_outlier'] = False
    for (nt, div), group in df.groupby(['NeuronType', 'DIV'], sort=False):
        vals = group[value_col].to_numpy(dtype=float)
        if vals.size < min_n:
            continue
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad <= 0:
            continue
        modz = 0.6745 * (vals - med) / mad
        df.loc[group.index[np.abs(modz) > z_thresh], 'is_outlier'] = True
    print(f'Marked {int(df["is_outlier"].sum())} MAD-based outliers (|modZ| > {z_thresh})')
    return df


def _moving_average(y: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return y
    y = np.asarray(y, dtype=float)
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    return np.convolve(ypad, np.ones(window) / window, mode='valid')


def detect_expected_curve_residuals(
    df: pd.DataFrame,
    value_col: str = 'Active_area',
    late_div_min: int = 14,
    min_div_points: int = 4,
    smooth_window: int = 3,
    resid_modz_thresh: float = 3.5,
    build_expected_from_non_outliers: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df['expected_y']      = np.nan
    df['resid_expected']  = np.nan
    df['is_bad_residual'] = False
    for nt, sub_all in df.groupby('NeuronType', sort=False):
        sub_fit = (
            sub_all[~sub_all['is_outlier']].copy()
            if build_expected_from_non_outliers and 'is_outlier' in df.columns
            else sub_all.copy()
        )
        div_meds = sub_fit.groupby('DIV')[value_col].median()
        if div_meds.shape[0] < min_div_points:
            continue
        divs        = div_meds.index.to_numpy(dtype=int)
        meds_smooth = _moving_average(div_meds.to_numpy(dtype=float), window=smooth_window)
        expected_map = dict(zip(divs, meds_smooth))
        idx_nt  = df['NeuronType'] == nt
        df.loc[idx_nt, 'expected_y'] = df.loc[idx_nt, 'DIV'].map(expected_map)
        has_exp = idx_nt & df['expected_y'].notna()
        df.loc[has_exp, 'resid_expected'] = (
            df.loc[has_exp, value_col] - df.loc[has_exp, 'expected_y']
        )
        cand  = has_exp & (df['DIV'] >= late_div_min)
        resid = df.loc[cand, 'resid_expected'].to_numpy(dtype=float)
        if resid.size < 5:
            continue
        med_r = np.median(resid)
        mad_r = np.median(np.abs(resid - med_r))
        if mad_r <= 0:
            continue
        modz_r = 0.6745 * (resid - med_r) / mad_r
        df.loc[df.loc[cand].index[modz_r < -resid_modz_thresh], 'is_bad_residual'] = True
    print(f'Flagged {int(df["is_bad_residual"].sum())} residual dips '
          f'(late DIV >= {late_div_min}, modZ < -{resid_modz_thresh})')
    return df


def mark_filtered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flag_cols = [c for c in ['below_floor', 'insufficient_divs', 'is_outlier', 'is_bad_residual']
                 if c in df.columns]
    df['to_exclude'] = df[flag_cols].any(axis=1)
    for col in flag_cols:
        print(f'  {col}: {int(df[col].sum())} rows flagged')
    print(f'Total excluded: {df["to_exclude"].sum()} / {len(df)} '
          f'({df["to_exclude"].mean()*100:.1f}%)')
    return df


def check_balance(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    clean = df[~df['to_exclude']]
    bal = (clean.groupby(['NeuronType', 'DIV'])['CHIP_WELL']
           .nunique()
           .unstack('DIV')
           .fillna(0)
           .astype(int))
    low = (bal < min_n) & (bal > 0)
    if low.any().any():
        print(f'WARNING: NeuronType×DIV cells with <{min_n} wells:')
        print(bal[low.any(axis=1)])
    else:
        print(f'Balance OK: all groups >=={min_n} wells per DIV')
    print('\nWells per NeuronType x DIV:')
    print(bal.to_string())
    return bal
