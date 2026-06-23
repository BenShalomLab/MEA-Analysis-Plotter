#!/usr/bin/env python3
"""
qc_threshold_pipeline.py

Two-stage MEA well QC pipeline:
  Step 1 — Channel Occupancy QC (CSV-based)
  Step 2 — Active Area % Tiered QC (Excel-based, KDE auto-thresholding)

Run as a script, or import and call the relevant sections from a notebook.
Update the CONFIG block below for each experiment.
"""

import pandas as pd
from scipy.signal import argrelmin
from scipy.stats import gaussian_kde
import numpy as np
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── UPDATE THESE FOR EACH EXPERIMENT ──────────────────────────────────────────
EXCEL_PATH    = "/mnt/disk15tb/sabadnoor/CDKL5_Dec2025/MaxTwo MEA Primary Neurons Assays.xlsx"
SHEET_NAME    = "CDKL5-R59X_T14_12152025_PS"
FIGURES_PATH  = "/mnt/disk15tb/sabadnoor/CDKL5_Dec2025/figures/"
CSV_PATH      = "/mnt/disk15tb/sabadnoor/CDKL5_Dec2025/final_merged_mea_data.csv"
DIV_COL       = 'DIV'
UNITS_COL     = 'diag_n_units'   # CDKL5 uses 'diag_n_units', Folic Acid uses 'n_units'
PEAK_DIV_MIN  = 15                # peak activity window
PEAK_DIV_MAX  = 22
# ──────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# STEP 1 — Channel Occupancy QC
# ------------------------------------------------------------------------------
# The MaxTwo chip has 1024 electrodes. Channel occupancy = the % of electrodes
# that detected at least one sorted unit. A healthy culture should have activity
# distributed across a large fraction of the array. Low occupancy means the
# culture is too sparse to form a meaningful network and should be excluded.
#
# Thresholds:
#   DIV < 15  → always pass (cultures are still developing, low occupancy normal)
#   DIV >= 15, occupancy < 40% → FAIL (sparse_culture)
#       40% is the minimum density for distributed network formation across array
#   DIV >= 21, nb_count == 0 AND bl_count == 0 → FAIL (no_activity)
#       By DIV 21 a healthy culture must show synchronized network bursting.
#       Zero bursts at this stage means the culture has failed to mature.
# ==============================================================================

df = pd.read_csv(CSV_PATH)

# Fix path parsing if Well column was not extracted correctly from file paths.
# This happens when the CSV was generated with a different folder structure —
# the Well field ends up containing 'network_results.json' instead of the well ID.
if df['Well'].str.contains('network_results').any():
    ANCHOR = SHEET_NAME.replace('_SA', '').replace('_ARedit', '')
    def parse_path(path):
        parts = path.replace("\\", "/").split("/")
        try:
            idx = next(i for i, p in enumerate(parts) if ANCHOR in p)
            return pd.Series({
                'Date':    parts[idx + 1],
                'Chip_ID': parts[idx + 2],
                'Run':     parts[idx + 4],
                'Well':    parts[idx + 5]
            })
        except (StopIteration, IndexError):
            return pd.Series({'Date': None, 'Chip_ID': None, 'Run': None, 'Well': None})
    parsed = df['full_path'].apply(parse_path)
    df['Date']    = parsed['Date']
    df['Chip_ID'] = parsed['Chip_ID']
    df['Run']     = parsed['Run']
    df['Well']    = parsed['Well']

df['channel_occupancy'] = df[UNITS_COL] / 1024 * 100

def qc_flag(row):
    # Early timepoints — culture still developing, QC not yet meaningful
    if row[DIV_COL] < 15:
        return pd.Series({'QC_pass': True, 'QC_fail_reason': ''})
    # Sparse culture — too few electrodes active to represent a distributed network
    if row['channel_occupancy'] < 40:
        return pd.Series({'QC_pass': False, 'QC_fail_reason': 'sparse_culture (<40% electrodes active)'})
    # No network activity — mature culture should be bursting by DIV 21
    if row[DIV_COL] >= 21 and row['nb_count'] == 0 and row['bl_count'] == 0:
        return pd.Series({'QC_pass': False, 'QC_fail_reason': 'no_activity (zero bursts at DIV>=21)'})
    return pd.Series({'QC_pass': True, 'QC_fail_reason': ''})

qc_results = df.apply(qc_flag, axis=1)
df = pd.concat([df, qc_results], axis=1)

total  = len(df)
passed = df['QC_pass'].sum()
failed = total - passed

print("=" * 60)
print("STEP 1 — Channel Occupancy QC")
print("  Metric: n_units (or diag_n_units) / 1024 * 100 = % of 1024 electrodes active")
print("  Cutoff: < 40% = sparse culture (excluded)")
print("          DIV >= 21 with zero bursts = no activity (excluded)")
print("=" * 60)
print(f"Total rows:  {total}")
print(f"QC pass:     {passed} ({round(100*passed/total, 1)}%)")
print(f"QC fail:     {failed} ({round(100*failed/total, 1)}%)")
print(f"\nFailed wells:")
print(df[df['QC_pass'] == False][
    ['Well', DIV_COL, 'channel_occupancy', 'QC_fail_reason']
].sort_values(['Well', DIV_COL]).to_string(index=False))

df_qc = df[df['QC_pass'] == True].copy()

# Summarise CSV metrics per well at peak DIV window — matches the same evaluation
# window used for Active Area % in Step 2. Occupancy uses mean (stable metric);
# burst counts use median (more robust to a single noisy/artifact recording).
peak_summary = (
    df_qc[df_qc[DIV_COL].between(PEAK_DIV_MIN, PEAK_DIV_MAX)]
    .groupby('Well')
    .agg(
        mean_occupancy=('channel_occupancy', 'mean'),
        mean_nb=('nb_count', 'median'),
        mean_bl=('bl_count', 'median')
    )
    .reset_index()
)


# ==============================================================================
# STEP 2 — Active Area % Tiered QC
# ------------------------------------------------------------------------------
# Channel occupancy alone cannot detect spatially clustered cultures — a well
# can have 50% occupancy but all activity in one corner of the chip. Active
# Area % measures the spatial footprint of activity across the physical array.
# This value is recorded manually in the Excel experiment log.
#
# PEAK_DIV_MIN-PEAK_DIV_MAX is used as the evaluation window because this is
# peak activity for the culture. Neurons reach maximum spatial coverage here
# before naturally declining after DIV 22-25. Evaluating at peak gives the
# fairest picture.
#
# Thresholds are NOT hardcoded — they are found automatically per dataset using
# Kernel Density Estimation (KDE). KDE fits a smooth curve to the distribution
# of Active Area % values across all wells. Natural valleys (dips) in this curve
# represent gaps between clusters of wells. These gaps are used as tier cutoffs
# so the thresholds adapt to each experiment's data rather than relying on a
# fixed number that may not suit every dataset.
#
# Tiers (collapsed to 3):
#   Bad              — below first valley: spatial coverage too low, excluded
#   Use with caution — between first and second valleys: marginal, usable but flagged
#   Good             — above second valley: healthy spatial distribution, suitable for analysis
# ==============================================================================

aa_raw   = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
all_cols = aa_raw.columns.tolist()

# Auto-detect column format — two formats exist across experiments:
#   Format A: W1, W2 ... WN         (CDKL5 — all wells labeled sequentially)
#   Format B: P1W1, P1W2 ... PnWm   (Folic Acid — organized by Plate x Well)
w_cols  = [c for c in all_cols if re.fullmatch(r'W\d+', str(c))]
pw_cols = [c for c in all_cols if re.fullmatch(r'P\d+W\d+', str(c))]

if w_cols:
    print(f"\nDetected Format A: {len(w_cols)} well columns (W1-W{len(w_cols)})")
    aa_data = aa_raw[['DIV.1'] + w_cols].copy()
    aa_data = aa_data.dropna(subset=['DIV.1'])
    aa_data = aa_data[(aa_data['DIV.1'] >= PEAK_DIV_MIN) & (aa_data['DIV.1'] <= PEAK_DIV_MAX)]
    aa_long = aa_data.melt(id_vars='DIV.1', var_name='Well_Excel', value_name='active_area_pct')
    aa_long['active_area_pct'] = pd.to_numeric(aa_long['active_area_pct'], errors='coerce')
    aa_long = aa_long.dropna(subset=['active_area_pct'])
    aa_stats = aa_long.groupby('Well_Excel').agg(
        mean_active_area_peak=('active_area_pct', 'mean')
    ).reset_index()
    aa_stats['Well_Label'] = aa_stats['Well_Excel'].apply(
        lambda x: f"well{str(int(x[1:])-1).zfill(3)}"
    )
    aa_stats = aa_stats.merge(peak_summary, left_on='Well_Label', right_on='Well', how='left')

elif pw_cols:
    print(f"\nDetected Format B: {len(pw_cols)} well columns (PxWy format)")
    aa_data = aa_raw[['DIV.1'] + pw_cols].copy()
    aa_data = aa_data.dropna(subset=['DIV.1'])
    aa_data = aa_data[(aa_data['DIV.1'] >= PEAK_DIV_MIN) & (aa_data['DIV.1'] <= PEAK_DIV_MAX)]
    aa_long = aa_data.melt(id_vars='DIV.1', var_name='Well_Excel', value_name='active_area_pct')
    aa_long['active_area_pct'] = pd.to_numeric(aa_long['active_area_pct'], errors='coerce')
    aa_long = aa_long.dropna(subset=['active_area_pct'])
    # Each P*W* is its own unique well — kept separate so the KDE uses all
    # individual data points, not averages collapsed by well position
    aa_stats = aa_long.groupby('Well_Excel').agg(
        mean_active_area_peak=('active_area_pct', 'mean')
    ).reset_index()
    aa_stats['Well_Label'] = aa_stats['Well_Excel']
    # Map well position (W1→well000 etc.) to join CSV metrics from peak_summary
    aa_stats['Well_pos'] = aa_stats['Well_Excel'].apply(
        lambda x: f"well{str(int(re.search(r'W(\d+)', x).group(1))-1).zfill(3)}"
    )
    aa_stats = aa_stats.merge(peak_summary, left_on='Well_pos', right_on='Well', how='left')

else:
    raise ValueError("Could not detect Active Area % columns. Expected W1/W2 or P1W1/P1W2 format.")

# Fit KDE and find natural valleys (thresholds) in the distribution
values = aa_stats['mean_active_area_peak'].values
kde    = gaussian_kde(values, bw_method=0.15)
x      = np.linspace(0, 100, 1000)
y      = kde(x)

valleys     = argrelmin(y, order=20)[0]
valley_vals = sorted([x[v] for v in valleys])

# Only need 2 thresholds for 3 tiers
bad_threshold     = valley_vals[0] if len(valley_vals) > 0 else float(np.percentile(values, 15))
caution_threshold = valley_vals[1] if len(valley_vals) > 1 else float(np.percentile(values, 50))

def assign_tier(val):
    if val < bad_threshold:         return 'Bad'
    elif val < caution_threshold:   return 'Use with caution'
    else:                           return 'Good'

def assign_explanation(row):
    val  = round(row['mean_active_area_peak'], 1)
    tier = row['QC_tier']
    occ  = f"{round(row['mean_occupancy'], 1)}%" if pd.notna(row.get('mean_occupancy')) else "N/A"
    nb   = round(row['mean_nb'], 1) if pd.notna(row.get('mean_nb')) else 'N/A'
    bl   = round(row['mean_bl'], 1) if pd.notna(row.get('mean_bl')) else 'N/A'

    if tier == 'Bad':
        return (f"Channel occupancy: {occ} | Network bursts (median, DIV {PEAK_DIV_MIN}-{PEAK_DIV_MAX}): {nb} | "
                f"Burst-like events (median): {bl} | "
                f"Active Area: {val}% — below exclusion threshold of {round(bad_threshold,1)}%. Excluded.")
    elif tier == 'Use with caution':
        return (f"Channel occupancy: {occ} | Network bursts (median, DIV {PEAK_DIV_MIN}-{PEAK_DIV_MAX}): {nb} | "
                f"Burst-like events (median): {bl} | "
                f"Active Area: {val}% — above exclusion threshold ({round(bad_threshold,1)}%) "
                f"but below healthy minimum ({round(caution_threshold,1)}%). Usable but flagged for caution.")
    else:
        return (f"Channel occupancy: {occ} | Network bursts (median, DIV {PEAK_DIV_MIN}-{PEAK_DIV_MAX}): {nb} | "
                f"Burst-like events (median): {bl} | "
                f"Active Area: {val}% — above healthy threshold of {round(caution_threshold,1)}%. Suitable for analysis.")

aa_stats['QC_tier']        = aa_stats['mean_active_area_peak'].apply(assign_tier)
aa_stats['QC_explanation'] = aa_stats.apply(assign_explanation, axis=1)

print("\n" + "=" * 60)
print("STEP 2 — Active Area % Tiered QC")
print("  Metric: % of chip spatial area with detected activity (from Excel)")
print(f"  Window: DIV {PEAK_DIV_MIN}-{PEAK_DIV_MAX} (peak activity — fairest evaluation point)")
print("  Method: KDE on distribution of mean Active Area % per well,")
print("          thresholds set at natural valleys in the distribution")
print(f"\n  Valleys found at: {[round(v,1) for v in valley_vals]}%")
print(f"  Bad:              < {round(bad_threshold,1)}%  — spatially clustered, excluded")
print(f"  Use with caution: {round(bad_threshold,1)}% – {round(caution_threshold,1)}%  — marginal spatial coverage")
print(f"  Good:             > {round(caution_threshold,1)}%  — healthy spatial distribution")
print("=" * 60)

for tier in ['Good', 'Use with caution', 'Bad']:
    subset = aa_stats[aa_stats['QC_tier'] == tier].sort_values('mean_active_area_peak', ascending=False)
    print(f"\n{'='*60}")
    print(f"{tier} wells ({len(subset)}):")
    print(f"{'='*60}")
    for _, row in subset.iterrows():
        print(f"  {row['Well_Label']}  ({round(row['mean_active_area_peak'],1)}%)")
        print(f"  → {row['QC_explanation']}")
        print()

# Plot KDE with tier thresholds and individual well lines
os.makedirs(FIGURES_PATH, exist_ok=True)
colors = {'Bad': 'red', 'Use with caution': 'orange', 'Good': 'darkgreen'}
plt.figure(figsize=(12, 5))
plt.plot(x, y, 'b-', linewidth=2, label='KDE distribution')
plt.axvline(bad_threshold,     color='red',    linestyle='--', label=f'Bad < {round(bad_threshold,1)}%')
plt.axvline(caution_threshold, color='orange', linestyle='--', label=f'Good > {round(caution_threshold,1)}%')
for _, row in aa_stats.iterrows():
    plt.axvline(row['mean_active_area_peak'], color=colors[row['QC_tier']], alpha=0.5, linewidth=2)
plt.xlabel(f'Mean Active Area % (DIV {PEAK_DIV_MIN}-{PEAK_DIV_MAX})')
plt.title('QC Tiers — Active Area % Distribution (thresholds from KDE valleys)')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_PATH + 'qc_tiers.png')
print(f"\nPlot saved to {FIGURES_PATH}qc_tiers.png")
