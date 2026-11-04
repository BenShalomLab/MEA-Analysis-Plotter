"""Check the reusable module against the original notebook on synthetic data."""
import ast
import contextlib
import io
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import pandas.testing as pdt
import activity_qc as module

ROOT=Path(__file__).resolve().parents[1]
NAMES=['filter_below_floor','filter_insufficient_divs','detect_outliers_within_groups','_moving_average','detect_expected_curve_residuals','mark_filtered','check_balance']

def notebook_functions():
    found={}
    for cell in json.loads((ROOT/'activity_outlier_filter.ipynb').read_text())['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef) and node.name in NAMES:found[node.name]=node
    assert set(found)==set(NAMES)
    ns={'np':np,'pd':pd};exec(compile(ast.Module(body=list(found.values()),type_ignores=[]),'synthetic_parity','exec'),ns)
    return ns

def fixture():
    rows=[]
    for group in ['A','B']:
        for well in range(6):
            for div in [4,7,14,21,28]:
                value=35+well*4+div*.3+(9 if group=='B' else 0)
                if well==0 and div==21:value=2
                if well==5 and div==28:value=98
                rows.append({'NeuronType':group,'CHIP_WELL':group+str(well),'DIV':div,'Active_area':value})
    return pd.DataFrame(rows)

def run(ns,df,changed=False):
    df=ns['filter_below_floor'](df,floor=40 if changed else 30)
    df=ns['filter_insufficient_divs'](df,min_divs=4)
    df=ns['detect_outliers_within_groups'](df,min_n=5,z_thresh=2.5 if changed else 3.5)
    df=ns['detect_expected_curve_residuals'](df,late_div_min=14,min_div_points=4,smooth_window=3,resid_modz_thresh=3.5)
    df=ns['mark_filtered'](df)
    return df,ns['check_balance'](df,min_n=3)

class Parity(unittest.TestCase):
    def test_module_matches_notebook_for_defaults_and_changed_settings(self):
        original=notebook_functions();new={n:getattr(module,n) for n in NAMES}
        for changed in [False,True]:
            with contextlib.redirect_stdout(io.StringIO()):
                before,balance_before=run(original,fixture(),changed);after,balance_after=run(new,fixture(),changed)
            pdt.assert_frame_equal(before,after);pdt.assert_frame_equal(balance_before,balance_after)
    def test_small_groups_and_zero_mad_remain_not_flagged(self):
        df=pd.DataFrame({'NeuronType':['A']*5,'CHIP_WELL':list('abcde'),'DIV':[14]*5,'Active_area':[50]*5})
        with contextlib.redirect_stdout(io.StringIO()):out=module.detect_outliers_within_groups(df)
        self.assertFalse(out['is_outlier'].any())
        with contextlib.redirect_stdout(io.StringIO()):out=module.detect_outliers_within_groups(df.iloc[:4])
        self.assertFalse(out['is_outlier'].any())
if __name__=='__main__':unittest.main()
