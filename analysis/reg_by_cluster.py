"""
reg_by_cluster.py  —  Section 2
Two-way fixed-effects panel regression by cluster x period.
ln(1+co2) ~ ln(1+urban_frac) + ln(1+height_m)
pixel FE + year FE; SE clustered at pixel level
"""

import sys, warnings
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS

from pathlib import Path
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

V3   = str(DATA_DIR)
V4   = str(DATA_DIR)
OUT  = str(OUT_DIR / "reg_by_cluster.txt")

LOG = open(OUT, 'w', encoding='utf-8')
def log(*a): print(*a); print(*a, file=LOG)

panel  = pd.read_pickle(f'{V3}/panel_v3_clean.pkl')
pixels = pd.read_pickle(f'{V3}/pixels_v3.pkl')
types  = pd.read_pickle(f'{V4}/cluster_types.pkl')

pix_idx = pixels[['row','col']].drop_duplicates().reset_index(drop=True)
pix_idx['pixel_id'] = np.arange(len(pix_idx))
panel = panel.merge(pix_idx[['row','col','pixel_id']], on=['row','col'], how='left')
panel['lco2']      = np.log1p(panel['co2_kt'])
panel['ln_urban']  = np.log1p(panel['urban_frac'])
panel['ln_height'] = np.log1p(panel['height_m'])
panel = panel.merge(types[['row','col','cluster']], on=['row','col'], how='left')

CLUSTERS = ['High-decel', 'Mid-decel', 'Mid-accel', 'Low-accel']
PERIODS  = [('Full', 2001, 2018), ('Early', 2001, 2009), ('Late', 2010, 2018)]
VARS     = ['ln_urban', 'ln_height']

def sig(p):
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '†' if p<0.1 else ''

def run(df):
    if df['pixel_id'].nunique() < 10: return None
    fe = df.set_index(['pixel_id','year'])
    try:
        return PanelOLS(fe['lco2'], fe[VARS],
                        entity_effects=True, time_effects=True,
                        drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    except: return None

log('='*80)
log('Section 2  —  By-cluster panel FE regression')
log('  Dep: ln(1+co2_kt)   Regressors: ln(1+urban_frac)  ln(1+height_m)')
log('  FE: pixel + year    SE: clustered by pixel')
log('='*80)

rows = []
for cl in CLUSTERS:
    sub_c = panel[panel['cluster'] == cl]
    n_pix = sub_c['pixel_id'].nunique()
    log(f'\nCluster: {cl}  (n_pixels={n_pix})')
    log(f"  {'Period':<8}  {'N_obs':>7}  {'R2w':>6}  "
        f"{'β_urban':>8} {'SE':>6} {'t':>6} {'p':>6}    "
        f"{'β_height':>8} {'SE':>6} {'t':>6} {'p':>6}")
    log('  ' + '-'*76)
    for pname, y0, y1 in PERIODS:
        sub = sub_c[(sub_c.year >= y0) & (sub_c.year <= y1)]
        res = run(sub)
        if res is None:
            log(f'  {pname:<8}  n/a')
            continue
        bu = res.params['ln_urban'];  seu = res.std_errors['ln_urban']
        tu = res.tstats['ln_urban'];  pu  = res.pvalues['ln_urban']
        bh = res.params['ln_height']; seh = res.std_errors['ln_height']
        th = res.tstats['ln_height']; ph  = res.pvalues['ln_height']
        log(f'  {pname:<8}  {res.nobs:>7,}  {res.rsquared_within:>6.4f}  '
            f'{bu:>8.4f} {seu:>6.4f} {tu:>6.2f} {pu:>6.4f}{sig(pu):<3}  '
            f'{bh:>8.4f} {seh:>6.4f} {th:>6.2f} {ph:>6.4f}{sig(ph):<3}')
        rows.append(dict(cluster=cl, period=pname, n_pix=n_pix, nobs=res.nobs,
                         r2w=res.rsquared_within,
                         b_urban=bu, se_urban=seu, t_urban=tu, p_urban=pu,
                         b_height=bh, se_height=seh, t_height=th, p_height=ph))

df_out = pd.DataFrame(rows)
df_out.to_csv(f'{V4}/reg_by_cluster.csv', index=False)
log(f'\nSaved: {V4}/reg_by_cluster.csv')
LOG.close()
print(f'Results: {OUT}')
