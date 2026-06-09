"""
reg_by_region.py  —  Section 3
(a) Cluster composition by UN sub-region
(b) Two-way FE panel regression by sub-region; full / early / late periods
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

V3  = str(DATA_DIR)
V4  = str(DATA_DIR)
OUT = str(OUT_DIR / "reg_by_region.txt")

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
PERIODS  = [('Full',2001,2018), ('Early',2001,2009), ('Late',2010,2018)]
VARS     = ['ln_urban', 'ln_height']
MIN_PIX  = 20

def sig(p):
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '†' if p<0.1 else ''

def run(df):
    if df['pixel_id'].nunique() < MIN_PIX: return None
    fe = df.set_index(['pixel_id','year'])
    try:
        return PanelOLS(fe['lco2'], fe[VARS],
                        entity_effects=True, time_effects=True,
                        drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    except: return None

# ── (a) Cluster composition by region ────────────────────────────────────────
log('='*70)
log('Section 3a  —  Cluster composition by region')
log('='*70)

snap = panel[panel.year == 2018][['row','col','region','cluster']].drop_duplicates(['row','col'])
comp = snap.groupby(['region','cluster']).size().unstack(fill_value=0)
comp = comp.reindex(columns=CLUSTERS, fill_value=0)
comp['total'] = comp.sum(axis=1)
comp_pct = comp[CLUSTERS].div(comp['total'], axis=0) * 100

region_order = comp['total'].sort_values(ascending=False).index.tolist()

log(f"\n{'Region':<22} {'Total':>6}  " + "  ".join(f"{c:>12}" for c in CLUSTERS))
log('-'*80)
for r in region_order:
    tot = comp.loc[r,'total']
    pcts = "  ".join(f"{comp_pct.loc[r,c]:>11.1f}%" for c in CLUSTERS)
    log(f"{r:<22} {tot:>6}  {pcts}")

comp_pct['total'] = comp['total']
comp_pct.to_csv(f'{V4}/cluster_by_region.csv')
log(f'\nSaved: {V4}/cluster_by_region.csv')

# ── (b) By-region regression ──────────────────────────────────────────────────
log('\n' + '='*70)
log('Section 3b  —  By-region panel FE regression')
log('='*70)

reg_rows = []
for region in region_order:
    sub_r = panel[panel['region'] == region]
    n_pix = sub_r['pixel_id'].nunique()
    if n_pix < MIN_PIX: continue
    log(f'\nRegion: {region}  (n_pixels={n_pix})')
    log(f"  {'Period':<8}  {'N_obs':>7}  {'R2w':>6}  "
        f"{'β_urban':>8} {'t':>6} {'p':>6}    {'β_height':>8} {'t':>6} {'p':>6}")
    log('  ' + '-'*68)
    for pname, y0, y1 in PERIODS:
        sub = sub_r[(sub_r.year >= y0) & (sub_r.year <= y1)]
        res = run(sub)
        if res is None:
            log(f'  {pname:<8}  n/a'); continue
        bu = res.params['ln_urban'];  tu = res.tstats['ln_urban'];  pu = res.pvalues['ln_urban']
        bh = res.params['ln_height']; th = res.tstats['ln_height']; ph = res.pvalues['ln_height']
        log(f'  {pname:<8}  {res.nobs:>7,}  {res.rsquared_within:>6.4f}  '
            f'{bu:>8.3f} {tu:>6.2f} {pu:>6.4f}{sig(pu):<3}  '
            f'{bh:>8.3f} {th:>6.2f} {ph:>6.4f}{sig(ph):<3}')
        reg_rows.append(dict(region=region, period=pname, n_pix=n_pix, nobs=res.nobs,
                             r2w=res.rsquared_within,
                             b_urban=bu, t_urban=tu, p_urban=pu,
                             b_height=bh, t_height=th, p_height=ph))

pd.DataFrame(reg_rows).to_csv(f'{V4}/reg_by_region.csv', index=False)
log(f'\nSaved: {V4}/reg_by_region.csv')
LOG.close()
print(f'Results: {OUT}')
