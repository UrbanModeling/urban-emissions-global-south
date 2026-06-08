"""
reg_by_agglom.py  —  Section 4
城市群内两路固定效应面板回归（≥30 pixels）
通过 8-邻域像素邻接识别城市群
"""

import sys, warnings
import numpy as np, pandas as pd
import networkx as nx
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
OUT = str(OUT_DIR / "reg_by_agglom.txt")
MIN_PIX = 30

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

city_info = pixels[['city_id','city_name','country','region']].drop_duplicates('city_id').set_index('city_id')
PERIODS = [('Full',2001,2018), ('Early',2001,2009), ('Late',2010,2018)]
VARS    = ['ln_urban','ln_height']

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

# ── Build adjacency graph ─────────────────────────────────────────────────────
log('Building agglomeration graph...')
rc_to_city = {(r['row'], r['col']): r['city_id'] for _, r in pixels.iterrows()}
pixel_set  = set(rc_to_city.keys())
G = nx.Graph()
G.add_nodes_from(pixels['city_id'].unique())
for (r, c), cid in rc_to_city.items():
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr == dc == 0: continue
            nb = (r+dr, c+dc)
            if nb in pixel_set:
                nb_cid = rc_to_city[nb]
                if nb_cid != cid:
                    G.add_edge(cid, nb_cid)

components = sorted([c for c in nx.connected_components(G) if len(c) >= 2],
                    key=len, reverse=True)
log(f'  Total agglomerations (≥2 cities): {len(components)}')

# ── Agglomeration labels (reuse from v3 where possible) ──────────────────────
AGGLOM_LABELS = {
    0:'Dhaka-Comilla', 1:'Yangtze Delta', 2:'Pearl River Delta',
    3:'Nile Delta', 4:'Kerala Coast (N)', 5:'Johannesburg',
    6:'Quanzhou-Xiamen', 7:'Kerala Coast (S)', 8:'Jakarta',
    9:'Surabaya', 10:'Bandung', 11:'Tianjin', 12:'Chongqing',
    13:'Wenzhou', 14:'Onitsha', 15:'Changsha', 16:'Gujranwala',
    17:'Asansol-Dhanbad', 18:'Delhi-Meerut', 19:'Manila',
    20:'Mumbai', 21:'Kolkata', 22:'Sao Paulo', 23:'Mexico City',
    24:'Istanbul', 25:'Buenos Aires', 26:'Lagos', 27:'Rio de Janeiro',
    28:'Kinshasa', 29:'Chengdu',
}

# ── Run regressions ───────────────────────────────────────────────────────────
log('\n' + '='*80)
log('Section 4  —  By-agglomeration panel FE regression  (min_pix=30)')
log('='*80)

p2018 = panel[panel.year==2018]
snap  = panel[panel.year==2018][['row','col','cluster']].drop_duplicates(['row','col'])
snap  = snap.merge(pixels[['row','col','city_id']], on=['row','col'], how='left')

reg_rows = []
for i, comp in enumerate(components):
    cids   = list(comp)
    n_pix  = p2018[p2018['city_id'].isin(cids)].shape[0]
    if n_pix < MIN_PIX: continue
    label  = AGGLOM_LABELS.get(i, f'Agglom#{i+1}')
    region = city_info.loc[city_info.index.isin(cids), 'region'].mode()[0]
    n_cities = len(comp)

    # dominant cluster
    snap_ag = snap[snap['city_id'].isin(cids)]
    dom_cl  = snap_ag['cluster'].value_counts().idxmax() if len(snap_ag) else 'n/a'
    dom_pct = snap_ag['cluster'].value_counts(normalize=True).max() * 100 if len(snap_ag) else 0

    sub_ag = panel[panel['city_id'].isin(cids)]

    log(f'\n{label}  ({region}, {n_cities} cities, {n_pix} pixels)')
    log(f'  Dominant cluster: {dom_cl} ({dom_pct:.0f}%)')
    log(f"  {'Period':<8}  {'N_obs':>7}  {'R2w':>6}  "
        f"{'β_urban':>8} {'t':>6} {'p':>6}    {'β_height':>8} {'t':>6} {'p':>6}")
    log('  ' + '-'*68)

    for pname, y0, y1 in PERIODS:
        sub = sub_ag[(sub_ag.year >= y0) & (sub_ag.year <= y1)]
        res = run(sub)
        if res is None:
            log(f'  {pname:<8}  n/a'); continue
        bu = res.params['ln_urban'];  tu = res.tstats['ln_urban'];  pu = res.pvalues['ln_urban']
        bh = res.params['ln_height']; th = res.tstats['ln_height']; ph = res.pvalues['ln_height']
        log(f'  {pname:<8}  {res.nobs:>7,}  {res.rsquared_within:>6.4f}  '
            f'{bu:>8.3f} {tu:>6.2f} {pu:>6.4f}{sig(pu):<3}  '
            f'{bh:>8.3f} {th:>6.2f} {ph:>6.4f}{sig(ph):<3}')
        reg_rows.append(dict(
            agglom=label, region=region, n_cities=n_cities, n_pix=n_pix,
            dom_cluster=dom_cl, dom_pct=round(dom_pct,1),
            period=pname, nobs=res.nobs, r2w=res.rsquared_within,
            b_urban=bu, t_urban=tu, p_urban=pu,
            b_height=bh, t_height=th, p_height=ph))

pd.DataFrame(reg_rows).to_csv(f'{V4}/reg_by_agglom.csv', index=False)
log(f'\nSaved: {V4}/reg_by_agglom.csv')
LOG.close()
print(f'Results: {OUT}')
