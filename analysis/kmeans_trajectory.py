"""
kmeans_trajectory.py
K-means clustering (k=4), no standardisation, log-transformed features only.

Features:
  mean_lco2   = mean(log1p(co2_kt), 2001-2018)   emission level
  dmean_lco2  = mean_late - mean_early            change in mean level
  dslope_lco2 = slope_late - slope_early          change in trend slope (acceleration)

Clusters (ordered by mean emission level, descending):
  High-decel   high emission, decelerating  (+57% -> +16%)
  Mid-decel    mid  emission, decelerating  (+61% -> +40%)
  Mid-accel    mid  emission, accelerating  (+45% -> +67%)
  Low-accel    low  emission, accelerating  (+35% -> +69%)

Outputs:
  output/kmeans_trajectory.txt
  data/processed/cluster_types.pkl
  data/processed/cluster_traj.pkl
"""

import sys, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import linregress

from pathlib import Path
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

V3_DATA  = str(DATA_DIR)
OUT_DATA = str(DATA_DIR)
OUT_TXT  = str(OUT_DIR / "kmeans_trajectory.txt")

LOG = open(OUT_TXT, 'w', encoding='utf-8')
def log(*args):
    print(*args)
    print(*args, file=LOG)

# ── Load ──────────────────────────────────────────────────────────────────────
log('Loading panel data from proj-v3...')
panel = pd.read_pickle(f'{V3_DATA}/panel_v3_clean.pkl')
panel['lco2'] = np.log1p(panel['co2_kt'].astype(float))

EARLY = list(range(2001, 2010))
LATE  = list(range(2010, 2019))
p_e = panel[panel.year.isin(EARLY)]
p_l = panel[panel.year.isin(LATE)]

# ── Features ──────────────────────────────────────────────────────────────────
log('Computing features...')
mean_all = panel.groupby(['row','col'])['lco2'].mean().rename('mean_lco2').reset_index()
mean_e   = p_e.groupby(['row','col'])['lco2'].mean().rename('lco2_mean_e').reset_index()
mean_l   = p_l.groupby(['row','col'])['lco2'].mean().rename('lco2_mean_l').reset_index()
feat = mean_all.merge(mean_e, on=['row','col']).merge(mean_l, on=['row','col'])
feat['dmean_lco2'] = feat['lco2_mean_l'] - feat['lco2_mean_e']

def pixel_slopes(df):
    records = []
    for (r, c), grp in df.groupby(['row', 'col']):
        x = grp['year'].values.astype(float)
        y = grp['lco2'].values.astype(float)
        slope = linregress(x, y).slope if len(x) >= 2 else np.nan
        records.append({'row': r, 'col': c, 'slope': slope})
    return pd.DataFrame(records)

log('  Computing early-period slopes (2001-2009)...')
sl_e = pixel_slopes(p_e).rename(columns={'slope': 'sl_e'})
log('  Computing late-period slopes (2010-2018)...')
sl_l = pixel_slopes(p_l).rename(columns={'slope': 'sl_l'})

feat = feat.merge(sl_e, on=['row','col']).merge(sl_l, on=['row','col'])
feat['dslope_lco2'] = feat['sl_l'] - feat['sl_e']
feat = feat.dropna()

FEATURES = ['mean_lco2', 'dmean_lco2', 'dslope_lco2']
log(f'\nPixels: {len(feat)}')
log('\nFeature summary (no standardization, log scale):')
for f in FEATURES:
    v = feat[f].values
    log(f'  {f:<16}  mean={v.mean():.4f}  std={v.std():.4f}  '
        f'min={v.min():.4f}  max={v.max():.4f}')

# ── Clustering quality sweep ───────────────────────────────────────────────────
X = feat[FEATURES].values

log('\n' + '='*55)
log('Clustering quality  (no standardization)')
log(f"{'k':>3}  {'Silhouette':>11}  {'Calinski-H':>12}  {'Davies-B':>10}  {'Inertia':>12}")
log('-'*55)
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
    ch  = calinski_harabasz_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    log(f'{k:>3}  {sil:>11.4f}  {ch:>12.1f}  {db:>10.4f}  {km.inertia_:>12.1f}')

# ── Final clustering: k=4 ─────────────────────────────────────────────────────
K = 4
log(f'\n{"="*58}')
log(f'Final clustering: k={K}, no standardization')
log(f'  Silhouette optimal: k=2 | Elbow optimal: k=4 | Selected: k={K}')
log('='*58)

km4 = KMeans(n_clusters=K, random_state=42, n_init=50)
feat['cluster_raw'] = km4.fit_predict(X)

panel_m = panel.merge(feat[['row','col','cluster_raw']], on=['row','col'], how='left')
order = feat.groupby('cluster_raw')['mean_lco2'].mean().sort_values(ascending=False).index

LABEL_NAMES = ['High-decel', 'Mid-decel', 'Mid-accel', 'Low-accel']
LABEL_MAP   = {c: lbl for c, lbl in zip(order, LABEL_NAMES)}

log(f"\n{'C':>3} {'Label':<12} {'N':>6} {'mean_lco2':>10} {'dmean':>7} {'dslope':>9}  "
    f"{'co2_2001':>9} {'co2_2009':>9} {'co2_2018':>9}  {'gr01-09':>8} {'gr10-18':>8}")
log('-'*100)

for c in order:
    sub = feat[feat['cluster_raw'] == c]
    s01 = panel_m[(panel_m['cluster_raw']==c) & (panel_m.year==2001)]['co2_kt'].mean()
    s09 = panel_m[(panel_m['cluster_raw']==c) & (panel_m.year==2009)]['co2_kt'].mean()
    s18 = panel_m[(panel_m['cluster_raw']==c) & (panel_m.year==2018)]['co2_kt'].mean()
    g1  = 100*(s09-s01)/s01
    g2  = 100*(s18-s09)/s09
    lbl = LABEL_MAP[c]
    log(f"{c:>3} {lbl:<12} {len(sub):>6} {sub['mean_lco2'].mean():>10.4f} "
        f"{sub['dmean_lco2'].mean():>7.4f} {sub['dslope_lco2'].mean():>9.5f}  "
        f"{s01:>9.1f} {s09:>9.1f} {s18:>9.1f}  {g1:>8.1f}% {g2:>8.1f}%")

feat['cluster'] = feat['cluster_raw'].map(LABEL_MAP)

# ── Save ──────────────────────────────────────────────────────────────────────
out_types = feat[['row','col','cluster_raw','cluster',
                  'mean_lco2','dmean_lco2','dslope_lco2']].copy()
out_types.to_pickle(f'{OUT_DATA}/cluster_types.pkl')
log(f'\nSaved: {OUT_DATA}/cluster_types.pkl')

# Yearly trajectory per cluster
panel_m2 = panel.merge(out_types[['row','col','cluster']], on=['row','col'], how='left')
traj = (panel_m2.groupby(['cluster','year'])
        .agg(co2=('co2_kt','mean'), urban=('urban_frac','mean'),
             height=('height_m','mean'), n=('co2_kt','count'))
        .reset_index())
traj.to_pickle(f'{OUT_DATA}/cluster_traj.pkl')
log(f'Saved: {OUT_DATA}/cluster_traj.pkl')
log(f'\nResults: {OUT_TXT}')
LOG.close()
