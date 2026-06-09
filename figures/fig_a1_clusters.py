"""
fig_a1_clusters.py  —  Appendix Fig. A1: Feature scatter plots and elbow curve
Nature style, 2×2 grid:
  (a) Scatter: mean_lco2  vs dmean_lco2    (coloured by cluster)
  (b) Scatter: mean_lco2  vs dslope_lco2   (coloured by cluster)
  (c) Scatter: dmean_lco2 vs dslope_lco2   (coloured by cluster)
  (d) Elbow plot: within-cluster sum of squares vs k (2–8), k=4 marked
"""

import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from scipy.stats import linregress

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
FIG_DIR  = ROOT / 'fig'
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from utils import next_path

warnings.filterwarnings('ignore')

# ── Nature rcParams ───────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':         7.5,
    'axes.labelsize':    7.5,
    'axes.titlesize':    7.5,
    'xtick.labelsize':   6.5,
    'ytick.labelsize':   6.5,
    'legend.fontsize':   7.5,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.dpi':       300,
    'pdf.fonttype':      42,
})

# ── Colours & order ───────────────────────────────────────────────────────────
CLUSTERS = ['High-decel', 'Mid-decel', 'Mid-accel', 'Low-accel']
CC = {
    'High-decel': '#E64B35',
    'Mid-decel':  '#F39B7F',
    'Mid-accel':  '#4DBBD5',
    'Low-accel':  '#3C5488',
}
CN = {
    'High-decel': 948,
    'Mid-decel':  2082,
    'Mid-accel':  2241,
    'Low-accel':  1252,
}

# ── Load cluster features ─────────────────────────────────────────────────────
print('Loading cluster data...')
types = pd.read_pickle(DATA_DIR / 'cluster_types.pkl')

# ── Compute elbow curve ───────────────────────────────────────────────────────
print('Computing elbow curve from panel data...')
panel = pd.read_pickle(DATA_DIR / 'panel_v3_clean.pkl')
panel['lco2'] = np.log1p(panel['co2_kt'].astype(float))

EARLY = list(range(2001, 2010))
LATE  = list(range(2010, 2019))
p_e = panel[panel.year.isin(EARLY)]
p_l = panel[panel.year.isin(LATE)]

mean_all = panel.groupby(['row','col'])['lco2'].mean().rename('mean_lco2').reset_index()
mean_e   = p_e.groupby(['row','col'])['lco2'].mean().rename('lco2_mean_e').reset_index()
mean_l   = p_l.groupby(['row','col'])['lco2'].mean().rename('lco2_mean_l').reset_index()
feat = mean_all.merge(mean_e, on=['row','col']).merge(mean_l, on=['row','col'])
feat['dmean_lco2'] = feat['lco2_mean_l'] - feat['lco2_mean_e']

def pixel_slopes(df_in):
    records = []
    for (r, c), grp in df_in.groupby(['row', 'col']):
        x = grp['year'].values.astype(float)
        y = grp['lco2'].values.astype(float)
        slope = linregress(x, y).slope if len(x) >= 2 else np.nan
        records.append({'row': r, 'col': c, 'slope': slope})
    return pd.DataFrame(records)

print('  Computing slopes...')
sl_e = pixel_slopes(p_e).rename(columns={'slope': 'sl_e'})
sl_l = pixel_slopes(p_l).rename(columns={'slope': 'sl_l'})
feat = feat.merge(sl_e, on=['row','col']).merge(sl_l, on=['row','col'])
feat['dslope_lco2'] = feat['sl_l'] - feat['sl_e']
feat = feat.dropna()

X = feat[['mean_lco2', 'dmean_lco2', 'dslope_lco2']].values

K_RANGE = range(2, 9)
inertias = []
print('  Fitting KMeans k=2..8...')
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(X)
    inertias.append(km.inertia_)
    print(f'    k={k}  inertia={km.inertia_:.1f}')

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.0, 5.0), facecolor='white')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.40)
fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.12)

# ── Helper: scatter panel ─────────────────────────────────────────────────────
def scatter_panel(ax, xcol, ycol, xlabel, ylabel, label):
    for cl in CLUSTERS[::-1]:
        sub = types[types['cluster'] == cl]
        ax.scatter(sub[xcol], sub[ycol],
                   c=CC[cl], s=0.8, alpha=0.50, linewidths=0,
                   rasterized=True, zorder=2)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='#ccc', lw=0.5, ls='--', zorder=1)
    ax.axvline(0, color='#ccc', lw=0.5, ls='--', zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(-0.22, 1.10, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

# ── Panels (a–c): scatter plots ───────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
scatter_panel(ax_a,
              xcol='mean_lco2', ycol='dmean_lco2',
              xlabel=r'Mean log CO$_2$ (2001–2018)',
              ylabel=r'$\Delta$mean log CO$_2$',
              label='a')

ax_b = fig.add_subplot(gs[0, 1])
scatter_panel(ax_b,
              xcol='mean_lco2', ycol='dslope_lco2',
              xlabel=r'Mean log CO$_2$ (2001–2018)',
              ylabel=r'$\Delta$slope (log CO$_2$ yr$^{-1}$)',
              label='b')

ax_c = fig.add_subplot(gs[1, 0])
scatter_panel(ax_c,
              xcol='dmean_lco2', ycol='dslope_lco2',
              xlabel=r'$\Delta$mean log CO$_2$',
              ylabel=r'$\Delta$slope (log CO$_2$ yr$^{-1}$)',
              label='c')

# ── Panel (d): elbow plot ─────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

ks = list(K_RANGE)
ax_d.plot(ks, inertias, color='#444', lw=1.2, marker='o',
          markersize=3.5, markerfacecolor='#444', markeredgewidth=0, zorder=3)

idx4 = ks.index(4)
ax_d.plot(4, inertias[idx4], marker='o', markersize=6,
          markerfacecolor='#E64B35', markeredgecolor='white',
          markeredgewidth=0.8, zorder=4)
ax_d.axvline(4, color='#E64B35', lw=0.8, ls='--', zorder=2, alpha=0.7)
ax_d.text(4.15, inertias[idx4], r'$k=4$', fontsize=6.5, color='#E64B35', va='center')

ax_d.set_xlabel('Number of clusters $k$', fontsize=7.5)
ax_d.set_ylabel('Within-cluster sum of squares', fontsize=7.5)
ax_d.set_xticks(ks)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.yaxis.set_major_locator(MaxNLocator(5))
ax_d.text(-0.22, 1.10, 'd', transform=ax_d.transAxes,
          fontsize=9, fontweight='bold', va='top')

# ── Shared legend ─────────────────────────────────────────────────────────────
leg_handles = [
    mpatches.Patch(facecolor=CC[cl], edgecolor='none',
                   label=f'{cl}  (n={CN[cl]:,})')
    for cl in CLUSTERS
]
fig.legend(handles=leg_handles,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=4,
           fontsize=6.5, frameon=False,
           columnspacing=1.2,
           handlelength=1.0, handletextpad=0.5)

out_path = next_path(str(FIG_DIR / 'fig_a1_clusters.png'))
fig.savefig(out_path, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
