"""
fig1_clusters.py  —  Fig. 1: Emission trajectory clusters and geographic distribution
Nature style, 2 rows:
  Row 1 (a, b, c): Box plots of the 3 clustering features by cluster
                   (a) mean_lco2  — mean ln CO2 level
                   (b) dmean_lco2 — change in mean ln CO2
                   (c) dslope_lco2 — change in ln CO2 trend slope
  Row 2 (d):       Global South map, Natural Earth projection, pixel dots coloured by cluster
"""

import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import geopandas as gpd
from pyproj import Transformer
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
FIG_DIR  = ROOT / 'fig'
FIG_DIR.mkdir(parents=True, exist_ok=True)

BOUND_SHP = 'path/to/geoBoundariesCGAZ_ADM0.shp'  # TODO: set path to world boundary shapefile

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

NATEARTH_CRS = '+proj=natearth +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

# ── Load ──────────────────────────────────────────────────────────────────────
print('Loading data...')
types  = pd.read_pickle(DATA_DIR / 'cluster_types.pkl')
pixels = pd.read_pickle(DATA_DIR / 'pixels_v3.pkl')
world  = gpd.read_file(BOUND_SHP)

pix_geo = pixels[['row','col','pixel_lat','pixel_lon']].drop_duplicates(['row','col'])
df = types.merge(pix_geo, on=['row','col'], how='left')

# ── Reproject to Natural Earth ────────────────────────────────────────────────
world_ne = world.to_crs(NATEARTH_CRS)
tr = Transformer.from_crs('EPSG:4326', NATEARTH_CRS, always_xy=True)
df['rx'], df['ry'] = tr.transform(df['pixel_lon'].values, df['pixel_lat'].values)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(5, 4), facecolor='white')
gs  = gridspec.GridSpec(2, 1, figure=fig,
                        height_ratios=[1, 1.8],
                        hspace=0.25)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.50)
fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.05)

# ── Helper: box panel ─────────────────────────────────────────────────────────
def box_panel(ax, col, ylabel, label):
    data_by_cluster = [df.loc[df['cluster'] == cl, col].values for cl in CLUSTERS]
    positions = list(range(len(CLUSTERS)))

    bp = ax.boxplot(
        data_by_cluster,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker='o', markersize=1.0, alpha=0.3,
                        markeredgewidth=0, linestyle='none'),
        medianprops=dict(color='white', linewidth=0.8),
        whiskerprops=dict(linewidth=0.6, color='#555'),
        capprops=dict(linewidth=0.6, color='#555'),
        boxprops=dict(linewidth=0.6),
    )

    for patch, cl in zip(bp['boxes'], CLUSTERS):
        patch.set_facecolor(CC[cl])
        patch.set_alpha(0.85)

    for flier, cl in zip(bp['fliers'], CLUSTERS):
        flier.set(markerfacecolor=CC[cl], markeredgecolor='none')

    ax.set_xticks(positions)
    ax.set_xticklabels(['H-dec', 'M-dec', 'M-acc', 'L-acc'], fontsize=6.0)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.axhline(0, color='#ccc', lw=0.5, ls='--', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(-0.28, 1.08, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

# ── Panels (a–c): box plots ───────────────────────────────────────────────────
ax_a = fig.add_subplot(gs_top[0])
box_panel(ax_a, col='mean_lco2',    ylabel=r'Mean ln CO$_2$',                    label='a')

ax_b = fig.add_subplot(gs_top[1])
box_panel(ax_b, col='dmean_lco2',   ylabel=r'$\Delta$mean ln CO$_2$',             label='b')

ax_c = fig.add_subplot(gs_top[2])
box_panel(ax_c, col='dslope_lco2',  ylabel=r'$\Delta$slope (ln CO$_2$ yr$^{-1}$)', label='c')

# ── Panel (d): Natural Earth map ─────────────────────────────────────────────
from matplotlib.patches import PathPatch
from matplotlib.path import Path

ax_m = fig.add_subplot(gs[1])

LAT_MIN = -58
lons_g  = np.linspace(-180, 180, 361)

n_arc = 200
lats_arc = np.linspace(LAT_MIN, 90, n_arc)
xl, yl = tr.transform([-180]*n_arc, lats_arc)
xt, yt = tr.transform(lons_g,       [ 90]*361)
xr, yr = tr.transform([ 180]*n_arc, lats_arc[::-1])
xb, yb = tr.transform(lons_g[::-1], [LAT_MIN]*361)

bnd_x = np.concatenate([xl, xt, xr, xb])
bnd_y = np.concatenate([yl, yt, yr, yb])

verts = np.column_stack([bnd_x, bnd_y])
codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
ax_m.add_patch(PathPatch(Path(verts, codes),
                         facecolor='white', edgecolor='none', zorder=0))

world_ne.plot(ax=ax_m, color='#efefef', edgecolor='#bbbbbb', linewidth=0.25, zorder=1)

lats_g = np.linspace(LAT_MIN, 90, 181)
for lon in range(-150, 181, 30):
    xs, ys = tr.transform([lon]*181, lats_g)
    ax_m.plot(xs, ys, color='#ddd', lw=0.3, zorder=0)
for lat in range(-30, 61, 30):
    xs, ys = tr.transform(lons_g, [lat]*361)
    ax_m.plot(xs, ys, color='#ddd', lw=0.3, zorder=0)

xs60, ys60 = tr.transform(lons_g, [-60]*361)
ax_m.plot(xs60, ys60, color='#ddd', lw=0.3, zorder=0)

xs_eq, ys_eq = tr.transform(lons_g, [0]*361)
ax_m.plot(xs_eq, ys_eq, color='#aaa', lw=0.6, ls=':', zorder=2)

ax_m.plot(bnd_x, bnd_y, color='#888', lw=0.6, zorder=5)

DOT_SIZE = 0.5
for cl in CLUSTERS[::-1]:
    sub = df[df['cluster'] == cl]
    ax_m.scatter(sub['rx'], sub['ry'],
                 c=CC[cl], s=DOT_SIZE, alpha=0.65,
                 linewidths=0, rasterized=True, zorder=3)

X_MAX     = np.nanmax(np.abs(bnd_x)) * 1.005
Y_MAX_top = np.nanmax(bnd_y) * 1.01
Y_MIN_bot = np.nanmin(bnd_y) * 1.01
ax_m.set_xlim(-X_MAX, X_MAX)
ax_m.set_ylim(Y_MIN_bot, Y_MAX_top)
ax_m.set_aspect('equal')
ax_m.spines[['top','right','bottom','left']].set_visible(False)
ax_m.set_xticks([]); ax_m.set_yticks([])
ax_m.set_facecolor('white')

map_handles = [
    mpatches.Patch(facecolor=CC[cl], edgecolor='none',
                   label=f'{cl}  (n={CN[cl]:,})')
    for cl in CLUSTERS
]
ax_m.legend(handles=map_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.10),
            ncol=4,
            fontsize=6.5, frameon=False,
            columnspacing=1.5,
            borderpad=0.6, handlelength=1.0, handletextpad=0.5)

ax_m.text(-0.02, 1.02, 'd', transform=ax_m.transAxes,
          fontsize=9, fontweight='bold', va='top')

out_path = next_path(str(FIG_DIR / 'fig1_clusters.png'))
fig.savefig(out_path, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
