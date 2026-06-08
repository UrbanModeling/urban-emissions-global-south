"""
plot_s1_clusters.py  —  Section 1 composite figure (v4, k=4)
Nature style, 2 rows:
  Row 1 (a, b, c): Scatter plots of all 3 clustering feature pairs, coloured by cluster
                   (a) mean_lco2 vs dmean_lco2
                   (b) mean_lco2 vs dslope_lco2
                   (c) dmean_lco2 vs dslope_lco2
  Row 2 (d):       Global South map, Natural Earth projection, pixel dots coloured by cluster
"""

import sys, warnings
import numpy as np, pandas as pd
import geopandas as gpd
from pyproj import Transformer
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent))
from utils import next_path

warnings.filterwarnings('ignore')

DATA_DIR  = str(DATA_DIR)
PIX_DIR   = str(DATA_DIR)
BOUND_SHP = 'F:/Database/Boundary/World_ACM0/geoBoundariesCGAZ_ADM0.shp'  # TODO: set path to world boundary shapefile
FIG_DIR   = str(FIG_DIR)

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
# RdBu diverging: warm (decel) → cool (accel)
CLUSTERS = ['High-decel', 'Mid-decel', 'Mid-accel', 'Low-accel']
CC = {
    'High-decel': '#E64B35',  # 朱红 (深暖)
    'Mid-decel':  '#F39B7F',  # 桃粉/亮橙 (浅暖)
    'Mid-accel':  '#4DBBD5',  # 明青 (浅冷)
    'Low-accel':  '#3C5488',  # 灰蓝/深蓝 (深冷)
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
types  = pd.read_pickle(f'{DATA_DIR}/cluster_types.pkl')
pixels = pd.read_pickle(f'{PIX_DIR}/pixels_v3.pkl')
world  = gpd.read_file(BOUND_SHP)

pix_geo = pixels[['row','col','pixel_lat','pixel_lon']].drop_duplicates(['row','col'])
df = types.merge(pix_geo, on=['row','col'], how='left')

# ── Reproject to Natural Earth ────────────────────────────────────────────────
world_ne = world.to_crs(NATEARTH_CRS)
tr = Transformer.from_crs('EPSG:4326', NATEARTH_CRS, always_xy=True)
df['rx'], df['ry'] = tr.transform(df['pixel_lon'].values, df['pixel_lat'].values)

# ── Figure layout ─────────────────────────────────────────────────────────────
FIG_W = 5
FIG_H = 4

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
gs  = gridspec.GridSpec(2, 1, figure=fig,
                         height_ratios=[1, 1.5],
                         hspace=0.28)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.40)

fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.05)

# ── Helper: scatter panel ─────────────────────────────────────────────────────
def scatter_panel(ax, xcol, ycol, xlabel, ylabel, label):
    for cl in CLUSTERS[::-1]:
        sub = df[df['cluster'] == cl]
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
    ax.text(-0.22, 1.08, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

# ── Panels (a–c): scatter ─────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs_top[0])
scatter_panel(ax_a,
              xcol='mean_lco2', ycol='dmean_lco2',
              xlabel=r'Mean log CO$_2$ (2001–2018)',
              ylabel=r'$\Delta$mean log CO$_2$',
              label='a')

ax_b = fig.add_subplot(gs_top[1])
scatter_panel(ax_b,
              xcol='mean_lco2', ycol='dslope_lco2',
              xlabel=r'Mean log CO$_2$ (2001–2018)',
              ylabel=r'$\Delta$slope (log CO$_2$ yr$^{-1}$)',
              label='b')

ax_c = fig.add_subplot(gs_top[2])
scatter_panel(ax_c,
              xcol='dmean_lco2', ycol='dslope_lco2',
              xlabel=r'$\Delta$mean log CO$_2$',
              ylabel=r'$\Delta$slope (log CO$_2$ yr$^{-1}$)',
              label='c')

# Shared legend on panel (c)
leg_handles = [
    mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=CC[cl],
                  markersize=6, markeredgewidth=0,
                  label=f'{cl}  (n={CN[cl]:,})')
    for cl in CLUSTERS
]
# ── Panel (d): Natural Earth map ─────────────────────────────────────────────
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from pathlib import Path
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
FIG_DIR  = ROOT / 'fig'
FIG_DIR.mkdir(parents=True, exist_ok=True)

ax_m = fig.add_subplot(gs[1])

LAT_MIN = -58   # crop Antarctica
lons_g  = np.linspace(-180, 180, 361)

# Outer boundary: left arc (LAT_MIN→90) + top line + right arc (90→LAT_MIN) + bottom line
n_arc = 200
lats_arc = np.linspace(LAT_MIN, 90, n_arc)
xl, yl = tr.transform([-180]*n_arc, lats_arc)
xt, yt = tr.transform(lons_g,       [ 90]*361)
xr, yr = tr.transform([ 180]*n_arc, lats_arc[::-1])
xb, yb = tr.transform(lons_g[::-1], [LAT_MIN]*361)

bnd_x = np.concatenate([xl, xt, xr, xb])
bnd_y = np.concatenate([yl, yt, yr, yb])

# Background fill
verts = np.column_stack([bnd_x, bnd_y])
codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
ax_m.add_patch(PathPatch(Path(verts, codes),
                         facecolor='white', edgecolor='none', zorder=0))

world_ne.plot(ax=ax_m, color='#efefef', edgecolor='#bbbbbb',
              linewidth=0.25, zorder=1)

# Graticule (only within cropped extent)
lats_g = np.linspace(LAT_MIN, 90, 181)
for lon in range(-150, 181, 30):
    xs, ys = tr.transform([lon]*181, lats_g)
    ax_m.plot(xs, ys, color='#ddd', lw=0.3, zorder=0)

for lat in range(-30, 61, 30):
    xs, ys = tr.transform(lons_g, [lat]*361)
    ax_m.plot(xs, ys, color='#ddd', lw=0.3, zorder=0)

# -60° parallel (sits just inside crop)
xs60, ys60 = tr.transform(lons_g, [-60]*361)
ax_m.plot(xs60, ys60, color='#ddd', lw=0.3, zorder=0)

# Equator
xs_eq, ys_eq = tr.transform(lons_g, [0]*361)
ax_m.plot(xs_eq, ys_eq, color='#aaa', lw=0.6, ls=':', zorder=2)

# Outer boundary line
ax_m.plot(bnd_x, bnd_y, color='#888', lw=0.6, zorder=5)

# Pixel dots
DOT_SIZE = 0.5
for cl in CLUSTERS[::-1]:
    sub = df[df['cluster'] == cl]
    ax_m.scatter(sub['rx'], sub['ry'],
                 c=CC[cl], s=DOT_SIZE, alpha=0.65,
                 linewidths=0, rasterized=True, zorder=3)

X_MAX = np.nanmax(np.abs(bnd_x)) * 1.005
Y_MAX_top = np.nanmax(bnd_y)  * 1.01
Y_MIN_bot = np.nanmin(bnd_y)  * 1.01
ax_m.set_xlim(-X_MAX, X_MAX)
ax_m.set_ylim(Y_MIN_bot, Y_MAX_top)
ax_m.set_aspect('equal')
ax_m.spines[['top','right','bottom','left']].set_visible(False)
ax_m.set_xticks([]); ax_m.set_yticks([])
ax_m.set_facecolor('white')

# Map legend
map_handles = [
    mpatches.Patch(facecolor=CC[cl], edgecolor='none',
                   label=f'{cl}')
    for cl in CLUSTERS
]

# --- 修改了下面这部分 ---
ax_m.legend(handles=map_handles, 
            loc='lower center',           # 定位基准点改为底边居中
            bbox_to_anchor=(0.5, -0.1),  # 将图例向下移动到地图外侧下方 (可微调 -0.05 这个值)
            ncol=4,                       # 横向排成 4 列 (因为你有 4 个 cluster)
            fontsize=6.5, frameon=False, framealpha=0.95,
            edgecolor='#ddd', 
            columnspacing=1.5,            # 控制横向各个图例项之间的间距
            borderpad=0.6, handlelength=1.0, handletextpad=0.5)

ax_m.text(-0.02, 1.02, 'd', transform=ax_m.transAxes,
          fontsize=9, fontweight='bold', va='top')

plt.show()
