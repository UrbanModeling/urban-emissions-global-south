# -*- coding: utf-8 -*-
"""
fig4_attrib_map.py  —  Maps only
Bivariate maps: annual attributed emission contribution of horizontal expansion vs vertical densification.
  x: beta_urban  x slope(ln_urban)
  y: beta_height x slope(ln_height)
"""

import sys, warnings
import numpy as np, pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy import stats

from pathlib import Path
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR  = ROOT / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')

V3  = str(DATA_DIR)
V4  = str(DATA_DIR)
FIG = str(FIG_DIR / "fig4_attrib_map.png")

mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial'],
    'font.size':         7,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size':  2,   'ytick.major.size':  2,
    'figure.facecolor':  'white', 'axes.facecolor': 'white',
    'savefig.dpi':       300,  'pdf.fonttype':     42,
})

PIXEL_DEG = 0.1
PAD       = 0.25
FS_TITLE  = 7
FS_AGGLOM = 7
FS_TICK   = 6
FS_ANNOT  = 6

# ── Load ──────────────────────────────────────────────────────────────────────
print('Loading data...')
panel  = pd.read_pickle(f'{V3}/panel_v3_clean.pkl')
pixels = pd.read_pickle(f'{V3}/pixels_v3.pkl')
reg_df = pd.read_csv(f'{V4}/reg_by_agglom.csv')

pix_coords = pixels[['row','col','pixel_lat','pixel_lon','city_id']].drop_duplicates(['row','col'])
city_info  = pixels[['city_id','region']].drop_duplicates('city_id').set_index('city_id')

panel['ln_urban']  = np.log1p(panel['urban_frac'])
panel['ln_height'] = np.log1p(panel['height_m'])

# ── Agglomeration graph ───────────────────────────────────────────────────────
print('Building graph...')
rc_to_city = {(r['row'], r['col']): r['city_id'] for _, r in pixels.iterrows()}
pixel_set  = set(rc_to_city.keys())
G = nx.Graph()
G.add_nodes_from(pixels['city_id'].unique())
for (r, c), cid in rc_to_city.items():
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr == dc == 0: continue
            nb = (r+dr, c+dc)
            if nb in pixel_set and rc_to_city[nb] != cid:
                G.add_edge(cid, rc_to_city[nb])

components = sorted([c for c in nx.connected_components(G) if len(c) >= 2],
                    key=len, reverse=True)

AGGLOM_LABELS = {
    0:'Dhaka–Comilla',   1:'Yangtze Delta',    2:'Pearl River Delta',
    5:'Johannesburg',    8:'Jakarta',           9:'Surabaya',
   21:'Kolkata',        23:'Mexico City',      26:'Lagos',
}
AGGLOM_REG_NAME = {
    0:'Dhaka-Comilla',   1:'Yangtze Delta',    2:'Pearl River Delta',
    5:'Johannesburg',    8:'Jakarta',           9:'Surabaya',
   21:'Kolkata',        23:'Mexico City',      26:'Lagos',
}

GRID = [
    [1,  2,  0 ],
    [21, 8,  9 ],
    [23, 5,  26],
]

# ── Pixel ln slopes ───────────────────────────────────────────────────────────
def pixel_ln_slopes(df, years):
    sub = df[df['year'].isin(years)].copy()
    recs = []
    for (row, col), g in sub.groupby(['row', 'col']):
        y = g['year'].values
        if len(y) < 3: continue
        recs.append({
            'row': row, 'col': col,
            'u_ln_rate': stats.linregress(y, g['ln_urban'].values)[0],
            'h_ln_rate': stats.linregress(y, g['ln_height'].values)[0],
        })
    return pd.DataFrame(recs).merge(pix_coords, on=['row','col'], how='left')

print('Computing ln slopes...')
early_slopes = pixel_ln_slopes(panel, range(2001, 2010))
late_slopes  = pixel_ln_slopes(panel, range(2010, 2019))

# ── Assign pixels to agglomerations ──────────────────────────────────────────
pix_comp = {}
for comp_i, comp in enumerate(components):
    for cid in comp:
        for _, row_p in pixels[pixels['city_id'] == cid].iterrows():
            pix_comp[(row_p['row'], row_p['col'])] = comp_i

def add_comp_idx(df):
    df = df.copy()
    df['comp_i'] = df.apply(lambda r: pix_comp.get((r['row'], r['col']), -1), axis=1)
    return df

early_slopes = add_comp_idx(early_slopes)
late_slopes  = add_comp_idx(late_slopes)

# ── Multiply by regression coefficients ──────────────────────────────────────
def apply_betas(slopes_df, period_name):
    df = slopes_df.copy()
    df['attrib_u'] = np.nan
    df['attrib_h'] = np.nan
    for ag_i, reg_name in AGGLOM_REG_NAME.items():
        rows_reg = reg_df[(reg_df['agglom'] == reg_name) & (reg_df['period'] == period_name)]
        if rows_reg.empty: continue
        b_u = rows_reg.iloc[0]['b_urban']
        b_h = rows_reg.iloc[0]['b_height']
        mask = df['comp_i'] == ag_i
        df.loc[mask, 'attrib_u'] = b_u * df.loc[mask, 'u_ln_rate']
        df.loc[mask, 'attrib_h'] = b_h * df.loc[mask, 'h_ln_rate']
    return df

print('Applying betas...')
early_attrib = apply_betas(early_slopes, 'Early')
late_attrib  = apply_betas(late_slopes,  'Late')

# ── Global quantile breakpoints ───────────────────────────────────────────────
target_comps = set(AGGLOM_REG_NAME.keys())
early_tgt = early_attrib[early_attrib['comp_i'].isin(target_comps)]
late_tgt  = late_attrib[ late_attrib['comp_i'].isin(target_comps)]

all_u = pd.concat([early_tgt['attrib_u'], late_tgt['attrib_u']]).dropna()
all_h = pd.concat([early_tgt['attrib_h'], late_tgt['attrib_h']]).dropna()
U_BREAKS = np.percentile(all_u, [33.3, 66.7])
H_BREAKS = np.percentile(all_h, [33.3, 66.7])

# ── Bivariate palette ─────────────────────────────────────────────────────────
NBIN    = 3
C_SLOW  = np.array([0.94, 0.92, 0.85])
C_OUT   = np.array([0.90, 0.42, 0.38])
C_UP    = np.array([0.30, 0.60, 0.83])
C_UPOUT = np.array([0.24, 0.16, 0.50])

def make_palette(n):
    pal = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            th = i/(n-1); tu = j/(n-1)
            pal[i,j] = np.clip(
                (1-th)*(1-tu)*C_SLOW + (1-th)*tu*C_OUT +
                th*(1-tu)*C_UP + th*tu*C_UPOUT, 0, 1)
    return pal

PALETTE = make_palette(NBIN)

def assign_colors(df):
    df = df.copy()
    df['u_bin'] = np.clip(np.digitize(df['attrib_u'], U_BREAKS), 0, NBIN-1)
    df['h_bin'] = np.clip(np.digitize(df['attrib_h'], H_BREAKS), 0, NBIN-1)
    df['color'] = [tuple(PALETTE[h, u]) for h, u in zip(df['h_bin'], df['u_bin'])]
    return df

early_c = assign_colors(early_attrib)
late_c  = assign_colors(late_attrib)

# ── Figure ────────────────────────────────────────────────────────────────────
col_r = [1, 1, 0.15, 1, 1, 0.15, 1, 1]
row_r = [1, 0.18, 1, 0.18, 1, 0.18]

fig = plt.figure(figsize=(6, 4), facecolor='white')
gs  = gridspec.GridSpec(6, 8, figure=fig,
                         width_ratios=col_r,
                         height_ratios=row_r,
                         hspace=0.1, wspace=0.15)
fig.subplots_adjust(left=0.02, right=0.99, top=0.96, bottom=0.04)

PERIODS = [('Early\n2001–2009', early_c), ('Late\n2010–2018', late_c)]

for ri, row_ag in enumerate(GRID):
    map_r = ri * 2
    lbl_r = ri * 2 + 1

    for ci, ag_i in enumerate(row_ag):
        if ag_i >= len(components): continue
        comp     = components[ag_i]
        label    = AGGLOM_LABELS.get(ag_i, f'#{ag_i}')
        n_cities = len(comp)

        e_c = ci * 3
        l_c = ci * 3 + 1

        city_coords = pix_coords[pix_coords['city_id'].isin(comp)]
        c_lon_min = city_coords['pixel_lon'].min()
        c_lon_max = city_coords['pixel_lon'].max()
        c_lat_min = city_coords['pixel_lat'].min()
        c_lat_max = city_coords['pixel_lat'].max()
        lon_center = (c_lon_max + c_lon_min) / 2
        lat_center = (c_lat_max + c_lat_min) / 2
        max_span = max(c_lon_max - c_lon_min, c_lat_max - c_lat_min) / 2

        ax_lbl = fig.add_subplot(gs[lbl_r, e_c : l_c+1])
        ax_lbl.axis('off')
        ax_lbl.text(0.5, 0.7, f'{label} • {n_cities} cities',
                    ha='center', va='top', fontsize=FS_AGGLOM,
                    transform=ax_lbl.transAxes, clip_on=False)

        for k, (p_label, rate_df) in enumerate(PERIODS):
            map_c = e_c if k == 0 else l_c
            ax = fig.add_subplot(gs[map_r, map_c])
            ax.set_facecolor('#f2f0eb')

            sub = rate_df[rate_df['city_id'].isin(comp)].dropna(subset=['color'])
            for _, pix in sub.iterrows():
                ax.add_patch(mpatches.Rectangle(
                    (pix['pixel_lon']-PIXEL_DEG/2, pix['pixel_lat']-PIXEL_DEG/2),
                    PIXEL_DEG, PIXEL_DEG,
                    lw=0.3, edgecolor='white',
                    facecolor=pix['color'], zorder=2))

            ax.set_xlim(lon_center - max_span - PAD, lon_center + max_span + PAD)
            ax.set_ylim(lat_center - max_span - PAD, lat_center + max_span + PAD)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=FS_TICK, length=2)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3))

            if ri == 0:
                ax.set_title(p_label, fontsize=FS_TITLE, pad=3)

            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.4)
            ax.spines['left'].set_linewidth(0.4)

            ax.text(0.03, 0.03, f'n={len(sub)} px',
                    transform=ax.transAxes, fontsize=FS_ANNOT, color='#555',
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1))

plt.savefig(FIG, dpi=300, bbox_inches='tight')
print(f'Saved: {FIG}')
plt.show()
