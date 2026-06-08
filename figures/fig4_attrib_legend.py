# -*- coding: utf-8 -*-
"""
plot_s4e_attrib_map_b.py  —  Legend only
对应 plot_s4e_attrib_map_a.py 的双变量图例
  x: β_urban  × slope(ln_urban)   — 水平扩张排放贡献
  y: β_height × slope(ln_height)  — 垂直高密化排放贡献
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
FIG_DIR = Path(__file__).resolve().parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial'],
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.dpi':      300,
    'pdf.fonttype':     42,
})

FIG = str(FIG_DIR / "fig4_attrib_legend.png")

FS_LEG_TITLE = 7
FS_LEG_AXIS  = 7
FS_LEG_TICK  = 7
FS_LEG_TEXT  = 7

# ── Bivariate 3×3 palette ─────────────────────────────────────────────────────
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

# ── Legend ────────────────────────────────────────────────────────────────────
def draw_legend(ax):
    cell = 1.0 / NBIN
    for i in range(NBIN):
        for j in range(NBIN):
            ax.add_patch(mpatches.FancyBboxPatch(
                (j*cell, i*cell), cell, cell,
                boxstyle='square,pad=0', lw=0.4,
                edgecolor='white', facecolor=PALETTE[i, j]))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(['Low', '', 'High'], fontsize=FS_LEG_TICK)
    ax.set_yticklabels(['Low', '', 'High'], fontsize=FS_LEG_TICK)
    ax.set_xlabel('→ Expansion  (β_urban × Δln_urban/yr)',  fontsize=FS_LEG_AXIS, labelpad=2)
    ax.set_ylabel('→ Height  (β_height × Δln_height/yr)',   fontsize=FS_LEG_AXIS, labelpad=2)
    ax.tick_params(length=0)

    for sp in ax.spines.values():
        sp.set_visible(False)

    kw = dict(transform=ax.transAxes, fontsize=FS_LEG_TEXT)
    ax.text(0.03, 0.04, 'Low',              ha='left',  va='bottom', color='#555',  **kw)
    ax.text(0.97, 0.04, 'Expansion\ndom.',  ha='right', va='bottom', color='#333',  **kw)
    ax.text(0.03, 0.96, 'Height\ndom.',     ha='left',  va='top',    color='white', **kw)
    ax.text(0.97, 0.96, 'Both\nhigh',       ha='right', va='top',    color='white', **kw)
    ax.set_title('Emission\ncontrib.', fontsize=FS_LEG_TITLE, pad=2)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(1.6, 1.6), facecolor='white')
fig.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.85)

draw_legend(ax)

plt.savefig(FIG, dpi=300, bbox_inches='tight')
print(f'Saved: {FIG}')
plt.show()
