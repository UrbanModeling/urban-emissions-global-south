"""
plot_s3_regional.py  —  Section 3 small-multiple arrow figure
Each panel = one region; arrow from Early (2001-09) to Late (2010-18)
in β_urban × β_height space.
Style mirrors Brelsford et al. small-multiple arrow grids.
"""

import sys, warnings
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as FancyArrow

sys.path.insert(0, str(Path(__file__).parent))
from utils import next_path

from pathlib import Path
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
FIG_DIR  = ROOT / 'fig'
FIG_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')

DATA_DIR = str(DATA_DIR)
FIG_DIR  = str(FIG_DIR)

mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':          7,
    'axes.labelsize':     7,
    'axes.titlesize':     7,
    'xtick.labelsize':    5.5,
    'ytick.labelsize':    5.5,
    'axes.linewidth':     0.5,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.major.size':   2,
    'ytick.major.size':   2,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.dpi':        300,
    'pdf.fonttype':       42,
})

# ── Data ──────────────────────────────────────────────────────────────────────
df    = pd.read_csv(f'{DATA_DIR}/reg_by_region.csv')
early = df[df['period'] == 'Early'].set_index('region')
late  = df[df['period'] == 'Late'].set_index('region')

# ── Panel layout (geographic grouping) ───────────────────────────────────────
GRID = [
    ['Eastern Asia',    'South-Central Asia', 'South-Eastern Asia'],
    ['Western Asia',    'Northern Africa',    'Western Africa'],
    ['Eastern Africa',  'Middle Africa',      'Southern Africa'],
    ['South America',   'Central America',    'Caribbean'],
]
N_ROWS, N_COLS = len(GRID), len(GRID[0])

# ── Arrow colours (two periods) ──────────────────────────────────────────────
COL1 = '#E05C5C'   # red   — Arrow 1: origin → Early  (Early period)
COL2 = '#2AADA6'   # teal  — Arrow 2: Early  → Late   (Late period)
SIG  = 0.10

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 4, 4
fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(FIG_W, FIG_H),
                          sharex=False, sharey=False)
fig.subplots_adjust(left=0.10, right=0.97, top=0.91,
                    bottom=0.1, hspace=0.62, wspace=0.38)

def _arrow(ax, x0, y0, x1, y1, color, lw=1):
    """Draw arrow from (x0,y0) to (x1,y1), skip if zero-length."""
    if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
        return
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=8),
                zorder=3)

def _dot(ax, x, y, color, sig, marker='o', s=10):
    fc = color if sig else 'white'
    ax.scatter(x, y, s=s, marker=marker, color=fc,
               edgecolors=color, linewidths=0.9, zorder=5)

for ri, row in enumerate(GRID):
    for ci, reg in enumerate(row):
        ax = axes[ri, ci]

        xe = early.loc[reg, 'b_urban'];  ye = early.loc[reg, 'b_height']
        xl = late.loc[reg,  'b_urban'];  yl = late.loc[reg,  'b_height']
        pue = early.loc[reg, 'p_urban']; phe = early.loc[reg, 'p_height']
        pul = late.loc[reg,  'p_urban']; phl = late.loc[reg,  'p_height']

        sig_e = (pue < SIG) or (phe < SIG)
        sig_l = (pul < SIG) or (phl < SIG)

        # ── Panel auto-limits with generous padding ───────────────────────────
        all_x = [0, xe, xl]; all_y = [0, ye, yl]
        xspan = max(abs(max(all_x) - min(all_x)), 0.5)
        yspan = max(abs(max(all_y) - min(all_y)), 0.2)
        xmid  = (max(all_x) + min(all_x)) / 2
        ymid  = (max(all_y) + min(all_y)) / 2
        ax.set_xlim(xmid - xspan * 1.4, xmid + xspan * 1.4)
        ax.set_ylim(ymid - yspan * 1.4, ymid + yspan * 1.4)

        # ── Grid & reference lines ────────────────────────────────────────────
        ax.set_facecolor('white')
        ax.grid(True, color='#e8e8e8', lw=0.4, zorder=0)
        ax.axhline(0, color='#bbb', lw=0.5, ls='--', zorder=1)
        ax.axvline(0, color='#bbb', lw=0.5, ls='--', zorder=1)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_color('#ccc')

        # ── Arrow 1 (red): origin → Early  ───────────────────────────────────
        _arrow(ax, 0, 0, xe, ye, COL1)
        _dot(ax, xe, ye, COL1, sig_e, marker='o')

        # ── Arrow 2 (teal): Early → Late  ────────────────────────────────────
        _arrow(ax, xe, ye, xl, yl, COL2)
        _dot(ax, xl, yl, COL2, sig_l, marker='o')

        # ── Panel title ───────────────────────────────────────────────────────
        ax.set_title(reg, fontsize=6.5, pad=3)

        # ── Axis labels (edge panels only) ────────────────────────────────────
        if ri == N_ROWS - 1:
            ax.set_xlabel(r'$\beta_{\rm urban}$', fontsize=6.5)
        if ci == 0:
            ax.set_ylabel(r'$\beta_{\rm height}$', fontsize=6.5)

        ax.tick_params(labelsize=5)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))

# ── Figure title ──────────────────────────────────────────────────────────────
fig.text(0.03, 0.97, 'Regions', fontsize=8, fontweight='bold', va='top')

# ── Legend ────────────────────────────────────────────────────────────────────
leg_items = [
    mlines.Line2D([0],[0], color=COL1, lw=1.3,
                  marker='>', markersize=5, markevery=[1],
                  label='2001–09 (origin → Early)'),
    mlines.Line2D([0],[0], color=COL2, lw=1.3,
                  marker='>', markersize=5, markevery=[1],
                  label='2010–18 (Early → Late)'),
    mlines.Line2D([0],[0], marker='o', color='#555',
                  markerfacecolor='white', markersize=5,
                  markeredgewidth=0.8, lw=0, label='Open = p ≥ 0.10'),
]
fig.legend(handles=leg_items, loc='lower center', ncol=3,
           fontsize=6.2, frameon=False,
           bbox_to_anchor=(0.53, -0.03))

plt.show()
