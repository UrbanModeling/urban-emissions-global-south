"""
plot_arrow_schematic.py — Standalone schematic showing the 4 arrow directions
in β_urban × β_height space (Early → Late period shift).
"""

import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent))
from utils import next_path

from pathlib import Path
FIG_DIR = Path(__file__).resolve().parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# FIG_DIR set above via pathlib

mpl.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':         7,
    'axes.labelsize':    7,
    'axes.titlesize':    7,
    'xtick.labelsize':   5.5,
    'ytick.labelsize':   5.5,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  2,
    'ytick.major.size':  2,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.dpi':       300,
    'pdf.fonttype':      42,
})

COL = '#333333'

fig, ax = plt.subplots(figsize=(1, 1))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.axis('off')

L = 1.3   # arrow length

# Horizontal arrow (left ← origin → right)
ax.annotate('', xy=( L, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=COL, lw=1.2, mutation_scale=9))
ax.annotate('', xy=(-L, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=COL, lw=1.2, mutation_scale=9))

# Vertical arrow (down ↓ origin ↑ up)
ax.annotate('', xy=(0,  L), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=COL, lw=1.2, mutation_scale=9))
ax.annotate('', xy=(0, -L), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=COL, lw=1.2, mutation_scale=9))

# Labels at arrow tips
PAD = 0.18
ax.text( L + PAD,  0,       'Sprawl\nstrengthens', ha='left',   va='center', fontsize=6.5)
ax.text(-L - PAD,  0,       'Sprawl\nweakens',     ha='right',  va='center', fontsize=6.5)
ax.text( 0,        L + PAD, 'Height\nstrengthens', ha='center', va='bottom', fontsize=6.5)
ax.text( 0,       -L - PAD, 'Height\nweakens',     ha='center', va='top',    fontsize=6.5)

# Axis labels near origin
ax.text( L * 0.5, -0.22, r'$\beta_{\rm urban}$',  ha='center', va='top',    fontsize=6.5, color='#555')
ax.text(-0.22,    L * 0.5, r'$\beta_{\rm height}$', ha='right',  va='center', fontsize=6.5, color='#555',
        rotation=90)

out = next_path(f'{FIG_DIR}/arrow_schematic.png')
fig.savefig(out, bbox_inches='tight')
print(f'Saved → {out}')
plt.show()
