"""
plot_s2_composite.py  —  Section 2 composite figure (v4, k=4)
Nature style, 3 rows × 4 columns

Row 1 (a–d): CO₂ area + urban fraction, actual values, dual y-axis
Row 2 (e–h): CO₂ area + building height, actual values, dual y-axis
Row 3 (i–l): Coefficient forest plots per cluster
             Full / Early / Late; 95% CI

Clusters: High-decel | Mid-decel | Mid-accel | Low-accel
"""

import sys, warnings
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

from pathlib import Path
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
FIG_DIR  = ROOT / 'fig'
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
# from utils import next_path # 視需求保留

warnings.filterwarnings('ignore')

DATA_DIR = str(DATA_DIR)
FIG_DIR  = str(FIG_DIR)

# ── Nature rcParams ───────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':         7,
    'axes.labelsize':    7,
    'axes.titlesize':    7.5,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  2,
    'ytick.major.size':  2,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.dpi':       300,
    'pdf.fonttype':      42,
})

# ── Colours ───────────────────────────────────────────────────────────────────
CLUSTERS = ['High-decel', 'Mid-decel', 'Mid-accel', 'Low-accel']
CC = {
    'High-decel': '#E64B35',  
    'Mid-decel':  '#F39B7F',  
    'Mid-accel':  '#4DBBD5',  
    'Low-accel':  '#3C5488',  
}
C_CO2    = '#444444'
C_URBAN  = '#1A6FAD'
C_HEIGHT = '#B5651D'
CV_URBAN  = '#1A6FAD'
CV_HEIGHT = '#C0392B'

# ── Load ──────────────────────────────────────────────────────────────────────
print('Loading data...')
traj   = pd.read_pickle(f'{DATA_DIR}/cluster_traj.pkl')
df_reg = pd.read_csv(f'{DATA_DIR}/reg_by_cluster.csv')

PERIOD_ORDER = ['Full\n2001–2018', 'Early\n2001–09', 'Late\n2010–18']

# map period label from csv
PMAP = {'Full': 'Full\n2001–2018', 'Early': 'Early\n2001–09', 'Late': 'Late\n2010–18'}
df_reg['period_label'] = df_reg['period'].map(PMAP)
df_reg['ci_lo_u'] = df_reg['b_urban']  - 1.96 * df_reg['se_urban']
df_reg['ci_hi_u'] = df_reg['b_urban']  + 1.96 * df_reg['se_urban']
df_reg['ci_lo_h'] = df_reg['b_height'] - 1.96 * df_reg['se_height']
df_reg['ci_hi_h'] = df_reg['b_height'] + 1.96 * df_reg['se_height']

# ── Helpers ───────────────────────────────────────────────────────────────────
XTICKS = [2001, 2006, 2010, 2014, 2018]
XLIM_COEF = (-2.0, 7.0)
YOFF = 0.15

def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return 't'
    return ''

def shade_periods(ax):
    ax.axvspan(2001, 2009.5, color='#EBF3FB', alpha=0.5, zorder=0)
    ax.axvspan(2009.5, 2018, color='#FEF3EC', alpha=0.5, zorder=0)
    ax.axvline(2009.5, color='#bbb', lw=0.6, ls=':', zorder=1)

def decorate_time_ax(ax):
    ax.set_xlim(2001, 2018)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(['2001','06','10','14','18'], fontsize=5.5)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', pad=2)

def panel_label(ax, letter, dx=-0.20, dy=1.18):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

# ── Figure layout ─────────────────────────────────────────────────────────────
FIG_W = 6
FIG_H = 5

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
gs_outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[1, 1, 1.1],
                              hspace=0.4)
gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[0], wspace=0.52)
gs_r2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[1], wspace=0.52)
gs_r3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[2], wspace=0.12)

# 這裡稍微調大了 bottom 留出更多空間給兩排圖例
fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.15) 

LABELS = list('abcdefghijkl')
pidx = 0
ax_r1, ax_r2 = [], []

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1  —  CO₂ + urban fraction
# ─────────────────────────────────────────────────────────────────────────────
for ci, cl in enumerate(CLUSTERS):
    ax1 = fig.add_subplot(gs_r1[ci])
    ax_r1.append(ax1)
    sub = traj[traj['cluster'] == cl].sort_values('year')
    yr  = sub['year'].values

    shade_periods(ax1)
    ax1.fill_between(yr, sub['co2'].values, alpha=0.15, color=C_CO2, zorder=2)
    ax1.plot(yr, sub['co2'].values, color=C_CO2, lw=1.5, zorder=3, solid_capstyle='round')
    if ci == 0:
        ax1.set_ylabel(r'CO$_2$ (kt yr$^{-1}$)', fontsize=6, color=C_CO2, labelpad=2)
    ax1.tick_params(axis='y', colors=C_CO2, labelsize=5.5)
    ax1.spines['left'].set_color(C_CO2); ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['right'].set_visible(False)

    ax2 = ax1.twinx()
    ax2.fill_between(yr, sub['urban'].values, alpha=0.15, color=C_URBAN, zorder=2)
    ax2.plot(yr, sub['urban'].values, color=C_URBAN, lw=1.2, ls='--', zorder=3)
    if ci == 3:
        ax2.set_ylabel('Urban fraction', fontsize=6, color=C_URBAN, labelpad=2)
    ax2.tick_params(axis='y', colors=C_URBAN, labelsize=5.5)
    ax2.spines['right'].set_color(C_URBAN); ax2.spines['right'].set_linewidth(0.5)
    ax2.spines['top'].set_visible(False); ax2.spines['left'].set_visible(False)
    ylo2, yhi2 = ax2.get_ylim(); ax2.set_ylim(ylo2, yhi2 * 1.15)

    decorate_time_ax(ax1)
    ax1.autoscale_view()
    ylo1, yhi1 = ax1.get_ylim(); ax1.set_ylim(ylo1, yhi1 * 1.15)

    ax1.set_title(cl, fontsize=7.5, fontweight='bold', color=CC[cl], pad=2)
    ax1.text(0.25, 0.94, 'Early', transform=ax1.transAxes,
             ha='center', va='top', fontsize=5, color='#999')
    ax1.text(0.75, 0.94, 'Late', transform=ax1.transAxes,
             ha='center', va='top', fontsize=5, color='#999')
    ax1.set_xlabel('Year', fontsize=6, labelpad=1)

    panel_label(ax1, LABELS[pidx], dx=-0.24 if ci==0 else -0.14)
    pidx += 1

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2  —  CO₂ + building height
# ─────────────────────────────────────────────────────────────────────────────
for ci, cl in enumerate(CLUSTERS):
    ax1 = fig.add_subplot(gs_r2[ci])
    ax_r2.append(ax1)
    sub = traj[traj['cluster'] == cl].sort_values('year')
    yr  = sub['year'].values

    shade_periods(ax1)
    ax1.fill_between(yr, sub['co2'].values, alpha=0.15, color=C_CO2, zorder=2)
    ax1.plot(yr, sub['co2'].values, color=C_CO2, lw=1.5, zorder=3, solid_capstyle='round')
    if ci == 0:
        ax1.set_ylabel(r'CO$_2$ (kt yr$^{-1}$)', fontsize=6, color=C_CO2, labelpad=2)
    ax1.tick_params(axis='y', colors=C_CO2, labelsize=5.5)
    ax1.spines['left'].set_color(C_CO2); ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['right'].set_visible(False)

    ax2 = ax1.twinx()
    ax2.fill_between(yr, sub['height'].values, alpha=0.15, color=C_HEIGHT, zorder=2)
    ax2.plot(yr, sub['height'].values, color=C_HEIGHT, lw=1.2, ls=(0,(3,1,1,1)), zorder=3)
    if ci == 3:
        ax2.set_ylabel('Building height (m)', fontsize=6, color=C_HEIGHT, labelpad=2)
    ax2.tick_params(axis='y', colors=C_HEIGHT, labelsize=5.5)
    ax2.spines['right'].set_color(C_HEIGHT); ax2.spines['right'].set_linewidth(0.5)
    ax2.spines['top'].set_visible(False); ax2.spines['left'].set_visible(False)
    ylo2, yhi2 = ax2.get_ylim(); ax2.set_ylim(ylo2, yhi2 * 1.15)

    decorate_time_ax(ax1)
    ax1.autoscale_view()
    ylo1, yhi1 = ax1.get_ylim(); ax1.set_ylim(ylo1, yhi1 * 1.15)

    ax1.set_title(cl, fontsize=7.5, fontweight='bold', color=CC[cl], pad=2)
    ax1.text(0.25, 0.94, 'Early', transform=ax1.transAxes,
             ha='center', va='top', fontsize=5, color='#999')
    ax1.text(0.75, 0.94, 'Late', transform=ax1.transAxes,
             ha='center', va='top', fontsize=5, color='#999')
    ax1.set_xlabel('Year', fontsize=6, labelpad=1)

    panel_label(ax1, LABELS[pidx], dx=-0.24 if ci==0 else -0.14)
    pidx += 1

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3  —  Coefficient forest plots
# ─────────────────────────────────────────────────────────────────────────────
N_PER = len(PERIOD_ORDER)

for ci, cl in enumerate(CLUSTERS):
    ax = fig.add_subplot(gs_r3[ci])
    sub_c = df_reg[df_reg['cluster'] == cl]

    ax.axvline(0, color='#999', lw=0.8, ls='--', zorder=1)

    for pi, plabel in enumerate(PERIOD_ORDER[::-1]):   # Late=0, Early=1, Full=2
        row = sub_c[sub_c['period_label'] == plabel]
        if row.empty: continue
        row = row.iloc[0]

        for (b, ci_lo, ci_hi, p, col, off, mrk) in [
            (row['b_urban'],  row['ci_lo_u'], row['ci_hi_u'],
             row['p_urban'],  CV_URBAN,  +YOFF, 'o'),
            (row['b_height'], row['ci_lo_h'], row['ci_hi_h'],
             row['p_height'], CV_HEIGHT, -YOFF, 's'),
        ]:
            y   = pi + off
            sig = p < 0.10
            b_d     = np.clip(b,     XLIM_COEF[0]+0.05, XLIM_COEF[1]-0.05)
            ci_lo_d = max(ci_lo, XLIM_COEF[0]+0.02)
            ci_hi_d = min(ci_hi, XLIM_COEF[1]-0.02)

            ax.plot([ci_lo_d, ci_hi_d], [y, y],
                    color=col, lw=0.9, alpha=0.5, zorder=2, solid_capstyle='round')
            if ci_lo < XLIM_COEF[0]:
                ax.annotate('', xy=(XLIM_COEF[0]+0.1, y), xytext=(XLIM_COEF[0]+0.4, y),
                            arrowprops=dict(arrowstyle='->', color=col, lw=0.7), zorder=3)
            if ci_hi > XLIM_COEF[1]:
                ax.annotate('', xy=(XLIM_COEF[1]-0.1, y), xytext=(XLIM_COEF[1]-0.4, y),
                            arrowprops=dict(arrowstyle='->', color=col, lw=0.7), zorder=3)

            ax.scatter(b_d, y, s=22, marker=mrk, zorder=4,
                       edgecolors=col, linewidths=0.7,
                       facecolors=col if sig else 'white')
            st = stars(p)
            if st:
                ax.text(b_d + 0.1, y + 0.04, st, va='bottom', ha='left',
                        fontsize=4.5, color=col)

    for pi in range(N_PER):
        if pi % 2 == 0:
            ax.axhspan(pi-0.45, pi+0.45, color='#f6f6f6', zorder=0)

    ax.set_xlim(XLIM_COEF)
    ax.set_ylim(-0.55, N_PER - 0.45)
    ax.set_yticks(range(N_PER))
    if ci == 0:
        ax.set_yticklabels(['Late\n2010–18', 'Early\n2001–09', 'Full\n2001–18'],
                           fontsize=6)
        ax.set_ylabel('Period', fontsize=6.5, labelpad=2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0, pad=2)
    ax.tick_params(axis='x', labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linewidth=0.3, color='#e0e0e0', zorder=0)
    ax.set_xlabel('Coefficient (β)', fontsize=6.5, labelpad=2)
    ax.set_title(cl, fontsize=7.5, fontweight='bold', color=CC[cl], pad=2)

    panel_label(ax, LABELS[pidx], dx=-0.26 if ci==0 else -0.10)
    pidx += 1


# ── Global Legends (Placed at the bottom) ─────────────────────────────────────

# 【第一排圖例】：折線圖圖例 (合併 CO2、Urban fraction、Building height)
leg_trend = [
    mlines.Line2D([0],[0], color=C_CO2,    lw=1.5, label=r'CO$_2$ (left axis)'),
    mlines.Line2D([0],[0], color=C_URBAN,  lw=1.2, ls='--', label='Urban fraction (right axis)'),
    mlines.Line2D([0],[0], color=C_HEIGHT, lw=1.2, ls=(0,(3,1,1,1)), label='Building height (right axis)'),
]
l1 = fig.legend(handles=leg_trend, loc='lower center', ncol=3,
                bbox_to_anchor=(0.5, 0.055), # Y軸座標偏上
                frameon=False, fontsize=6, handletextpad=0.4, columnspacing=1.5)

# 必須手動將第一個 legend 寫入，避免被第二個 fig.legend() 覆蓋
fig.add_artist(l1)

# 【第二排圖例】：森林圖圖例
h_u  = mlines.Line2D([0],[0], marker='o', color=CV_URBAN,  lw=1.0, markersize=4.5,
                      markerfacecolor=CV_URBAN, label='Urban fraction (horizontal expansion)')
h_h  = mlines.Line2D([0],[0], marker='s', color=CV_HEIGHT, lw=1.0, markersize=4.5,
                      markerfacecolor=CV_HEIGHT, label='Building height (vertical densification)')
h_sg = mlines.Line2D([0],[0], marker='o', color='#555', lw=0, markersize=4.5,
                      markerfacecolor='#555', label='p < 0.10 (filled)')
h_ns = mlines.Line2D([0],[0], marker='o', color='#555', lw=0, markersize=4.5,
                      markerfacecolor='white', markeredgewidth=0.7, label='p ≥ 0.10 (open)')

l2 = fig.legend(handles=[h_u, h_h, h_sg, h_ns], loc='lower center', ncol=4,
                bbox_to_anchor=(0.5, 0.03), # Y軸座標偏下
                frameon=False, fontsize=6, handletextpad=0.4, columnspacing=0.9)

# 顯著性文字說明 (移到最右下角)
fig.text(0.97, 0.01,
         't p<0.10  * p<0.05  ** p<0.01  *** p<0.001\nError bars: 95% CI',
         ha='right', va='bottom', fontsize=5, color='#777')

plt.show()