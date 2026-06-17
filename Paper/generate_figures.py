"""Generate publication-quality figures for the manuscript."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

OUT = r'c:\Users\jaime\OneDrive\Cloud Sabana\MIT\optimizing_pallet_configuration\Paper\figures'

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

COLORS = {
    'hist': '#555555',
    'bl':   '#1f77b4',
    'baf':  '#ff7f0e',
    'blsf': '#2ca02c',
}

# ── Data ───────────────────────────────────────────────────────────────────
flight_ids = [f'F{i:02d}' for i in range(1, 17)]

historical = np.array([0.812, 0.811, 0.862, 0.838, 0.833, 0.848, 0.833, 0.825,
                       0.854, 0.829, 0.841, 0.852, 0.886, 0.831, 0.834, 0.854])
bl   = np.array([0.886, 0.895, 0.906, 0.889, 0.900, 0.895, 0.901, 0.895,
                 0.883, 0.866, 0.883, 0.833, 0.908, 0.889, 0.900, 0.890])
baf  = np.array([0.845, 0.832, 0.808, 0.800, 0.837, 0.831, 0.835, 0.838,
                 0.834, 0.836, 0.830, 0.829, 0.803, 0.831, 0.832, 0.840])
blsf = np.array([0.791, 0.780, 0.820, 0.786, 0.790, 0.800, 0.806, 0.795,
                 0.799, 0.828, 0.830, 0.828, 0.690, 0.818, 0.810, 0.834])

# ── Figure A: Per-flight grouped bar chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 3.5))
x  = np.arange(16)
w  = 0.20

for data, label, color, off in [
    (historical, 'Historical', COLORS['hist'], -1.5*w),
    (bl,         'BL',         COLORS['bl'],    -0.5*w),
    (baf,        'BAF',        COLORS['baf'],    0.5*w),
    (blsf,       'BLSF',       COLORS['blsf'],   1.5*w),
]:
    ax.bar(x + off, data * 100, w, label=label, color=color, alpha=0.88, edgecolor='white', linewidth=0.4)

ax.axhline(historical.mean()*100, color=COLORS['hist'], ls='--', lw=1.1, alpha=0.7)
ax.axhline(bl.mean()*100,         color=COLORS['bl'],   ls='--', lw=1.1, alpha=0.7)

ax.set_xlabel('Flight')
ax.set_ylabel('Volumetric utilisation (%)')
ax.set_xticks(x)
ax.set_xticklabels(flight_ids, rotation=45, ha='right')
ax.set_ylim(60, 97)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='lower right', framealpha=0.9, ncol=2)
ax.grid(axis='y', alpha=0.3, linewidth=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(f'{OUT}/fig_per_flight.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig_per_flight.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig_per_flight')

# ── Figure B: Box plot comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.5))

data_list  = [historical*100, bl*100, baf*100, blsf*100]
labels     = ['Historical', 'BL', 'BAF', 'BLSF']
colors     = [COLORS['hist'], COLORS['bl'], COLORS['baf'], COLORS['blsf']]

bp = ax.boxplot(data_list, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1),
                flierprops=dict(marker='o', markersize=4, linestyle='none'))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticklabels(labels)
ax.set_ylabel('Volumetric utilisation (%)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(60, 97)
ax.grid(axis='y', alpha=0.3, linewidth=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(f'{OUT}/fig_boxplot.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig_boxplot')

# ── Figure C: Pareto chart (box types by volume) ──────────────────────────
# Data from Table 12 of the Word document (top-15 box types + rest)
box_types = [f'A{i}' for i in range(1, 16)] + ['Other']
volumes   = [1497, 1296, 1228, 1048, 790, 506, 499, 455, 375, 336,
             327, 177, 168, 152, 146,
             9912 - (1497+1296+1228+1048+790+506+499+455+375+336+327+177+168+152+146)]

# Recompute from actual total shipped
total_vol = sum(volumes)
cum_pct   = np.cumsum(volumes) / total_vol * 100

fig, ax1 = plt.subplots(figsize=(7.5, 3.8))
ax2 = ax1.twinx()

bar_colors = ['#1f77b4' if i < 15 else '#aec7e8' for i in range(len(volumes))]
ax1.bar(range(len(volumes)), volumes, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.4)
ax2.plot(range(len(volumes)), cum_pct, color='#d62728', marker='o', markersize=3, linewidth=1.5)

# Highlight 90.8% at A15
ax2.axhline(90.81, color='#d62728', ls='--', lw=0.9, alpha=0.6)
ax1.axvline(14, color='gray', ls=':', lw=1)
ax2.annotate('90.8%\n(top 15)', xy=(14, 90.81), xytext=(16, 75),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1),
             fontsize=8, color='#d62728')

ax1.set_xlabel('Box type')
ax1.set_ylabel('Total volume shipped (m³)')
ax2.set_ylabel('Cumulative share (%)')
ax2.set_ylim(0, 105)
ax1.set_xticks(range(len(volumes)))
ax1.set_xticklabels(box_types, rotation=45, ha='right', fontsize=8)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.grid(axis='y', alpha=0.3, linewidth=0.6)

plt.tight_layout()
fig.savefig(f'{OUT}/fig_pareto.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig_pareto.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig_pareto')
print('\nAll figures generated successfully.')
