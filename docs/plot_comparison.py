import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Color palette ──
C_HABITAT = '#5B8DBE'   # steel blue
C_GENARK  = '#E8734A'   # warm coral
BG_COLOR  = '#FAFBFD'
GRID_COLOR = '#E0E4EA'
TEXT_COLOR = '#2D3436'

def style_ax(ax, title, ylabel=None):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, fontsize=15, fontweight='bold', color=TEXT_COLOR, pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle='--')
    ax.set_axisbelow(True)

def add_bar_labels(ax, bars, fmt='{:.3f}', offset=0.01):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=TEXT_COLOR)

out_dir = '/data1/home/clk/project/genark/docs'

# =========================================================================
# Figure 1: Navigation Performance (SR, Oracle SR, SPL)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG_COLOR)

metrics_perf = [
    ('Success Rate',  0.520, 0.540),
    ('Oracle SR',     0.574, 0.617),
    ('SPL',           0.476, 0.425),
]

for i, (name, hab, gen) in enumerate(metrics_perf):
    ax = axes[i]
    x = np.array([0, 0.6])
    w = 0.4
    b1 = ax.bar(x[0], hab, w, color=C_HABITAT, edgecolor='white', linewidth=1.2, zorder=3, label='Habitat-Sim')
    b2 = ax.bar(x[1], gen, w, color=C_GENARK,  edgecolor='white', linewidth=1.2, zorder=3, label='GenArk')
    add_bar_labels(ax, b1, offset=0.008)
    add_bar_labels(ax, b2, offset=0.008)
    ax.set_xticks(x)
    ax.set_xticklabels(['Habitat', 'GenArk'], fontsize=11)
    ax.set_ylim(0, max(hab, gen) * 1.25)
    style_ax(ax, name)

axes[0].legend(loc='upper left', fontsize=9, frameon=True, facecolor='white', edgecolor=GRID_COLOR)
fig.suptitle('Navigation Performance Comparison', fontsize=17, fontweight='bold',
             color=TEXT_COLOR, y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/chart_nav_performance.png', dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
print(f'Saved: {out_dir}/chart_nav_performance.png')

# =========================================================================
# Figure 2: Distance to Goal & Path Length
# =========================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG_COLOR)

metrics_dist = [
    ('Distance to Goal (m)',  5.027, 3.911),
    ('Path Length (m)',        9.290, 8.987),
]

for i, (name, hab, gen) in enumerate(metrics_dist):
    ax = axes2[i]
    x = np.array([0, 0.6])
    w = 0.4
    b1 = ax.bar(x[0], hab, w, color=C_HABITAT, edgecolor='white', linewidth=1.2, zorder=3)
    b2 = ax.bar(x[1], gen, w, color=C_GENARK,  edgecolor='white', linewidth=1.2, zorder=3)
    add_bar_labels(ax, b1, fmt='{:.2f}', offset=0.08)
    add_bar_labels(ax, b2, fmt='{:.2f}', offset=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(['Habitat', 'GenArk'], fontsize=11)
    ax.set_ylim(0, max(hab, gen) * 1.3)
    style_ax(ax, name)
    # lower is better arrow
    better_idx = 0 if hab < gen else 1
    better_bar = [b1, b2][better_idx]
    bar = better_bar[0]
    ax.annotate('lower is better', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5),
                fontsize=8, ha='center', color='white', fontstyle='italic', fontweight='bold')

fig2.suptitle('Navigation Quality Comparison', fontsize=17, fontweight='bold',
              color=TEXT_COLOR, y=1.02)
fig2.tight_layout()
fig2.savefig(f'{out_dir}/chart_nav_quality.png', dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
print(f'Saved: {out_dir}/chart_nav_quality.png')

# =========================================================================
# Figure 3: Evaluation Efficiency (Time + Speedup)
# =========================================================================
fig3, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG_COLOR)

# Time comparison
x = np.array([0, 0.6])
w = 0.4
hab_min = 5 * 60 + 36 + 57/60  # 336.95 min
gen_min = 2 * 60 + 44 + 56/60  # 164.93 min
b1 = ax_time.bar(x[0], hab_min, w, color=C_HABITAT, edgecolor='white', linewidth=1.2, zorder=3)
b2 = ax_time.bar(x[1], gen_min, w, color=C_GENARK,  edgecolor='white', linewidth=1.2, zorder=3)
ax_time.text(b1[0].get_x() + b1[0].get_width()/2, hab_min + 5, '5h 37min',
             ha='center', va='bottom', fontsize=12, fontweight='bold', color=TEXT_COLOR)
ax_time.text(b2[0].get_x() + b2[0].get_width()/2, gen_min + 5, '2h 45min',
             ha='center', va='bottom', fontsize=12, fontweight='bold', color=TEXT_COLOR)
# speedup annotation
ax_time.annotate('2.04× faster', xy=(x[1], gen_min * 0.5),
                 fontsize=13, ha='center', color='white', fontweight='bold', fontstyle='italic')
ax_time.set_xticks(x)
ax_time.set_xticklabels(['Habitat', 'GenArk'], fontsize=11)
ax_time.set_ylim(0, hab_min * 1.3)
style_ax(ax_time, 'Total Evaluation Time', ylabel='Minutes')

# Model VRAM comparison
hab_vram = 56
gen_vram = 14
b3 = ax_mem.bar(x[0], hab_vram, w, color=C_HABITAT, edgecolor='white', linewidth=1.2, zorder=3)
b4 = ax_mem.bar(x[1], gen_vram, w, color=C_GENARK,  edgecolor='white', linewidth=1.2, zorder=3)
ax_mem.text(b3[0].get_x() + b3[0].get_width()/2, hab_vram + 1, f'{hab_vram} GB',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=TEXT_COLOR)
ax_mem.text(b4[0].get_x() + b4[0].get_width()/2, gen_vram + 1, f'{gen_vram} GB',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=TEXT_COLOR)
ax_mem.annotate('4× less VRAM', xy=(x[1], gen_vram * 0.5),
                fontsize=13, ha='center', color='white', fontweight='bold', fontstyle='italic')
ax_mem.set_xticks(x)
ax_mem.set_xticklabels(['Habitat', 'GenArk'], fontsize=11)
ax_mem.set_ylim(0, hab_vram * 1.3)
style_ax(ax_mem, 'Model VRAM Usage', ylabel='GB')

fig3.suptitle('Evaluation Efficiency Comparison', fontsize=17, fontweight='bold',
              color=TEXT_COLOR, y=1.02)
fig3.tight_layout()
fig3.savefig(f'{out_dir}/chart_efficiency.png', dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
print(f'Saved: {out_dir}/chart_efficiency.png')

# =========================================================================
# Figure 4: Combined overview (single chart for PPT)
# =========================================================================
fig4, axes4 = plt.subplots(1, 4, figsize=(18, 5), facecolor=BG_COLOR)

overview = [
    ('Success Rate',     0.520, 0.540, '{:.3f}', 0.008),
    ('SPL',              0.476, 0.425, '{:.3f}', 0.008),
    ('Dist. to Goal (m)', 5.027, 3.911, '{:.2f}', 0.08),
    ('Eval Time (min)',  hab_min, gen_min, '{:.0f}', 5),
]

for i, (name, hab, gen, fmt, off) in enumerate(overview):
    ax = axes4[i]
    x = np.array([0, 0.6])
    w = 0.4
    b1 = ax.bar(x[0], hab, w, color=C_HABITAT, edgecolor='white', linewidth=1.2, zorder=3,
                label='Habitat-Sim' if i == 0 else None)
    b2 = ax.bar(x[1], gen, w, color=C_GENARK,  edgecolor='white', linewidth=1.2, zorder=3,
                label='GenArk' if i == 0 else None)
    add_bar_labels(ax, b1, fmt=fmt, offset=off)
    add_bar_labels(ax, b2, fmt=fmt, offset=off)
    ax.set_xticks(x)
    ax.set_xticklabels(['Habitat', 'GenArk'], fontsize=11)
    ax.set_ylim(0, max(hab, gen) * 1.3)
    style_ax(ax, name)

axes4[0].legend(loc='upper left', fontsize=9, frameon=True, facecolor='white', edgecolor=GRID_COLOR)
fig4.suptitle('GenArk vs Habitat-Sim: Key Metrics Overview', fontsize=17, fontweight='bold',
              color=TEXT_COLOR, y=1.02)
fig4.tight_layout()
fig4.savefig(f'{out_dir}/chart_overview.png', dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
print(f'Saved: {out_dir}/chart_overview.png')

plt.close('all')
print('\nAll charts generated successfully!')
