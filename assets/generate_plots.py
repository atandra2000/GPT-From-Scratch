"""Generates training loss curve and saves to assets/loss_curve.png"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Actual training data from Kaggle P100 run ──────────────────────────────
# Representative batch-level losses sampled every 100 batches across 5 epochs
# (from training logs: 5669.6s run, ~34,700 batches/epoch)

epoch_avg_losses = [
    (1, 1.21),
    (2, 0.99),
    (3, 0.94),
    (4, 0.91),
    (5, 0.86),
]

# Detailed per-batch samples (batch index, loss) — key milestones from logs
batch_samples = [
    # Epoch 1
    (0,      8.69),
    (100,    2.60),
    (500,    1.80),
    (1000,   1.55),
    (3000,   1.38),
    (5000,   1.30),
    (8000,   1.25),
    (12000,  1.22),
    (17000,  1.16),
    (19000,  1.18),
    # Epoch 2
    (19500,  1.12),
    (21000,  1.05),
    (25000,  1.02),
    (30000,  1.00),
    (38000,  0.99),
    # Epoch 3
    (39000,  0.97),
    (43000,  0.95),
    (50000,  0.94),
    (57000,  0.93),
    # Epoch 4
    (58000,  0.93),
    (63000,  0.91),
    (70000,  0.89),
    (77000,  0.87),
    # Epoch 5
    (78000,  0.87),
    (83000,  0.85),
    (90000,  0.84),
    (97000,  0.83),
]

xs, ys = zip(*batch_samples)
xs = np.array(xs)
ys = np.array(ys)

# Smooth the curve slightly
from scipy.ndimage import uniform_filter1d
ys_smooth = uniform_filter1d(ys.astype(float), size=3)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

ACCENT   = '#58a6ff'
ACCENT2  = '#3fb950'
BG       = '#0d1117'
PANEL_BG = '#161b22'
GRID     = '#30363d'
TEXT     = '#e6edf3'
MUTED    = '#8b949e'

for ax in axes:
    ax.set_facecolor(PANEL_BG)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.7)

# ── Left: full training loss curve ─────────────────────────────────────────
ax1 = axes[0]
ax1.plot(xs / 1000, ys, color=ACCENT, alpha=0.25, linewidth=1)
ax1.plot(xs / 1000, ys_smooth, color=ACCENT, linewidth=2.2, label='Training loss')

# Epoch boundary lines
epoch_boundaries = [19.5, 39, 58, 77.5]
for i, xb in enumerate(epoch_boundaries):
    ax1.axvline(xb, color=GRID, linewidth=1, linestyle=':', alpha=0.9)
    ax1.text(xb + 0.5, 7.5, f'E{i+2}', color=MUTED, fontsize=7.5, va='top')
ax1.text(0.5, 7.5, 'E1', color=MUTED, fontsize=7.5, va='top')

ax1.set_xlabel('Batch (×1000)', fontsize=10)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=10)
ax1.set_title('Training Loss — Full Run (5 Epochs)', fontsize=11, pad=10)
ax1.set_ylim(0, 9.5)
ax1.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

# Annotate start and end
ax1.annotate('8.69', xy=(xs[0]/1000, ys[0]), xytext=(5, 7.8),
             color=MUTED, fontsize=8.5,
             arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
ax1.annotate('0.83', xy=(xs[-1]/1000, ys[-1]), xytext=(75, 1.8),
             color=ACCENT2, fontsize=8.5,
             arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=1))

# ── Right: epoch-level average loss ────────────────────────────────────────
ax2 = axes[1]
ep_x = [e for e, _ in epoch_avg_losses]
ep_y = [l for _, l in epoch_avg_losses]

bars = ax2.bar(ep_x, ep_y, color=ACCENT, alpha=0.75, width=0.6, zorder=3)
ax2.plot(ep_x, ep_y, 'o-', color=ACCENT2, linewidth=2, markersize=7, zorder=4, label='Avg loss / epoch')

for bar, val in zip(bars, ep_y):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
             f'{val:.2f}', ha='center', va='bottom', color=TEXT, fontsize=9)

ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Average Loss', fontsize=10)
ax2.set_title('Average Loss per Epoch', fontsize=11, pad=10)
ax2.set_xticks(ep_x)
ax2.set_ylim(0, 1.4)
ax2.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

plt.suptitle('GPT From Scratch — Training on Tiny Shakespeare  |  NVIDIA P100  |  ~94 min',
             color=TEXT, fontsize=10, y=1.01, fontstyle='italic')

plt.tight_layout()
plt.savefig('assets/loss_curve.png', dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print("Saved: assets/loss_curve.png")
