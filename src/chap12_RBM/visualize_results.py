#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RBM Training Results Visualization (Updated for Optimized RBM)"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

# Set font - use DejaVu for English, supports all systems
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


def visualize_rbm_results(seed=42):
    try:
        training_errors = np.load('training_errors.npy')
        generated_samples = np.load('generated_samples.npy')
        mnist_data = np.load('mnist_bin.npy')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    rng = np.random.default_rng(seed)
    n_gen = len(generated_samples)

    # 12-column grid: loss curve takes 7 cols, stats takes 5 cols on top row
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 12, figure=fig,
                  height_ratios=[1.4, 1.0, 1.0],
                  hspace=0.35, wspace=0.25)

    # ---------- Top-left: Training loss curve ----------
    ax_loss = fig.add_subplot(gs[0, :7])
    epochs = np.arange(1, len(training_errors) + 1)
    ax_loss.plot(epochs, training_errors, 'b-', linewidth=2.5,
                 marker='o', markersize=7)
    ax_loss.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax_loss.set_ylabel('Reconstruction Error (MSE)',
                       fontsize=12, fontweight='bold')
    ax_loss.set_title('Training Loss Curve (Lower is Better)',
                      fontsize=13, fontweight='bold')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xticks(epochs)

    # Relative offset (based on data range) instead of a hard-coded 0.0008
    err_range = float(training_errors.max() - training_errors.min())
    offset = err_range * 0.06 if err_range > 1e-9 else max(abs(training_errors.mean()) * 0.02, 1e-4)
    for x, y in zip(epochs, training_errors):
        ax_loss.text(x, y + offset, f'{y:.5f}',
                     fontsize=8, ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='yellow', alpha=0.4,
                               edgecolor='none'))
    # Give the labels some headroom
    ax_loss.set_ylim(training_errors.min() - offset,
                     training_errors.max() + 3.5 * offset)

    # ---------- Top-right: Stats panel ----------
    ax_stats = fig.add_subplot(gs[0, 7:])
    ax_stats.axis('off')

    init_err = float(training_errors[0])
    final_err = float(training_errors[-1])
    improvement = (init_err - final_err) / init_err * 100 if init_err > 0 else 0.0

    stats_text = (
        f"Initial Error : {init_err:.6f}\n"
        f"Final Error   : {final_err:.6f}\n"
        f"Improvement   : {improvement:.2f}%\n"
        f"Epochs        : {len(training_errors)}\n"
        f"\n"
        f"Optimizations\n"
        f"-------------\n"
        f"- float32 (SIMD/BLAS x2)\n"
        f"- Numerically stable sigmoid\n"
        f"- Fast Bernoulli sampling\n"
        f"- Variance-reduced gradient\n"
        f"  (h0_prob + v1_prob)\n"
        f"- Momentum + L2 weight decay\n"
        f"- Index-based shuffling\n"
        f"- Xavier initialization"
    )
    ax_stats.text(0.02, 0.98, stats_text,
                  transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='top',
                  family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow',
                            alpha=0.85, pad=1.0))

    # ---------- Middle row: real MNIST samples (for comparison) ----------
    col_width = 12 // n_gen  # 5 samples -> 2 cols each (uses 10 of 12)
    col_offset = (12 - col_width * n_gen) // 2  # center the strip

    real_idx = rng.choice(mnist_data.shape[0], size=n_gen, replace=False)
    for i, ridx in enumerate(real_idx):
        c0 = col_offset + i * col_width
        ax = fig.add_subplot(gs[1, c0:c0 + col_width])
        ax.imshow(mnist_data[ridx].reshape(28, 28),
                  cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Real #{i+1}', fontsize=11, fontweight='bold')
        ax.axis('off')

    # ---------- Bottom row: generated samples ----------
    for i, img in enumerate(generated_samples):
        c0 = col_offset + i * col_width
        ax = fig.add_subplot(gs[2, c0:c0 + col_width])
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Generated #{i+1}', fontsize=11, fontweight='bold')
        ax.axis('off')

    fig.suptitle('RBM Training Results (Optimized Version)',
                 fontsize=15, fontweight='bold', y=0.995)

    output_path = 'rbm_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_path}")
    print(f"Error improvement: {improvement:.2f}%")
    plt.show()


if __name__ == '__main__':
    visualize_rbm_results()