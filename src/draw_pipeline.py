import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_box(ax, x, y, w, h, text, fc='#E8F0FE', ec='#1F4E79', fontsize=9, fontweight='normal'):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round, pad=0.02",
        linewidth=1.2, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight)

def draw_arrow(ax, x1, y1, x2, y2, color='#444444'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=15,
        linewidth=1.5, color=color
    )
    ax.add_patch(arrow)

def main():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    draw_box(ax, 0.5, 5.8, 11, 1.0, '', fc='#F5F5F5', ec='#888888')
    ax.text(6, 6.55, 'Depth Anything V2 Variants', ha='center', fontsize=10, fontweight='bold')

    models = [
        (1.5, 5.95, 'ViT-S\n24.8M params'),
        (5.0, 5.95, 'ViT-B\n97.5M params'),
        (8.5, 5.95, 'ViT-L\n335.3M params')
    ]
    for x, y, txt in models:
        draw_box(ax, x, y, 2.0, 0.55, txt, fc='#D6E4F5', ec='#1F4E79', fontsize=8)

    for x in [2.5, 6.0, 9.5]:
        draw_arrow(ax, x, 5.9, x, 5.2)

    draw_box(ax, 0.5, 3.5, 11, 1.6, '', fc='#F5F5F5', ec='#888888')
    ax.text(6, 4.95, 'Conversion Pipeline', ha='center', fontsize=10, fontweight='bold')

    chain_y = 4.2
    chain_items = [
        (1.0, 'PyTorch'),
        (3.0, 'ONNX\n(opset 17)'),
        (5.2, 'Simplified\nONNX'),
        (7.4, 'TensorRT\nEngine')
    ]
    for x, txt in chain_items:
        draw_box(ax, x, chain_y, 1.5, 0.55, txt, fc='#FFF2CC', ec='#B8860B', fontsize=8)

    for i in range(len(chain_items) - 1):
        x1 = chain_items[i][0] + 1.5
        x2 = chain_items[i+1][0]
        draw_arrow(ax, x1, chain_y + 0.275, x2, chain_y + 0.275)

    draw_box(ax, 9.3, 4.3, 1.0, 0.4, 'FP32', fc='#E1F5E1', ec='#2E7D32', fontsize=8)
    draw_box(ax, 9.3, 3.65, 1.0, 0.4, 'FP16', fc='#FFE0B2', ec='#E65100', fontsize=8)

    draw_arrow(ax, 8.9, 4.6, 9.3, 4.5)
    draw_arrow(ax, 8.9, 4.6, 9.3, 3.85)

    draw_arrow(ax, 6, 3.4, 6, 2.7)

    draw_box(ax, 0.5, 0.4, 11, 2.2, '', fc='#F5F5F5', ec='#888888')
    ax.text(6, 2.45, 'Benchmark on NVIDIA RTX 4070 Laptop GPU', ha='center', fontsize=10, fontweight='bold')

    measurements = [
            (0.9, 'Latency\n(CUDA Events)\n• mean, median\n• p95, p99\n• 200 iterations'),
            (4.4, 'Memory & Power\n(NVML)\n• VRAM peak\n• Avg power (W)\n'),
            (7.9, 'Accuracy\n(NYU Depth V2)\n• AbsRel, RMSE\n• δ₁, δ₂, δ₃\n• 654 samples'),
    ]
    for x, txt in measurements:
        draw_box(ax, x, 0.6, 3.2, 1.7, txt, 
                 fc='#FCE4EC', ec='#AD1457', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/tables/figure1_pipeline.png', 
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('results/tables/figure1_pipeline.pdf', 
                bbox_inches='tight', facecolor='white')
    print('Saved figure 1 (pipeline) to results/tables/')

if __name__ == '__main__':
    main()