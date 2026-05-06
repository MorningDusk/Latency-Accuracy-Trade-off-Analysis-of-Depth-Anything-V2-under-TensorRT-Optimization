import json
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    results = []
    for model in ['vits', 'vitb', 'vitl']:
        for prec in ['fp32', 'fp16']:
            with open(f'results/raw/{model}_{prec}.json') as f:
                lat = json.load(f)
            with open(f'results/raw/{model}_{prec}_accuracy.json') as f:
                acc = json.load(f)
            results.append({
                'model': model,
                'precision': prec,
                'mean_ms': lat['mean_ms'],
                'fps': lat['fps'],
                'delta1': acc['delta1'],
            })
    return results

def main():
    results = load_results()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    color_map = {'vits': '#1F77B4', 'vitb': '#FF7F0E', 'vitl': '#2CA02C'}
    marker_map = {'fp32': 'o', 'fp16': 's'}
    label_map = {'vits': 'ViT-S', 'vitb': 'ViT-B', 'vitl': 'ViT-L'}
    
    for model in ['vits', 'vitb', 'vitl']:
        pts = [r for r in results if r['model'] == model]
        fp32 = next(r for r in pts if r['precision'] == 'fp32')
        fp16 = next(r for r in pts if r['precision'] == 'fp16')
        
        ax.annotate(
            '', 
            xy=(fp16['mean_ms'], fp16['delta1']),
            xytext=(fp32['mean_ms'], fp32['delta1']),
            arrowprops=dict(arrowstyle='->', color=color_map[model],
                            alpha=0.5, linewidth=1.5,
                            connectionstyle='arc3,rad=0'),
            zorder=1
        )
    
    label_offsets = {
        ('vits', 'fp32'): (10, 8),
        ('vits', 'fp16'): (-5, -18),
        ('vitb', 'fp32'): (10, 8),
        ('vitb', 'fp16'): (10, -5),
        ('vitl', 'fp32'): (-50, 8),
        ('vitl', 'fp16'): (10, -5),
    }
    
    for r in results:
        ax.scatter(r['mean_ms'], r['delta1'],
                   s=180,
                   c=color_map[r['model']],
                   marker=marker_map[r['precision']],
                   edgecolors='black', linewidth=1.2,
                   zorder=3)
        
        offset = label_offsets.get((r['model'], r['precision']), (10, 8))
        ax.annotate(
            f"{label_map[r['model']]}\n{r['precision'].upper()}",
            (r['mean_ms'], r['delta1']),
            xytext=offset,
            textcoords='offset points',
            fontsize=8.5,
            ha='left' if offset[0] > 0 else 'right',
        )
    
    ax.axvline(x=1000/30, color='red', linestyle=':', 
               linewidth=1.5, alpha=0.6, zorder=2)
    ax.text(1000/30 * 0.95, 0.893, '30 FPS',
            color='red', fontsize=9, ha='right', va='bottom',
            rotation=90)
    
    ax.set_xlabel('Mean Latency (ms, log scale)', fontsize=11)
    ax.set_ylabel(r'$\delta_1$ Accuracy on NYU Depth V2', fontsize=11)
    ax.set_title('Latency-Accuracy Trade-off',
                 fontsize=12, pad=10)
    ax.set_xscale('log')
    ax.set_xlim(3, 250)
    ax.set_ylim(0.892, 0.920)
    ax.grid(True, alpha=0.3, zorder=0)
    
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=10, label='FP32'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=10, label='FP16'),
    ]
    ax.legend(handles=legend_elems, loc='lower left', fontsize=10,
              framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('results/tables/figure2_tradeoff.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('results/tables/figure2_tradeoff.pdf',
                bbox_inches='tight', facecolor='white')
    print('Saved figure 2 (trade-off)')

if __name__ == '__main__':
    main()