import json
import matplotlib.pyplot as plt
import numpy as np
import os


CATEGORY_ORDER = [
    'Backbone_Attention',
    'Backbone_FFN',
    'Backbone_Norm',
    'Backbone_PatchEmbed',
    'Backbone_Other',
    'DPT_Head',
    'Tensor_Ops',
    'Other',
]

CATEGORY_COLORS = {
    'Backbone_Attention':  '#1F77B4',
    'Backbone_FFN':        '#FF7F0E',
    'Backbone_Norm':       '#2CA02C',
    'Backbone_PatchEmbed': '#9467BD',
    'Backbone_Other':      '#8C564B',
    'DPT_Head':            '#E377C2',
    'Tensor_Ops':          '#BCBD22',
    'Other':               '#7F7F7F',
}

CATEGORY_LABELS = {
    'Backbone_Attention':  'Attention',
    'Backbone_FFN':        'FFN',
    'Backbone_Norm':       'LayerNorm',
    'Backbone_PatchEmbed': 'PatchEmbed',
    'Backbone_Other':      'BB Other',
    'DPT_Head':            'DPT Head',
    'Tensor_Ops':          'Tensor Ops',
    'Other':               'Other',
}


def main():
    # Load data
    data = {}
    for model in ['vits', 'vitb', 'vitl']:
        for prec in ['fp32', 'fp16']:
            path = f'results/profile/{model}_{prec}_categories.json'
            if os.path.exists(path):
                with open(path) as f:
                    data[(model, prec)] = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={'width_ratios': [1, 1]})
    
    # ===== Subplot 1: Absolute latency stacked bar =====
    configs = [('vits', 'fp32'), ('vits', 'fp16'),
               ('vitb', 'fp32'), ('vitb', 'fp16'),
               ('vitl', 'fp32'), ('vitl', 'fp16')]
    labels = [f'{m.replace("vit", "ViT-").upper()[:5]}\n{p.upper()}' 
              for m, p in configs]
    
    x = np.arange(len(configs))
    bottom = np.zeros(len(configs))
    
    for cat in CATEGORY_ORDER:
        values = []
        for cfg in configs:
            if cfg in data and cat in data[cfg]['categories']:
                values.append(data[cfg]['categories'][cat]['total_ms_per_inference'])
            else:
                values.append(0)
        values = np.array(values)
        if values.sum() < 0.01:
            continue
        ax1.bar(x, values, bottom=bottom, 
                label=CATEGORY_LABELS[cat],
                color=CATEGORY_COLORS[cat],
                edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Latency (ms)', fontsize=11)
    ax1.set_title('(a) Absolute Latency', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 총 latency 표시
    for i, cfg in enumerate(configs):
        if cfg in data:
            total = data[cfg]['total_ms']
            ax1.text(i, total + max(bottom)*0.02, f'{total:.1f}', 
                     ha='center', fontsize=8, fontweight='bold')
    
    # ===== Subplot 2: Relative breakdown (%) =====
    bottom = np.zeros(len(configs))
    for cat in CATEGORY_ORDER:
        values = []
        for cfg in configs:
            if cfg in data and cat in data[cfg]['categories']:
                values.append(data[cfg]['categories'][cat]['percentage'])
            else:
                values.append(0)
        values = np.array(values)
        if values.sum() < 0.5:
            continue
        ax2.bar(x, values, bottom=bottom,
                color=CATEGORY_COLORS[cat],
                edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Latency Share (%)', fontsize=11)
    ax2.set_title('(b) Relative Breakdown', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 범례 (subplot 1에만)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.95, ncol=1)
    
    plt.tight_layout()
    out_path = 'results/tables/figure4_layer_breakdown.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()