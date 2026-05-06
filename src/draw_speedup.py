import json
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    data = {}
    for model in ['vits', 'vitb', 'vitl']:
        with open(f'results/raw/{model}_fp32.json') as f:
            d32 = json.load(f)
        with open(f'results/raw/{model}_fp16.json') as f:
            d16 = json.load(f)
        data[model] = {
            'speedup': d32['mean_ms'] / d16['mean_ms'],
            'mem_reduction': (d32['vram_total_MB'] - d16['vram_total_MB']) 
                              / d32['vram_total_MB'] * 100,
            'lat_fp32': d32['mean_ms'],
            'lat_fp16': d16['mean_ms'],
        }
    return data

def main():
    data = load_data()
    models = ['vits', 'vitb', 'vitl']
    labels = ['ViT-S', 'ViT-B', 'ViT-L']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    # Subplot 1: Speedup
    speedups = [data[m]['speedup'] for m in models]
    bars1 = ax1.bar(labels, speedups, color=['#1F77B4', '#FF7F0E', '#2CA02C'],
                    edgecolor='black', linewidth=1.2)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax1.set_ylabel('FP16 Speedup over FP32', fontsize=11)
    ax1.set_title('(a) Inference Speedup', fontsize=11)
    ax1.set_ylim(0, 6)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, speedups):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                 f'{val:.2f}×', ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 2: Memory reduction
    mem_red = [data[m]['mem_reduction'] for m in models]
    bars2 = ax2.bar(labels, mem_red, color=['#1F77B4', '#FF7F0E', '#2CA02C'],
                    edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('VRAM Reduction (%)', fontsize=11)
    ax2.set_title('(b) Memory Footprint Reduction', fontsize=11)
    ax2.set_ylim(0, 50)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, mem_red):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1,
                 f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/tables/figure3_speedup.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('results/tables/figure3_speedup.pdf',
                bbox_inches='tight', facecolor='white')
    print('Saved figure 3 (speedup) to results/tables/')

if __name__ == '__main__':
    main()