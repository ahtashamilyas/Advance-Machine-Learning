"""
Visualization utilities for experiment results.
Creates comprehensive plots and analysis reports.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_json_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_confidence_intervals(analysis, save_path='ci_plot.png'):
    """
    Create a comprehensive plot showing confidence intervals for all configurations.
    """
    configs = list(analysis.keys())
    means = [analysis[c]['mean'] for c in configs]
    stds = [analysis[c]['std'] for c in configs]
    ci_lower = [analysis[c]['ci_lower'] for c in configs]
    ci_upper = [analysis[c]['ci_upper'] for c in configs]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_pos = np.arange(len(configs))
    colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
    
    # Plot bars with error bars
    bars = ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add confidence interval error bars
    errors = [[m - l for m, l in zip(means, ci_lower)],
              [u - m for m, u in zip(means, ci_upper)]]
    ax.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black', 
                capsize=8, capthick=2, linewidth=2, label='95% CI')
    
    # Plot individual run points
    for i, config in enumerate(configs):
        runs = analysis[config]['accuracies']
        x_scatter = np.random.normal(i, 0.06, size=len(runs))
        ax.scatter(x_scatter, runs, alpha=0.8, color='darkred', 
                  s=50, zorder=3, edgecolors='black', linewidths=0.5)
    
    # Styling
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('CIFAR-10 Model Performance with 95% Confidence Intervals', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.7, 
              linewidth=2, label='Random Guess (10%)')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{mean:.2f}%\n±{std:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confidence interval plot saved to {save_path}")
    plt.close()


def plot_variance_comparison(analysis, save_path='variance_plot.png'):
    """
    Compare variance across configurations.
    """
    configs = list(analysis.keys())
    variances = [analysis[c]['std']**2 for c in configs]
    stds = [analysis[c]['std'] for c in configs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Variance plot
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(configs)))
    bars1 = ax1.bar(configs, variances, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Variance Across Configurations', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, var in zip(bars1, variances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Standard deviation plot
    bars2 = ax2.bar(configs, stds, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Standard Deviation (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Standard Deviation', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, std in zip(bars2, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Variance comparison plot saved to {save_path}")
    plt.close()


def plot_training_curves(results, save_path='training_curves.png'):
    """
    Plot training curves for all configurations.
    """
    # Group results by configuration
    from collections import defaultdict
    configs = defaultdict(list)
    
    for result in results:
        config_key = f"{result['activation']}_{result['regularization']}"
        configs[config_key].append(result['history'])
    
    n_configs = len(configs)
    fig, axes = plt.subplots(2, (n_configs + 1) // 2, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (config_name, histories) in enumerate(configs.items()):
        ax = axes[idx]
        
        # Plot all runs for this configuration
        for i, history in enumerate(histories):
            epochs = range(1, len(history['test_acc']) + 1)
            ax.plot(epochs, history['test_acc'], alpha=0.5, linewidth=1.5, 
                   label=f'Run {i+1}')
        
        # Plot mean curve
        test_accs = np.array([h['test_acc'] for h in histories])
        mean_acc = np.mean(test_accs, axis=0)
        epochs = range(1, len(mean_acc) + 1)
        ax.plot(epochs, mean_acc, color='black', linewidth=3, 
               label='Mean', linestyle='--')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(config_name, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    
    # Hide unused subplots
    for idx in range(len(configs), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Training Curves Across All Configurations', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves plot saved to {save_path}")
    plt.close()


def plot_seed_effect_analysis(seed_results_dict, save_path='seed_effects.png'):
    """
    Visualize the effect of different seed configurations.
    """
    configs = list(seed_results_dict.keys())
    
    # Calculate statistics for each configuration
    means = []
    stds = []
    variances = []
    
    for config_name in configs:
        results = seed_results_dict[config_name]
        accs = [r['final_test_acc'] for r in results]
        means.append(np.mean(accs))
        stds.append(np.std(accs, ddof=1))
        variances.append(np.var(accs, ddof=1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    data_for_box = []
    for config_name in configs:
        results = seed_results_dict[config_name]
        accs = [r['final_test_acc'] for r in results]
        data_for_box.append(accs)
    
    bp = ax1.boxplot(data_for_box, labels=configs, patch_artist=True,
                     notch=True, showmeans=True)
    
    # Color boxes
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Seed Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Accuracies by Seed Configuration', 
                 fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Variance comparison
    bars = ax2.bar(configs, variances, color=colors, edgecolor='black', 
                   linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Seed Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Variance by Seed Configuration\n(Lower = More Stable)', 
                 fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Seed effect analysis plot saved to {save_path}")
    plt.close()


def create_summary_report(results, analysis, output_file='summary_report.txt'):
    """
    Create a text summary report of all results.
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CIFAR-10 EXPERIMENT SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total experiments run: {len(results)}\n")
        f.write(f"Configurations tested: {len(analysis)}\n")
        
        all_accs = [r['final_test_acc'] for r in results]
        f.write(f"Overall mean accuracy: {np.mean(all_accs):.2f}%\n")
        f.write(f"Overall std: {np.std(all_accs, ddof=1):.2f}%\n")
        f.write(f"Overall range: [{min(all_accs):.2f}%, {max(all_accs):.2f}%]\n\n")
        
        # Configuration results
        f.write("\nCONFIGURATION RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        # Sort by mean accuracy
        sorted_configs = sorted(analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for rank, (config_name, stats) in enumerate(sorted_configs, 1):
            f.write(f"{rank}. {config_name}\n")
            f.write(f"   Mean: {stats['mean']:.2f}% ± {stats['std']:.2f}%\n")
            f.write(f"   95% CI: [{stats['ci_lower']:.2f}%, {stats['ci_upper']:.2f}%]\n")
            f.write(f"   CI Width: {stats['ci_width']:.2f}%\n")
            f.write(f"   Range: [{stats['min']:.2f}%, {stats['max']:.2f}%]\n")
            f.write(f"   Individual runs: {[f'{acc:.2f}' for acc in stats['accuracies']]}\n\n")
        
        # Best configuration
        best_config = sorted_configs[0]
        f.write("\n" + "="*80 + "\n")
        f.write(f"BEST CONFIGURATION: {best_config[0]}\n")
        f.write(f"Mean Accuracy: {best_config[1]['mean']:.2f}%\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Summary report saved to {output_file}")


def visualize_all(experiment1_file='experiment1_results.json',
                 experiment2_file='experiment2_seed_results.json'):
    """
    Create all visualizations from experiment results.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Create output directory
    Path('visualizations').mkdir(exist_ok=True)
    
    # Load Experiment 1 results
    if Path(experiment1_file).exists():
        print(f"Loading {experiment1_file}...")
        data = load_json_results(experiment1_file)
        results = data['results']
        analysis = data['analysis']
        
        # Create plots
        plot_confidence_intervals(analysis, 'visualizations/confidence_intervals.png')
        plot_variance_comparison(analysis, 'visualizations/variance_comparison.png')
        plot_training_curves(results, 'visualizations/training_curves.png')
        create_summary_report(results, analysis, 'visualizations/summary_report.txt')
    else:
        print(f"⚠ {experiment1_file} not found. Run main.py first.")
    
    # Load Experiment 2 results (seed effects)
    if Path(experiment2_file).exists():
        print(f"\nLoading {experiment2_file}...")
        data = load_json_results(experiment2_file)
        results = data['results']
        
        # Group by seed configuration
        from collections import defaultdict
        seed_results = defaultdict(list)
        
        for result in results:
            # Determine seed config from seeds used
            seeds = result.get('seeds', {})
            if seeds.get('init_seed') not in ['random', None]:
                config = 'Fixed_Init'
            elif seeds.get('dropout_seed') not in ['random', None]:
                config = 'Fixed_Dropout'
            elif seeds.get('shuffle_seed') not in ['random', None]:
                config = 'Fixed_Shuffle'
            else:
                config = 'All_Random'
            
            seed_results[config].append(result)
        
        plot_seed_effect_analysis(seed_results, 'visualizations/seed_effects.png')
    else:
        print(f"⚠ {experiment2_file} not found. Skipping seed effect visualization.")
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETED")
    print("✓ Check the 'visualizations/' directory")
    print("="*80)


if __name__ == "__main__":
    visualize_all()
