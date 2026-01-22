"""
Utility functions for experiment management and analysis.
Helper functions for loading, comparing, and exporting results.
"""

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime


def export_to_csv(results, filename='results.csv'):
    """
    Export experiment results to CSV file for external analysis.
    
    Args:
        results: List of experiment results
        filename: Output CSV filename
    """
    if not results:
        print("No results to export")
        return
    
    # Prepare data
    rows = []
    for result in results:
        row = {
            'run_id': result['run_id'],
            'activation': result['activation'],
            'regularization': result['regularization'],
            'final_test_acc': result['final_test_acc'],
            'best_test_acc': result['best_test_acc'],
            'init_seed': result.get('seeds', {}).get('init_seed', 'random'),
            'dropout_seed': result.get('seeds', {}).get('dropout_seed', 'random'),
            'shuffle_seed': result.get('seeds', {}).get('shuffle_seed', 'random')
        }
        rows.append(row)
    
    # Write to CSV
    with open(filename, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"✓ Results exported to {filename}")


def load_and_compare_experiments(file1, file2):
    """
    Load and compare results from two different experiment runs.
    Useful for comparing different hyperparameter settings.
    
    Args:
        file1: First results JSON file
        file2: Second results JSON file
    """
    print("\n" + "="*80)
    print("COMPARING TWO EXPERIMENTS")
    print("="*80 + "\n")
    
    # Load files
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    analysis1 = data1.get('analysis', {})
    analysis2 = data2.get('analysis', {})
    
    print(f"Experiment 1: {file1}")
    print(f"Experiment 2: {file2}\n")
    
    # Compare configurations
    configs1 = set(analysis1.keys())
    configs2 = set(analysis2.keys())
    
    common_configs = configs1.intersection(configs2)
    
    if not common_configs:
        print("⚠ No common configurations found")
        return
    
    print(f"Comparing {len(common_configs)} common configurations:\n")
    
    for config in sorted(common_configs):
        stats1 = analysis1[config]
        stats2 = analysis2[config]
        
        print(f"{config}:")
        print(f"  Experiment 1: {stats1['mean']:.2f}% ± {stats1['std']:.2f}%")
        print(f"  Experiment 2: {stats2['mean']:.2f}% ± {stats2['std']:.2f}%")
        
        diff = stats2['mean'] - stats1['mean']
        print(f"  Difference: {diff:+.2f}%")
        
        if diff > 0:
            print(f"  → Experiment 2 is better")
        elif diff < 0:
            print(f"  → Experiment 1 is better")
        else:
            print(f"  → Same performance")
        print()


def filter_results_by_config(results, activation=None, regularization=None):
    """
    Filter results by activation function or regularization method.
    
    Args:
        results: List of all results
        activation: Filter by activation function (None = all)
        regularization: Filter by regularization method (None = all)
    
    Returns:
        list: Filtered results
    """
    filtered = results
    
    if activation:
        filtered = [r for r in filtered if r['activation'] == activation]
    
    if regularization:
        filtered = [r for r in filtered if r['regularization'] == regularization]
    
    return filtered


def get_best_run_for_config(results, activation, regularization):
    """
    Get the best performing run for a specific configuration.
    
    Args:
        results: List of all results
        activation: Activation function name
        regularization: Regularization method name
    
    Returns:
        dict: Best result for this configuration
    """
    filtered = filter_results_by_config(results, activation, regularization)
    
    if not filtered:
        return None
    
    best = max(filtered, key=lambda x: x['final_test_acc'])
    return best


def compare_activation_functions(results, regularization=None):
    """
    Compare all activation functions, optionally for a specific regularization.
    
    Args:
        results: List of all results
        regularization: Specific regularization method (None = all)
    """
    print("\n" + "="*80)
    print("ACTIVATION FUNCTION COMPARISON")
    if regularization:
        print(f"(Regularization: {regularization})")
    print("="*80 + "\n")
    
    # Group by activation
    from collections import defaultdict
    activation_results = defaultdict(list)
    
    for result in results:
        if regularization is None or result['regularization'] == regularization:
            activation_results[result['activation']].append(result['final_test_acc'])
    
    # Compute statistics
    for activation in sorted(activation_results.keys()):
        accs = activation_results[activation]
        mean = np.mean(accs)
        std = np.std(accs, ddof=1)
        
        print(f"{activation}:")
        print(f"  Mean: {mean:.2f}% ± {std:.2f}%")
        print(f"  Range: [{min(accs):.2f}%, {max(accs):.2f}%]")
        print(f"  N runs: {len(accs)}")
        print()


def compare_regularization_methods(results, activation=None):
    """
    Compare all regularization methods, optionally for a specific activation.
    
    Args:
        results: List of all results
        activation: Specific activation function (None = all)
    """
    print("\n" + "="*80)
    print("REGULARIZATION METHOD COMPARISON")
    if activation:
        print(f"(Activation: {activation})")
    print("="*80 + "\n")
    
    # Group by regularization
    from collections import defaultdict
    reg_results = defaultdict(list)
    
    for result in results:
        if activation is None or result['activation'] == activation:
            reg_results[result['regularization']].append(result['final_test_acc'])
    
    # Compute statistics
    for reg in sorted(reg_results.keys()):
        accs = reg_results[reg]
        mean = np.mean(accs)
        std = np.std(accs, ddof=1)
        
        print(f"{reg}:")
        print(f"  Mean: {mean:.2f}% ± {std:.2f}%")
        print(f"  Range: [{min(accs):.2f}%, {max(accs):.2f}%]")
        print(f"  N runs: {len(accs)}")
        print()


def create_latex_table(analysis, filename='results_table.tex'):
    """
    Create a LaTeX table from analysis results for reports.
    
    Args:
        analysis: Analysis dictionary from analyze_results()
        filename: Output LaTeX filename
    """
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{|l|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Configuration} & \textbf{Mean Acc.} & \textbf{Std Dev} & \textbf{95\% CI} \\")
    lines.append(r"\hline")
    
    # Sort by mean accuracy
    sorted_configs = sorted(analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for config_name, stats in sorted_configs:
        mean = stats['mean']
        std = stats['std']
        ci_lower = stats['ci_lower']
        ci_upper = stats['ci_upper']
        
        # Format configuration name
        config_formatted = config_name.replace('_', ' ')
        
        lines.append(
            f"{config_formatted} & "
            f"{mean:.2f}\\% & "
            f"{std:.2f}\\% & "
            f"[{ci_lower:.2f}, {ci_upper:.2f}]\\% \\\\"
        )
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{CIFAR-10 Classification Results with 95\% Confidence Intervals}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\end{table}")
    
    # Write to file
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ LaTeX table saved to {filename}")


def create_markdown_table(analysis):
    """
    Create a Markdown table from analysis results for GitHub/reports.
    
    Args:
        analysis: Analysis dictionary from analyze_results()
    
    Returns:
        str: Markdown table
    """
    lines = []
    lines.append("| Configuration | Mean Accuracy | Std Dev | 95% CI | Range |")
    lines.append("|--------------|---------------|---------|---------|-------|")
    
    # Sort by mean accuracy
    sorted_configs = sorted(analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for config_name, stats in sorted_configs:
        mean = stats['mean']
        std = stats['std']
        ci_lower = stats['ci_lower']
        ci_upper = stats['ci_upper']
        min_acc = stats['min']
        max_acc = stats['max']
        
        lines.append(
            f"| {config_name} | "
            f"{mean:.2f}% | "
            f"{std:.2f}% | "
            f"[{ci_lower:.2f}, {ci_upper:.2f}]% | "
            f"[{min_acc:.2f}, {max_acc:.2f}]% |"
        )
    
    return '\n'.join(lines)


def generate_full_report(experiment1_file='experiment1_results.json',
                        experiment2_file='experiment2_seed_results.json',
                        output_file='FULL_REPORT.md'):
    """
    Generate a comprehensive markdown report combining all results.
    
    Args:
        experiment1_file: Experiment 1 results JSON
        experiment2_file: Experiment 2 results JSON
        output_file: Output markdown file
    """
    print(f"\nGenerating comprehensive report: {output_file}")
    
    lines = []
    lines.append("# CIFAR-10 Neural Network Experiment - Full Report\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("---\n")
    
    # Load Experiment 1
    if Path(experiment1_file).exists():
        with open(experiment1_file, 'r') as f:
            data1 = json.load(f)
        
        results1 = data1['results']
        analysis1 = data1['analysis']
        
        lines.append("## Experiment 1: Activation Functions × Regularization\n")
        lines.append(f"**Total experiments:** {len(results1)}\n")
        lines.append(f"**Configurations:** {len(analysis1)}\n")
        
        # Add results table
        lines.append("\n### Results Summary\n")
        lines.append(create_markdown_table(analysis1))
        lines.append("\n")
        
        # Best configuration
        best_config = max(analysis1.items(), key=lambda x: x[1]['mean'])
        lines.append(f"\n**Best Configuration:** {best_config[0]}\n")
        lines.append(f"- Mean Accuracy: {best_config[1]['mean']:.2f}%\n")
        lines.append(f"- Standard Deviation: {best_config[1]['std']:.2f}%\n")
        lines.append(f"- 95% CI: [{best_config[1]['ci_lower']:.2f}%, {best_config[1]['ci_upper']:.2f}%]\n")
        lines.append("\n")
    
    # Load Experiment 2
    if Path(experiment2_file).exists():
        with open(experiment2_file, 'r') as f:
            data2 = json.load(f)
        
        results2 = data2['results']
        
        lines.append("## Experiment 2: Sources of Randomness Analysis\n")
        lines.append(f"**Total experiments:** {len(results2)}\n")
        lines.append("\n### Seed Configuration Results\n")
        
        # Group by seed config
        from collections import defaultdict
        seed_groups = defaultdict(list)
        
        for result in results2:
            seeds = result.get('seeds', {})
            if seeds.get('init_seed') not in ['random', None]:
                group = 'Fixed_Initialization'
            elif seeds.get('dropout_seed') not in ['random', None]:
                group = 'Fixed_Dropout'
            elif seeds.get('shuffle_seed') not in ['random', None]:
                group = 'Fixed_Shuffling'
            else:
                group = 'All_Random'
            
            seed_groups[group].append(result['final_test_acc'])
        
        lines.append("| Seed Configuration | Mean Accuracy | Std Dev | Variance |")
        lines.append("|--------------------|---------------|---------|----------|")
        
        for group in sorted(seed_groups.keys()):
            accs = seed_groups[group]
            mean = np.mean(accs)
            std = np.std(accs, ddof=1)
            var = np.var(accs, ddof=1)
            
            lines.append(f"| {group} | {mean:.2f}% | {std:.2f}% | {var:.4f} |")
        
        lines.append("\n")
    
    # Add conclusions section
    lines.append("## Key Findings\n")
    lines.append("1. **Best Activation Function:** [Based on results above]\n")
    lines.append("2. **Best Regularization:** [Based on results above]\n")
    lines.append("3. **Statistical Significance:** [Check p-values in console output]\n")
    lines.append("4. **Primary Randomness Source:** [Based on seed analysis variance]\n")
    lines.append("\n")
    
    lines.append("## Recommendations\n")
    lines.append("- Use the best-performing configuration for production\n")
    lines.append("- Consider ensemble methods if top configurations have similar performance\n")
    lines.append("- Control initialization seed for reproducibility\n")
    lines.append("- Run multiple seeds and average for robust performance\n")
    lines.append("\n---\n")
    lines.append("*For detailed visualizations, see the visualizations/ directory*\n")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write(''.join(lines))
    
    print(f"✓ Full report saved to {output_file}")


if __name__ == "__main__":
    print("Utility functions for experiment management.")
    print("\nAvailable functions:")
    print("  - export_to_csv(results, filename)")
    print("  - load_and_compare_experiments(file1, file2)")
    print("  - filter_results_by_config(results, activation, regularization)")
    print("  - compare_activation_functions(results, regularization)")
    print("  - compare_regularization_methods(results, activation)")
    print("  - create_latex_table(analysis, filename)")
    print("  - create_markdown_table(analysis)")
    print("  - generate_full_report(exp1_file, exp2_file, output_file)")
