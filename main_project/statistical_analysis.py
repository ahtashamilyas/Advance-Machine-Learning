"""
Additional statistical analysis utilities integrating with deep-significance concepts.
This module provides additional tools for statistical significance testing.
"""

import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt


def bootstrap_ci(scores, n_bootstrap=10000, confidence=0.95):
    """
    Compute bootstrap confidence interval.
    
    Args:
        scores: Array of scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        tuple: (mean, lower, upper)
    """
    scores = np.array(scores)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    mean = np.mean(scores)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return mean, lower, upper


def effect_size_cohens_d(scores1, scores2):
    """
    Compute Cohen's d effect size between two groups.
    
    Args:
        scores1: Array of scores for group 1
        scores2: Array of scores for group 2
    
    Returns:
        float: Cohen's d effect size
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    
    n1, n2 = len(scores1), len(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    return d


def interpret_effect_size(d):
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
    
    Returns:
        str: Interpretation
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def multiple_comparison_correction(p_values, method='bonferroni'):
    """
    Apply multiple comparison correction to p-values.
    
    Args:
        p_values: List of p-values
        method: Correction method ('bonferroni' or 'holm')
    
    Returns:
        list: Corrected p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    if method == 'bonferroni':
        return np.minimum(p_values * n, 1.0)
    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        corrected = np.zeros_like(p_values)
        
        for i, idx in enumerate(sorted_indices):
            corrected[idx] = min(sorted_p[i] * (n - i), 1.0)
        
        return corrected
    else:
        return p_values


def comprehensive_comparison(analysis_dict, alpha=0.05):
    """
    Perform comprehensive pairwise comparisons with multiple testing correction.
    
    Args:
        analysis_dict: Dictionary from analyze_results()
        alpha: Significance level
    
    Returns:
        dict: Comparison results
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL COMPARISON")
    print("="*80 + "\n")
    
    config_names = list(analysis_dict.keys())
    comparisons = []
    p_values = []
    
    # Perform all pairwise comparisons
    for config1, config2 in combinations(config_names, 2):
        scores1 = analysis_dict[config1]['accuracies']
        scores2 = analysis_dict[config2]['accuracies']
        
        # t-test
        t_stat, p_val = stats.ttest_ind(scores1, scores2)
        
        # Effect size
        d = effect_size_cohens_d(scores1, scores2)
        
        comparisons.append({
            'config1': config1,
            'config2': config2,
            'mean_diff': np.mean(scores1) - np.mean(scores2),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': d,
            'effect_interpretation': interpret_effect_size(d)
        })
        
        p_values.append(p_val)
    
    # Apply Bonferroni correction
    corrected_p = multiple_comparison_correction(p_values, method='bonferroni')
    
    # Add corrected p-values to comparisons
    for i, comp in enumerate(comparisons):
        comp['p_value_corrected'] = corrected_p[i]
    
    # Print results
    print(f"Total comparisons: {len(comparisons)}")
    print(f"Significance level (α): {alpha}")
    print(f"Bonferroni-corrected α: {alpha / len(comparisons):.4f}\n")
    
    for comp in comparisons:
        print(f"{comp['config1']} vs {comp['config2']}:")
        print(f"  Mean difference: {comp['mean_diff']:+.2f}%")
        print(f"  t-statistic: {comp['t_statistic']:.3f}")
        print(f"  p-value (uncorrected): {comp['p_value']:.4f}")
        print(f"  p-value (Bonferroni): {comp['p_value_corrected']:.4f}")
        print(f"  Cohen's d: {comp['cohens_d']:.3f} ({comp['effect_interpretation']})")
        
        if comp['p_value_corrected'] < alpha:
            print(f"  ✓ SIGNIFICANT (after correction)")
        elif comp['p_value'] < alpha:
            print(f"  ⚠ Significant before correction, not after")
        else:
            print(f"  ✗ Not significant")
        print()
    
    return comparisons


def plot_results(analysis_dict, filename='results_plot.png'):
    """
    Create visualization of results with confidence intervals.
    
    Args:
        analysis_dict: Dictionary from analyze_results()
        filename: Output filename for plot
    """
    config_names = list(analysis_dict.keys())
    means = [analysis_dict[c]['mean'] for c in config_names]
    ci_lower = [analysis_dict[c]['ci_lower'] for c in config_names]
    ci_upper = [analysis_dict[c]['ci_upper'] for c in config_names]
    
    # Calculate error bars
    errors = [[m - l for m, l in zip(means, ci_lower)],
              [u - m for m, u in zip(means, ci_upper)]]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(config_names))
    ax.bar(x_pos, means, yerr=errors, capsize=10, alpha=0.7, color='steelblue')
    
    # Plot individual runs as scatter
    for i, config in enumerate(config_names):
        runs = analysis_dict[config]['accuracies']
        x_scatter = np.random.normal(i, 0.04, size=len(runs))
        ax.scatter(x_scatter, runs, alpha=0.6, color='red', s=30, zorder=3)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Random guess (10%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {filename}")
    plt.close()


def variance_decomposition(seed_results_dict):
    """
    Decompose variance to understand sources of randomness.
    
    Args:
        seed_results_dict: Dictionary mapping seed configurations to results
    
    Returns:
        dict: Variance analysis
    """
    print("\n" + "="*80)
    print("VARIANCE DECOMPOSITION ANALYSIS")
    print("="*80 + "\n")
    
    variances = {}
    
    for config_name, results in seed_results_dict.items():
        accuracies = [r['final_test_acc'] for r in results]
        variance = np.var(accuracies, ddof=1)
        variances[config_name] = {
            'variance': variance,
            'std': np.sqrt(variance),
            'cv': np.sqrt(variance) / np.mean(accuracies) * 100  # Coefficient of variation
        }
    
    # Sort by variance
    sorted_configs = sorted(variances.items(), key=lambda x: x[1]['variance'])
    
    print("Configurations sorted by variance (lowest to highest):\n")
    for config_name, stats in sorted_configs:
        print(f"{config_name}:")
        print(f"  Variance: {stats['variance']:.4f}")
        print(f"  Std Dev: {stats['std']:.2f}%")
        print(f"  CV: {stats['cv']:.2f}%")
        print()
    
    # Identify the source with lowest variance
    lowest_var_config = sorted_configs[0][0]
    highest_var_config = sorted_configs[-1][0]
    
    print(f"\n✓ Lowest variance: {lowest_var_config}")
    print(f"  → Fixing this source of randomness reduces variance most")
    print(f"\n✗ Highest variance: {highest_var_config}")
    print(f"  → This configuration has most variability")
    
    return variances


if __name__ == "__main__":
    print("This module provides statistical analysis utilities.")
    print("Import and use with main.py results.")
