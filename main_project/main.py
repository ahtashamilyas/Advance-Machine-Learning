"""
CIFAR-10 Neural Network Experiment: Activation Functions, Regularization & Confidence Intervals
Study the impact of activation functions, regularization, and sources of randomness on NN training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import random


# =============================================================================
# 1. NEURAL NETWORK WITH CONFIGURABLE ACTIVATION AND REGULARIZATION
# =============================================================================

class CIFAR10Net(nn.Module):
    """
    Simple CNN for CIFAR-10 with configurable activation functions and regularization.
    Architecture: 3 Conv layers + 2 FC layers (kept simple for faster training)
    """
    def __init__(self, activation='relu', regularization='dropout', dropout_rate=0.5):
        super(CIFAR10Net, self).__init__()
        
        self.activation_name = activation
        self.regularization = regularization
        
        # Define activation function
        self.activation = self._get_activation(activation)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Batch normalization layers (used if regularization='batchnorm' or 'both')
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # Dropout (used if regularization='dropout' or 'both')
        self.dropout = nn.Dropout(dropout_rate)
        
    def _get_activation(self, activation):
        """Return activation function based on name"""
        activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        if self.regularization in ['batchnorm', 'both']:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        if self.regularization in ['batchnorm', 'both']:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        if self.regularization in ['batchnorm', 'both']:
            x = self.bn3(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # FC layers
        x = self.fc1(x)
        x = self.activation(x)
        if self.regularization in ['dropout', 'both']:
            x = self.dropout(x)
        
        x = self.fc2(x)
        return x


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def get_cifar10_loaders(batch_size=128, num_workers=2, shuffle_train=True, shuffle_seed=None):
    """
    Load CIFAR-10 dataset with data augmentation for training.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        shuffle_seed: Optional seed for data shuffling
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    
    # Create data loader with optional seed for shuffling
    if shuffle_seed is not None and shuffle_train:
        generator = torch.Generator()
        generator.manual_seed(shuffle_seed)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers,
                                                  generator=generator)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=shuffle_train, num_workers=num_workers)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader


# =============================================================================
# 3. TRAINING AND EVALUATION
# =============================================================================

def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(trainloader), 100. * correct / total


def evaluate(model, testloader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(testloader), 100. * correct / total


def train_model(model, trainloader, testloader, epochs=20, lr=0.001, 
                weight_decay=0.0, device='cuda', verbose=True):
    """
    Complete training loop for a model.
    
    Returns:
        dict: Training history with losses and accuracies
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
    
    for epoch in iterator:
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        scheduler.step()
        
        if verbose:
            iterator.set_postfix({
                'train_acc': f'{train_acc:.2f}%',
                'test_acc': f'{test_acc:.2f}%'
            })
    
    return history


# =============================================================================
# 4. EXPERIMENT RUNNER
# =============================================================================

def set_seed_partially(init_seed=None, dropout_seed=None, shuffle_seed=None):
    """
    Set seeds for different sources of randomness independently.
    
    Args:
        init_seed: Seed for weight initialization (None = random)
        dropout_seed: Seed for dropout RNG (None = random)
        shuffle_seed: Seed for data shuffling (None = random)
    
    Returns:
        dict: Seeds used (actual values)
    """
    seeds_used = {}
    
    # Weight initialization seed
    if init_seed is not None:
        torch.manual_seed(init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(init_seed)
        seeds_used['init_seed'] = init_seed
    else:
        seeds_used['init_seed'] = 'random'
    
    # Dropout seed (controlled via torch manual_seed before dropout operations)
    if dropout_seed is not None:
        # This will be applied before model creation
        seeds_used['dropout_seed'] = dropout_seed
    else:
        seeds_used['dropout_seed'] = 'random'
    
    # Shuffle seed (passed to dataloader)
    seeds_used['shuffle_seed'] = shuffle_seed if shuffle_seed is not None else 'random'
    
    return seeds_used


def run_single_experiment(activation, regularization, run_id, epochs=20, 
                          device='cuda', init_seed=None, dropout_seed=None, 
                          shuffle_seed=None, l2_weight=0.0):
    """
    Run a single training experiment with specified configuration.
    
    Args:
        activation: Activation function name
        regularization: Regularization method name
        run_id: Run identifier
        epochs: Number of training epochs
        device: Device to train on
        init_seed: Seed for weight initialization
        dropout_seed: Seed for dropout (applied via torch seed)
        shuffle_seed: Seed for data shuffling
        l2_weight: L2 regularization weight
    
    Returns:
        dict: Results including final accuracy and training history
    """
    # Set seeds as specified
    seeds_used = set_seed_partially(init_seed, dropout_seed, shuffle_seed)
    
    # Create model
    model = CIFAR10Net(activation=activation, regularization=regularization).to(device)
    
    # Get data loaders
    trainloader, testloader = get_cifar10_loaders(
        batch_size=128,
        num_workers=2,
        shuffle_train=True,
        shuffle_seed=shuffle_seed
    )
    
    # Train model
    history = train_model(
        model, trainloader, testloader,
        epochs=epochs,
        lr=0.001,
        weight_decay=l2_weight,
        device=device,
        verbose=False
    )
    
    # Get final test accuracy
    final_test_acc = history['test_acc'][-1]
    best_test_acc = max(history['test_acc'])
    
    return {
        'run_id': run_id,
        'activation': activation,
        'regularization': regularization,
        'final_test_acc': final_test_acc,
        'best_test_acc': best_test_acc,
        'history': history,
        'seeds': seeds_used
    }


def run_experiments(activations, regularizations, num_runs=5, epochs=20, 
                   device='cuda', seed_control=None, l2_weight=0.0):
    """
    Run all permutations of activation functions and regularization methods.
    
    Args:
        activations: List of activation function names
        regularizations: List of regularization method names
        num_runs: Number of independent runs per configuration
        epochs: Number of epochs per run
        device: Device to train on
        seed_control: Dict specifying which seeds to fix (None = all random)
                     Example: {'init_seed': 42, 'dropout_seed': None, 'shuffle_seed': None}
        l2_weight: L2 regularization weight (if using L2 regularization)
    
    Returns:
        list: Results for all experiments
    """
    results = []
    total_experiments = len(activations) * len(regularizations) * num_runs
    
    print(f"\n{'='*70}")
    print(f"Running {total_experiments} experiments:")
    print(f"  Activations: {activations}")
    print(f"  Regularizations: {regularizations}")
    print(f"  Runs per config: {num_runs}")
    print(f"  Epochs per run: {epochs}")
    if seed_control:
        print(f"  Seed control: {seed_control}")
    print(f"{'='*70}\n")
    
    experiment_num = 0
    
    for activation in activations:
        for regularization in regularizations:
            config_name = f"{activation}_{regularization}"
            print(f"\n--- Configuration: {config_name} ---")
            
            for run in range(num_runs):
                experiment_num += 1
                print(f"[{experiment_num}/{total_experiments}] {config_name} - Run {run+1}/{num_runs}")
                
                # Determine seeds for this run
                if seed_control is None:
                    # All random
                    init_seed = None
                    dropout_seed = None
                    shuffle_seed = None
                else:
                    # Use fixed seeds where specified, random otherwise
                    init_seed = seed_control.get('init_seed', None)
                    dropout_seed = seed_control.get('dropout_seed', None)
                    shuffle_seed = seed_control.get('shuffle_seed', None)
                    
                    # If a seed control value is 'vary', generate a random seed
                    if init_seed == 'vary':
                        init_seed = random.randint(0, 999999)
                    if dropout_seed == 'vary':
                        dropout_seed = random.randint(0, 999999)
                    if shuffle_seed == 'vary':
                        shuffle_seed = random.randint(0, 999999)
                
                # Adjust L2 weight based on regularization type
                current_l2 = l2_weight if 'l2' in regularization.lower() else 0.0
                
                result = run_single_experiment(
                    activation=activation,
                    regularization=regularization,
                    run_id=run,
                    epochs=epochs,
                    device=device,
                    init_seed=init_seed,
                    dropout_seed=dropout_seed,
                    shuffle_seed=shuffle_seed,
                    l2_weight=current_l2
                )
                
                results.append(result)
                print(f"  -> Test Accuracy: {result['final_test_acc']:.2f}% (Best: {result['best_test_acc']:.2f}%)")
    
    return results


# =============================================================================
# 5. STATISTICAL ANALYSIS
# =============================================================================

def compute_confidence_interval(scores, confidence=0.95):
    """
    Compute confidence interval for a list of scores.
    Uses t-distribution for small samples.
    
    Args:
        scores: List or array of scores
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (mean, lower_bound, upper_bound, std)
    """
    from scipy import stats
    
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)  # Sample standard deviation
    
    # Use t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * (std / np.sqrt(n))
    
    return mean, mean - margin, mean + margin, std


def analyze_results(results):
    """
    Analyze results by configuration: compute means, confidence intervals, etc.
    
    Args:
        results: List of experiment results
    
    Returns:
        dict: Analysis summary by configuration
    """
    # Group results by configuration
    configs = defaultdict(list)
    
    for result in results:
        config_key = f"{result['activation']}_{result['regularization']}"
        configs[config_key].append(result['final_test_acc'])
    
    # Compute statistics for each configuration
    analysis = {}
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - 95% Confidence Intervals")
    print("="*80 + "\n")
    
    for config_name, accuracies in sorted(configs.items()):
        mean, lower, upper, std = compute_confidence_interval(accuracies)
        
        analysis[config_name] = {
            'mean': mean,
            'std': std,
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_width': upper - lower,
            'accuracies': accuracies,
            'min': min(accuracies),
            'max': max(accuracies)
        }
        
        print(f"{config_name}:")
        print(f"  Mean Accuracy: {mean:.2f}% ± {std:.2f}%")
        print(f"  95% CI: [{lower:.2f}%, {upper:.2f}%]")
        print(f"  CI Width: {upper - lower:.2f}%")
        print(f"  Range: [{min(accuracies):.2f}%, {max(accuracies):.2f}%]")
        print(f"  Individual runs: {[f'{acc:.2f}' for acc in accuracies]}")
        print()
    
    # Analyze configuration comparisons
    print("\n" + "-"*80)
    print("CONFIGURATION COMPARISON")
    print("-"*80 + "\n")
    
    # Find best configuration
    best_config = max(analysis.items(), key=lambda x: x[1]['mean'])
    print(f"Best Mean Performance: {best_config[0]} ({best_config[1]['mean']:.2f}%)")
    
    # Check CI overlaps
    print("\nConfidence Interval Overlaps:")
    config_names = list(analysis.keys())
    for i, config1 in enumerate(config_names):
        for config2 in config_names[i+1:]:
            ci1 = (analysis[config1]['ci_lower'], analysis[config1]['ci_upper'])
            ci2 = (analysis[config2]['ci_lower'], analysis[config2]['ci_upper'])
            
            overlap = max(0, min(ci1[1], ci2[1]) - max(ci1[0], ci2[0]))
            if overlap > 0:
                print(f"  {config1} ↔ {config2}: OVERLAP ({overlap:.2f}% overlap)")
            else:
                print(f"  {config1} ↔ {config2}: NO OVERLAP (potentially significant difference)")
    
    return analysis


def permutation_test(scores1, scores2, n_permutations=10000):
    """
    Perform permutation test to determine if two score distributions are significantly different.
    
    Args:
        scores1: Array of scores for configuration 1
        scores2: Array of scores for configuration 2
        n_permutations: Number of permutations to perform
    
    Returns:
        tuple: (p_value, observed_difference)
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    
    # Observed difference
    observed_diff = np.mean(scores1) - np.mean(scores2)
    
    # Combine all scores
    combined = np.concatenate([scores1, scores2])
    n1 = len(scores1)
    
    # Permutation test
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    
    p_value = count / n_permutations
    return p_value, observed_diff


def statistical_significance_tests(analysis):
    """
    Perform pairwise statistical significance tests between configurations.
    
    Args:
        analysis: Analysis dictionary from analyze_results()
    """
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS (Permutation Test)")
    print("="*80 + "\n")
    
    config_names = list(analysis.keys())
    
    for i, config1 in enumerate(config_names):
        for config2 in config_names[i+1:]:
            scores1 = analysis[config1]['accuracies']
            scores2 = analysis[config2]['accuracies']
            
            p_value, diff = permutation_test(scores1, scores2, n_permutations=10000)
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"{config1} vs {config2}:")
            print(f"  Mean difference: {diff:.2f}%")
            print(f"  p-value: {p_value:.4f} {significance}")
            print(f"  Significance: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")
            print()


# =============================================================================
# 6. SEED ANALYSIS
# =============================================================================

def analyze_seed_effects(results_dict):
    """
    Analyze how different sources of randomness affect performance.
    
    Args:
        results_dict: Dictionary mapping seed configuration names to results lists
    """
    print("\n" + "="*80)
    print("SEED EFFECT ANALYSIS")
    print("="*80 + "\n")
    
    for config_name, results in results_dict.items():
        accuracies = [r['final_test_acc'] for r in results]
        mean, lower, upper, std = compute_confidence_interval(accuracies)
        
        print(f"{config_name}:")
        print(f"  Mean: {mean:.2f}% ± {std:.2f}%")
        print(f"  95% CI: [{lower:.2f}%, {upper:.2f}%]")
        print(f"  Variance: {std**2:.2f}")
        print(f"  Range: [{min(accuracies):.2f}%, {max(accuracies):.2f}%]")
        print()


# =============================================================================
# 7. RESULTS SAVING AND LOADING
# =============================================================================

def save_results(results, analysis, filename='results.json'):
    """Save results and analysis to JSON file"""
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'analysis': {
            k: {
                'mean': v['mean'],
                'std': v['std'],
                'ci_lower': v['ci_lower'],
                'ci_upper': v['ci_upper'],
                'accuracies': v['accuracies']
            }
            for k, v in analysis.items()
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def load_results(filename='results.json'):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['results'], data['analysis']


# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    activations = ['relu', 'leakyrelu', 'elu']  # At least 2 activation functions
    regularizations = ['dropout', 'batchnorm']  # At least 2 regularization methods
    num_runs = 5  # At least 5 runs per configuration
    epochs = 20  # Reasonable number for fast experimentation
    
    # ==========================================================================
    # EXPERIMENT 1: All permutations without global seed
    # ==========================================================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 1: Training all configurations without global seed")
    print("#"*80)
    
    results_exp1 = run_experiments(
        activations=activations,
        regularizations=regularizations,
        num_runs=num_runs,
        epochs=epochs,
        device=device,
        seed_control=None,  # All random
        l2_weight=0.0001
    )
    
    # Analyze results
    analysis_exp1 = analyze_results(results_exp1)
    
    # Perform statistical significance tests
    statistical_significance_tests(analysis_exp1)
    
    # Save results
    save_results(results_exp1, analysis_exp1, 'experiment1_results.json')
    
    # ==========================================================================
    # EXPERIMENT 2: Seed control experiments
    # ==========================================================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 2: Analyzing sources of randomness")
    print("#"*80)
    
    # Use best performing configuration from Experiment 1
    best_config = max(analysis_exp1.items(), key=lambda x: x[1]['mean'])
    best_activation = best_config[0].split('_')[0]
    best_regularization = best_config[0].split('_')[1]
    
    print(f"\nUsing best configuration: {best_activation} + {best_regularization}")
    
    # Configuration 1: Fix initialization seed, vary others
    print("\n--- Seed Config 1: Fixed initialization, random dropout & shuffling ---")
    results_seed1 = run_experiments(
        activations=[best_activation],
        regularizations=[best_regularization],
        num_runs=5,
        epochs=epochs,
        device=device,
        seed_control={'init_seed': 42, 'dropout_seed': 'vary', 'shuffle_seed': 'vary'}
    )
    
    # Configuration 2: Fix dropout seed, vary others  
    print("\n--- Seed Config 2: Fixed dropout, random initialization & shuffling ---")
    results_seed2 = run_experiments(
        activations=[best_activation],
        regularizations=[best_regularization],
        num_runs=5,
        epochs=epochs,
        device=device,
        seed_control={'init_seed': 'vary', 'dropout_seed': 42, 'shuffle_seed': 'vary'}
    )
    
    # Configuration 3: Fix shuffling seed, vary others
    print("\n--- Seed Config 3: Fixed shuffling, random initialization & dropout ---")
    results_seed3 = run_experiments(
        activations=[best_activation],
        regularizations=[best_regularization],
        num_runs=5,
        epochs=epochs,
        device=device,
        seed_control={'init_seed': 'vary', 'dropout_seed': 'vary', 'shuffle_seed': 42}
    )
    
    # Configuration 4: All random (baseline)
    print("\n--- Seed Config 4: All random (baseline) ---")
    results_seed4 = run_experiments(
        activations=[best_activation],
        regularizations=[best_regularization],
        num_runs=5,
        epochs=epochs,
        device=device,
        seed_control=None
    )
    
    # Analyze seed effects
    seed_results_dict = {
        'Fixed_Init_Random_Dropout_Shuffle': results_seed1,
        'Random_Init_Fixed_Dropout_Random_Shuffle': results_seed2,
        'Random_Init_Dropout_Fixed_Shuffle': results_seed3,
        'All_Random': results_seed4
    }
    
    analyze_seed_effects(seed_results_dict)
    
    # Save seed experiment results
    all_seed_results = results_seed1 + results_seed2 + results_seed3 + results_seed4
    save_results(all_seed_results, {}, 'experiment2_seed_results.json')
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
