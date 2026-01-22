"""
Quick Test Script - Run a minimal version of the experiment to verify everything works.
This is useful for testing the setup before running the full experiments.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from main import CIFAR10Net, get_cifar10_loaders, train_model


def quick_test():
    """
    Run a quick test with 2 configurations and 2 runs each.
    Uses only 5 epochs to verify everything is working.
    """
    print("\n" + "="*80)
    print("QUICK TEST - Verifying Setup")
    print("="*80 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test configurations
    test_configs = [
        ('relu', 'dropout'),
        ('leakyrelu', 'batchnorm')
    ]
    
    print("\nDownloading CIFAR-10 dataset (if needed)...")
    trainloader, testloader = get_cifar10_loaders(batch_size=128, num_workers=2)
    print("✓ Dataset loaded successfully")
    
    print("\nTesting configurations...")
    for activation, regularization in test_configs:
        config_name = f"{activation}_{regularization}"
        print(f"\n--- Testing: {config_name} ---")
        
        # Create model
        model = CIFAR10Net(activation=activation, regularization=regularization).to(device)
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Quick training (5 epochs)
        print("Training for 5 epochs...")
        history = train_model(
            model, trainloader, testloader,
            epochs=5,
            lr=0.001,
            device=device,
            verbose=True
        )
        
        final_acc = history['test_acc'][-1]
        print(f"✓ Training completed. Final test accuracy: {final_acc:.2f}%")
    
    print("\n" + "="*80)
    print("✓ QUICK TEST PASSED - Everything is working!")
    print("You can now run the full experiments with: python main.py")
    print("="*80)


if __name__ == "__main__":
    quick_test()
