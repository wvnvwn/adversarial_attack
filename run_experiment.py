"""
Main execution script for the adversarial attack experiment.
Initializes configuration, prepares data, builds a model, runs the
training and evaluation pipeline, and reports attack results.
"""
import argparse
from source.utils.config import configure_environment
from source.data.data_provider import load_dataset
from source.models import build_model
from source.core.runner import ExperimentRunner

def main(args):
    # 1. Configure environment (seed, device)
    device = configure_environment(args.seed)
    print(f"Environment configured. Using device: {device}")

    # 2. Load data
    train_loader, test_loader, output_dim = load_dataset(args.dataset, args.batch_size)
    print(f"Dataset '{args.dataset}' loaded.")

    # 3. Build model
    model = build_model(args.dataset, output_dim).to(device)
    print(f"Model for '{args.dataset}' built.")

    # 4. Initialize and run the experiment
    runner = ExperimentRunner(model, device, args)
    
    print("\n--- Starting Model Training ---")
    runner.train(train_loader)
    
    print("\n--- Evaluating on Clean Data ---")
    runner.evaluate_clean(test_loader)
    
    print("\n--- Performing Adversarial Attacks ---")
    runner.evaluate_attacks(test_loader)
    
    print("\n--- Experiment Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Adversarial Attack Experiment")
    # General args
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--seed', type=int, default=123)
    # Training args
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    # Attack args
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--pgd_iter', type=int, default=10)
    parser.add_argument('--pgd_step', type=float, default=0.01)

    cli_args = parser.parse_args()
    main(cli_args)