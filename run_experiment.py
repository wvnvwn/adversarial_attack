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
    
    # --- Save the trained model ---
    os.makedirs("checkpoints", exist_ok=True)
    model_path = f"checkpoints/{args.dataset}_epochs{args.epochs}_seed{args.seed}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
    
    print("\n--- Evaluating on Clean Data ---")
    clean_accuracy = runner.evaluate_clean(test_loader)
    
    print("\n--- Performing Adversarial Attacks ---")
    attack_results = runner.evaluate_attacks(test_loader)
    
    # --- 5. Save results to a file ---
    final_results = {
        "config": vars(args),
        "clean_accuracy": clean_accuracy,
        "attack_results": attack_results,
    }

    os.makedirs("results", exist_ok=True)
    results_filename = f"results/{args.dataset}_eps{args.eps}_epochs{args.epochs}_seed{args.seed}.json"
    with open(results_filename, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\n--- Experiment Complete ---")
    print(f"Results saved to {results_filename}")

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