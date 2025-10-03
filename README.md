# Adversarial Attack Implementation for CV Models

## 1. Project Goal

This project delivers a Python-based implementation of FGSM and PGD adversarial attacks. It serves as a solution for "Assignment #1," focusing on demonstrating these attacks on custom-built neural networks trained on the MNIST and CIFAR-10 datasets. The entire workflow is automated through a single execution script.

## 2. Core Components

- **Attack Algorithms**: FGSM (Targeted/Untargeted) and PGD (Targeted/Untargeted).
- **Custom Models**: Unique, simple CNN architectures for MNIST (`SourceNet`) and CIFAR-10 (`ColorNet`).
- **Reproducibility**: All randomness is controlled by a global seed, and dependencies are listed in `requirements.txt`.
- **Workflow Automation**: `run_experiment.py` script manages all tasks from start to finish.

## 3. Environment Setup

It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

## 4. Usage

This project includes a shell script to automate the entire experimental process for both MNIST and CIFAR-10.
1. Make the script executable in your terminal: Before the first run, grant execution permissions to the script.

```bash
chmod +x run_all_experiments.sh
```

2. Run the automated experiment script: Execute the script from the project root. It will sequentially run the MNIST experiment followed by the CIFAR-10 experiment.

```bash
./run_all_experiments.sh
```

## 5. Script Arguments (for manual execution)
- ```--dataset```: ```mnist``` or ```cifar10```. Default is ```mnist```.
- ```--epochs```: Default is ```3```.
- ```--batch_size```: Default is ```64```.
- ```--learning_rate```: Default is ```0.001```.
- ```--seed```: Global random seed for reproducability. Default is ```42```.
- ```eps```: Perturbation budget for attacks. Default is ```0.2```.
- ```--pgd-iter```: Number of iterations for PGD. Default is ```10```.
- ```--pgd_step_size```: Step size for each PGD iteration. Default is ```0.01```.
