#!/bin/bash

# ==============================================================================
# Neural Network Reliability Verification - Assignment #1 Execution Script
# ==============================================================================
# Usage:
# 1. Grant execution permission to this file: chmod +x run_all_experiments.sh
# 2. Run the script: ./run_all_experiments.sh
# ==============================================================================

# If an error occurs during script execution, stop immediately
set -e

echo "🚀 [START] All adversarial attack experiments are starting..."
echo "----------------------------------------------------------------"

# --- Step 0: Verify Code Integrity with Unit Tests ---
echo "✅ Running unit tests to verify implementations..."
python -m pytest
echo "✅ All tests passed successfully."
echo "----------------------------------------------------------------"

# Setting Hyperparameters
EPOCHS_MNIST=10
EPS_MNIST=0.2
LEARNING_RATE_MNIST=0.001
PGD_ITER_MNIST=10
PGD_STEP_MNIST=0.02

EPOCHS_CIFAR10=10
EPS_CIFAR10=0.2
LEARNING_RATE_CIFAR10=0.001
PGD_ITER_CIFAR10=10
PGD_STEP_CIFAR10=0.02

# --- Step 1. MNIST dataset experiment ---
echo "1️⃣  MNIST dataset experiment is running..."
echo "   - epochs: $EPOCHS_MNIST"
echo "   - epsilon(eps): $EPS_MNIST (relatively large value)"
echo ""

python run_experiment.py \
    --dataset mnist \
    --epochs $EPOCHS_MNIST \
    --batch_size 64 \
    --learning_rate $LEARNING_RATE_MNIST \
    --seed 42 \
    --eps $EPS_MNIST \
    --pgd_iter $PGD_ITER_MNIST \
    --pgd_step $PGD_STEP_MNIST

echo ""
echo "✅ MNIST experiment completed successfully."
echo "----------------------------------------------------------------"


# --- Step 2. CIFAR-10 dataset experiment ---
echo "2️⃣  CIFAR-10 dataset experiment is running..."
echo "   - epochs: $EPOCHS_CIFAR10 (more complex dataset, so more epochs)"
echo "   - epsilon(eps): $EPS_CIFAR10 (relatively small value)"
echo ""

python run_experiment.py \
    --dataset cifar10 \
    --epochs $EPOCHS_CIFAR10 \
    --batch_size 64 \
    --learning_rate $LEARNING_RATE_CIFAR10 \
    --seed 42 \
    --eps $EPS_CIFAR10 \
    --pgd_iter $PGD_ITER_CIFAR10 \
    --pgd_step $PGD_STEP_CIFAR10

echo ""
echo "✅ CIFAR-10 experiment completed successfully."
echo "----------------------------------------------------------------"

echo "🎉 [COMPLETE] All experiments completed successfully."