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
pytest
echo "✅ All tests passed successfully."
echo "----------------------------------------------------------------"


# --- Step 1. MNIST dataset experiment ---
echo "1️⃣  MNIST dataset experiment is running..."
echo "   - epochs: 5"
echo "   - epsilon(eps): 0.3 (relatively large value)"
echo ""

python run_experiment.py \
    --dataset mnist \
    --epochs 5 \
    --batch_size 128 \
    --seed 42 \
    --eps 0.3 \
    --pgd_iter 10 \
    --pgd_step 0.03

echo ""
echo "✅ MNIST experiment completed successfully."
echo "----------------------------------------------------------------"


# --- Step 2. CIFAR-10 dataset experiment ---
echo "2️⃣  CIFAR-10 dataset experiment is running..."
echo "   - epochs: 10 (more complex dataset, so more epochs)"
echo "   - epsilon(eps): 0.05 (relatively small value)"
echo ""

python run_experiment.py \
    --dataset cifar10 \
    --epochs 10 \
    --batch_size 128 \
    --seed 42 \
    --eps 0.05 \
    --pgd_iter 7 \
    --pgd_step 0.01

echo ""
echo "✅ CIFAR-10 experiment completed successfully."
echo "----------------------------------------------------------------"

echo "🎉 [COMPLETE] All experiments completed successfully."