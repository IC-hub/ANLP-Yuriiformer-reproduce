#!/bin/bash
#SBATCH --job-name=test-causality
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-16,babel-m9-20,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28
#SBATCH --output=logs/test_causality_%j.out
#SBATCH --error=logs/test_causality_%j.err

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
source activate jspo
export PYTHONUNBUFFERED=1
python test_causality.py
