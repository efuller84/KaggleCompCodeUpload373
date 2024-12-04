#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1           # GPU requested; if not needed, remove this line.
#SBATCH --cpus-per-task=1      # Number of CPU cores.
#SBATCH --mem=80G              # Memory allocation.
#SBATCH --time=2:00:00           # Time limit (hh:mm:ss).
#SBATCH --job-name=ProductRecommender
#SBATCH --output=ProductRecommender.out
#SBATCH --error=ProductRecommender.err
# Load necessary modules
module load anaconda/2024.02-py311
# Activate the environment
source activate CS373Kaggle
# Run your Python script
python3 ~/KaggleCompetition/similarity_model_1.0.py

