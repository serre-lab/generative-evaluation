#!/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=a5000
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=20g
#SBATCH -o logs/inference_%A_%a.out
#SBATCH -e logs/inference_%A_%a.err
#SBATCH -t 2:00:00

export PYTHONPATH=/users/jamullik/scratch/generative-evaluation:$PYTHONPATH
export HF_HOME=/users/jamullik/scratch/.cache/huggingface/hub
./env/bin/python experiments/neural_eval.py --model sdxl --dataset arcaro --t_max 501 --n_reps 4
