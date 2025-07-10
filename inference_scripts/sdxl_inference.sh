#!/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=a5000
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=60g
#SBATCH -o logs/inference_%A_%a.out
#SBATCH -e logs/inference_%A_%a.err
#SBATCH -t 48:00:00

module load cuda cudnn
export HF_HOME=/users/jamullik/scratch/.cache/huggingface/hub
python sdxl_inference.py
