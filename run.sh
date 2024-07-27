#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=80gb
#SBATCH --time=2-00:00
#SBATCH --exclude=cn-g026

source ~/.bashrc
conda activate py311
cd /home/mila/t/tianyu.zhang/scratch/decentralized
python /home/mila/t/tianyu.zhang/scratch/decentralized/main.py --nonIID 1
python /home/mila/t/tianyu.zhang/scratch/decentralized/main.py --nonIID 0