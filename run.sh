#!/bin/bash
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/DLS/plots:$PYTHONPATH
export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113
export WANDB_API_KEY=37a72a78c5f940ce06577aa12da3a035e20fde05
python3  /mnt/csp/mmvision/home/lwh/DLS/main.py