#!/bin/bash
#SBATCH --job-name=visualize_frames
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=visualize_frames_%j.log

source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate unidepth_env

python visualize_labels.py \
    --dataset_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/dataset_player_only \
    --split train \
    --n_samples 8 \
    --output_dir spot_check_output