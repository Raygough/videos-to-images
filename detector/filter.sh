#!/bin/bash
#SBATCH --job-name=visualize_frames
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=filter_classes_%j.log

source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate unidepth_env

python filter_player_class.py \
    --input_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/dataset \
    --output_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/dataset_player_only \
    --player_class_id 4