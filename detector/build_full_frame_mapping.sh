#!/bin/bash
#SBATCH --job-name=refine_frame_mapping_v3
#SBATCH --partition=day
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/nfs/roberts/pi/pi_aw989/rjg73/logs/refine_mapping_v3_%j.out
#SBATCH --error=/nfs/roberts/pi/pi_aw989/rjg73/logs/refine_mapping_v3_%j.err

mkdir -p /nfs/roberts/pi/pi_aw989/rjg73/logs
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate /nfs/roberts/pi/pi_aw989/rjg73/ycrc_conda/envs/detector

set -e
echo "===== Job started at $(date) ====="

DETECTOR_DIR=/nfs/roberts/pi/pi_aw989/rjg73/videos-to-images/detector
DATA_DIR=/nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection

python -u "$DETECTOR_DIR/build_full_frame_mapping_v3.py" \
    --dataset_dir "$DATA_DIR/dataset_player_only" \
    --frames_dir /nfs/roberts/pi/pi_aw989/rjg73/frames \
    --hash_index_path "$DATA_DIR/frame_hash_index.json" \
    --output_path "$DATA_DIR/frame_to_sequence_map_v3.json" \
    --search_window 200

echo "===== Job finished at $(date) ====="