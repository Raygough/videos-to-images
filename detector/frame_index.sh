#!/bin/bash
#SBATCH --job-name=match_jobs_to_sequences
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/nfs/roberts/pi/pi_aw989/rjg73/logs/match_sequences_%j.out
#SBATCH --error=/nfs/roberts/pi/pi_aw989/rjg73/logs/match_sequences_%j.err

mkdir -p /nfs/roberts/pi/pi_aw989/rjg73/logs
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate /nfs/roberts/pi/pi_aw989/rjg73/ycrc_conda/envs/detector

set -e
echo "===== Job started at $(date) ====="

DETECTOR_DIR=/nfs/roberts/pi/pi_aw989/rjg73/videos-to-images/detector
DATA_DIR=/nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection

python -u "$DETECTOR_DIR/build_frame_hash_index.py" \
    --frames_dir /nfs/roberts/pi/pi_aw989/rjg73/frames \
    --output_path "$DATA_DIR/frame_hash_index.json"

python -u "$DETECTOR_DIR/match_job_to_sequence.py" \
    --jobs_dir "$DATA_DIR/dataset/_raw_extracted" \
    --hash_index_path "$DATA_DIR/frame_hash_index.json" \
    --output_path "$DATA_DIR/job_to_sequence_map.json" \
    --n_samples_per_job 8

echo "===== Job finished at $(date) ====="