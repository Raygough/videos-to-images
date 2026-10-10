#!/bin/bash
#SBATCH --job-name=train_player_detector
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nfs/roberts/pi/pi_aw989/rjg73/logs/train_detector_%j.out
#SBATCH --error=/nfs/roberts/pi/pi_aw989/rjg73/logs/train_detector_%j.err

mkdir -p /nfs/roberts/pi/pi_aw989/rjg73/logs

source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate /nfs/roberts/pi/pi_aw989/rjg73/ycrc_conda/envs/detector

set -e

echo "===== Job started at $(date) ====="
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

DATASET_DIR=/nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/dataset_player_only

awk -v d="$DATASET_DIR" '{print d"/"$0}' "$DATASET_DIR/train.txt" > "$DATASET_DIR/train_abs.txt" && mv "$DATASET_DIR/train_abs.txt" "$DATASET_DIR/train.txt"
awk -v d="$DATASET_DIR" '{print d"/"$0}' "$DATASET_DIR/val.txt" > "$DATASET_DIR/val_abs.txt" && mv "$DATASET_DIR/val_abs.txt" "$DATASET_DIR/val.txt"

RUNS_DIR=/nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/runs

yolo detect train \
    data="$DATASET_DIR/data.yaml" \
    model=yolov8s.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    patience=20 \
    project="$RUNS_DIR" \
    name=player_detector_v1 \
    device=0

echo "===== Job finished at $(date) ====="