#!/bin/bash
#SBATCH --job-name=create_detector_env
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/nfs/roberts/pi/pi_aw989/rjg73/logs/create_env_%j.out
#SBATCH --error=/nfs/roberts/pi/pi_aw989/rjg73/logs/create_env_%j.err

mkdir -p /nfs/roberts/pi/pi_aw989/rjg73/logs

source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh

set -e

echo "===== Job started at $(date) ====="

conda create --prefix /nfs/roberts/pi/pi_aw989/rjg73/ycrc_conda/envs/detector \
    python=3.10 -c conda-forge -y

echo "===== Job finished at $(date) ====="