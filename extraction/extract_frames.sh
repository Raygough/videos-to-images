#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=extract_frames_%j.log

module reset
module load miniconda
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh

conda activate /nfs/roberts/pi/pi_aw989/rjg73/ycrc_conda/envs/frame-extraction

python /nfs/roberts/pi/pi_aw989/rjg73/videos-to-images/extraction/main.py /nfs/roberts/pi/pi_aw989/share/videos /nfs/roberts/pi/pi_aw989/rjg73/frames