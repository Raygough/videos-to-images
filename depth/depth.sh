#!/bin/bash
#SBATCH --job-name=unidepth_extract
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=unidepth_extract_%j.log
 
module load miniconda
conda activate /nfs/roberts/pi/pi_aw989/rjg73/ycrc_conda/envs/unidepth_env

python /nfs/roberts/pi/pi_aw989/rjg73/videos-to-images/depth/uni_loop.py \
        --frames_root /nfs/roberts/pi/pi_aw989/rjg73/frames \
        --output_root /nfs/roberts/pi/pi_aw989/rjg73/videos-to-images/depth/depth_maps \
        --model_name lpiccinelli/unidepth-v2-vitl14 \
        --device cuda \