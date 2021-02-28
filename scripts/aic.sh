#!/bin/bash
#SBATCH --job-name=aicity2021
#SBATCH --output=scripts/aicity2021.out.%j
#SBATCH --error=scripts/aicity2021.out.%j
#SBATCH --time=24:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --mem=16gb
set -x
module add rclone/1.51.0

source ~/.bashrc
conda deactivate
conda activate detectron2


rsync -v -a /fs/diva-scratch/aicity_2021/AIC21_Track1_Vehicle_Counting /scratch0/vshenoy/

python src/main.py config/cam_10.ini
#rclone copy /vulcanscratch/vshenoy/VehicleCounting/src/vc_outputs/deepsort/counting_output/cam_1/cam_1.avi gdrive:/deepsort/

rm -rf /scratch0/vshenoy