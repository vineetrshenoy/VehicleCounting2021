#!/bin/bash

set -x
module add rclone/1.51.0
#
source ~/.bashrc
conda deactivate
conda activate detectron2

start=`date +%s`


#python src/main.py config/cam_10.ini
CUDA_VISIBLE_DEVICES=4 nohup python src/main.py config/cam_10.ini > scripts/cam_10.out & disown
#CUDA_VISIBLE_DEVICES=4 python src/main.py config/cam_10.ini 
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_10/cam_10.avi gdrive:/aicity2021/online



end=`date +%s`
echo runtime=$((end-start))
