#!/bin/bash

set -x
module add rclone/1.51.0
#
source ~/.bashrc
conda deactivate
conda activate detectron2

start=`date +%s`



CUDA_VISIBLE_DEVICES=3 nohup python src/main.py config/cam_1.ini > scripts/cam_1.out & disown
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_10/cam_10.avi gdrive:/aicity2021/online



end=`date +%s`
echo runtime=$((end-start))
