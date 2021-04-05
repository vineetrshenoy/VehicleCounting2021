#!/bin/bash

source ~/.bashrc
conda deactivate
conda activate detectron2
#


start=`date +%s`



CUDA_VISIBLE_DEVICES=7 nohup python src/main.py config/cam_12.ini > scripts/cam_12.out & disown
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_10/cam_10.avi gdrive:/aicity2021/online

#CUDA_VISIBLE_DEVICES=7 python src/main.py config/cam_12.ini > scripts/cam_12.out


end=`date +%s`
#echo runtime=$((end-start))
