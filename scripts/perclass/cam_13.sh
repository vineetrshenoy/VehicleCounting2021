#!/bin/bash

set -x
module add rclone/1.51.0
#


start=`date +%s`



CUDA_VISIBLE_DEVICES=8 nohup python src/main.py config/cam_13.ini > scripts/cam_13.out & disown
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_10/cam_10.avi gdrive:/aicity2021/online

#CUDA_VISIBLE_DEVICES=8 python src/main.py config/cam_13.ini > scripts/cam_13.out

end=`date +%s`
#echo runtime=$((end-start))
