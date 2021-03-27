#!/bin/bash
#SBATCH --job-name=group1
#SBATCH --output=scripts/group1.out.%j
#SBATCH --error=scripts/group1.out.%j
#SBATCH --time=3-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --mem=16gb
set -x
module add rclone/1.51.0
#
source ~/.bashrc
conda deactivate
conda activate detectron2
#
echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
#
rsync -v -a /fs/diva-scratch/aicity_2021/AIC21_Track1_Vehicle_Counting /scratch0/vshenoy/

start=`date +%s`

python src/main.py config/cam_1.ini
rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_1/cam_1.avi gdrive:/aicity2021/trackermot/



#python src/main.py config/cam_3.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_3/cam_3.avi gdrive:/aicity2021/trackermot/
#
#
#
python src/main.py config/cam_5.ini
rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_5/cam_5.avi gdrive:/aicity2021/trackermot/
#
#
#
#python src/main.py config/cam_7.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_7/cam_7.avi gdrive:/aicity2021/trackermot/
#
#
#python src/main.py config/cam_9.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_9/cam_9.avi gdrive:/aicity2021/online
#
#python src/main.py config/cam_10.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_10/cam_10.avi gdrive:/aicity2021/online
#
#python src/main.py config/cam_11.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_11/cam_11.avi gdrive:/aicity2021/online
#
#python src/main.py config/cam_12.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_12/cam_12.avi gdrive:/aicity2021/online
#
#rm -rf /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/
#
#
#
#python src/main.py config/cam_13.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_13/cam_13.avi gdrive:/aicity2021/online
#
#
#python src/main.py config/cam_14.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_14/cam_14.avi gdrive:/aicity2021/online
#
#python src/main.py config/cam_15.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_15/cam_15.avi gdrive:/aicity2021/online
#
#python src/main.py config/cam_16.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_16/cam_16.avi gdrive:/aicity2021/online
#
#rm -rf /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/
#
#
#python src/main.py config/cam_17.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_17/cam_17.avi gdrive:/aicity2021/online
#
#python src/main.py config/cam_18.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_18/cam_18.avi gdrive:/aicity2021/online
#
#python src/main.py config/cam_19.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_19/cam_19.avi gdrive:/aicity2021/online
#
#python src/main.py config/cam_20.ini
#rclone copy /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/trackermot/counting_output/cam_20/cam_20.avi gdrive:/aicity2021/online


#rm -rf /fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/


end=`date +%s`
echo runtime=$((end-start))

#rm -rf /scratch0/vshenoy