#!/bin/bash
#SBATCH --job-name=aicity2021
#SBATCH --output=scripts/aicity2021.out.%j
#SBATCH --error=scripts/aicity2021.out.%j
#SBATCH --time=3-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --mem=16gb
set -x

source ~/.bashrc
conda deactivate
conda activate detectron2

echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
#
rsync -v -a /fs/diva-scratch/aicity_2021/AIC21_Track1_Vehicle_Counting /scratch0/vshenoy/

python src/efficiency_base.py


start=`date +%s`

python src/main.py config/cam_1.ini
python src/main.py config/cam_1_dawn.ini
python src/main.py config/cam_1_rain.ini
python src/main.py config/cam_2.ini
python src/main.py config/cam_2_rain.ini
python src/main.py config/cam_3.ini
python src/main.py config/cam_3_rain.ini
python src/main.py config/cam_4.ini
python src/main.py config/cam_4_dawn.ini
python src/main.py config/cam_4_rain.ini
python src/main.py config/cam_5.ini
python src/main.py config/cam_5_dawn.ini
python src/main.py config/cam_5_rain.ini
python src/main.py config/cam_6.ini
python src/main.py config/cam_6_snow.ini
python src/main.py config/cam_7.ini
python src/main.py config/cam_7_dawn.ini
python src/main.py config/cam_7_rain.ini
python src/main.py config/cam_8.ini
python src/main.py config/cam_9.ini
python src/main.py config/cam_10.ini
python src/main.py config/cam_11.ini
python src/main.py config/cam_12.ini
python src/main.py config/cam_13.ini
python src/main.py config/cam_14.ini
python src/main.py config/cam_15.ini
python src/main.py config/cam_16.ini
python src/main.py config/cam_17.ini
python src/main.py config/cam_18.ini
python src/main.py config/cam_19.ini
python src/main.py config/cam_20.ini


end=`date +%s`
echo runtime=$((end-start))

