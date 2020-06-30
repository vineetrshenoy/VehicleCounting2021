#!/bin/bash
#SBATCH --job-name=aicity
#SBATCH --output=aicity.out.%j
#SBATCH --error=aicity.out.%j
#SBATCH --time=12:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
set -x


singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python efficiency_base.py

start=`date +%s`
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_1.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_1_dawn.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_1_rain.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_2.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_2_rain.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_3.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_3_rain.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_4.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_4_dawn.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_4_rain.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_5.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_5_dawn.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_5_rain.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_6.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_6_snow.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_7.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_7_dawn.ini
singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_7_rain.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_8.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_9.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_10.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_11.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_12.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_13.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_14.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_15.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_16.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_17.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_18.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_19.ini

singularity exec --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt detectron.sif python src/main.py config/cam_20.ini


end=`date +%s`
echo runtime=$((end-start))

