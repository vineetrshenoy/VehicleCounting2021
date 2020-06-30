# Vehicle Counting
This document explains how to download and run the vehicle counting algorithm

## Instructions to create Container
1. Clone the Github respository
2. (Optional) Create a folder in `src/` called `checkpoints` and use the `wget` command to download checkpoints from the EfficientDet repository 
3. Build the container. If using the EfficientDet detector, run `singularity build --fakeroot tensorflow.sif tensorflow.def`. If using detectron2 detectors, run `singularity build --fakeroot detectron.sif detectron.def`. This creates the container image

## Instructions to run code
1. Start the container using `singularity shell --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt tensorflow.sif`. This allows you to shell into the container. The `--nv` flag instructs the container to use the GPUs. The command `/vulcan/scratch/vshenoy/aicity2020:/mnt` mounts an external dataset to the `:/mnt` area of the container. This is exactly like a symbolic link
2. Run the entire workflow like `python src/main.py config/cam_1.ini`. Always run the command from the same folder as `src/`