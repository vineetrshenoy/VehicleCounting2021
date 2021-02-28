# Vehicle Counting
This document explains how to download and run the vehicle counting algorithm. The containers for this project are built using [Singularity](https://sylabs.io/guides/3.4/user-guide/)

## Instructions to create Container
1. Clone the Github respository
2. Run the command `mkdir logs && touch logs/app.log`
3. (Optional) Create a folder in `src/` called `checkpoints` and use the `wget` command to download checkpoints from the [EfficientDet repository](https://github.com/google/automl/tree/master/efficientdet). This is only needed if using the EfficientDet Detector.
4. Build the container. If using the EfficientDet detector, run `singularity build --fakeroot tensorflow.sif tensorflow.def`. If using detectron2 detectors, run `singularity build --fakeroot detectron.sif detectron.def`. The `.sif` files are images while the `.def` files are build recipes.

Building the `.sif` files allow us to completely encapsulate the code (for example, check the operating system in the container vs that of your system). I would recommend you check [Interacting with Images](https://sylabs.io/guides/3.4/user-guide/quick_start.html#interact-with-images) to get a better idea of what you can do with the images


## Instructions to run code
1. Shell into the container using `singularity shell --nv --bind /vulcan/scratch/vshenoy/aicity2020:/mnt tensorflow.sif`. The `--nv` flag instructs the container to use the GPUs. The command `/vulcan/scratch/vshenoy/aicity2020:/mnt` mounts an external folder to the `:/mnt` area of the container. The external folder path plus the `data_dir` location in the configuration file should be the directory which containers folders of images. You'll notice that you will get a `File Not Found` error when trying to access files outside the subdirectory of your container. Try some commands like `ls` and `pwd`. 
2. Run the entire workflow like `python src/main.py config/cam_1.ini`. Always run the command from the same folder as `src/`

The entire workflow is run from the configuation files in `config/`