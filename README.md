# Vehicle Counting
![cam10](https://github.com/vineetrshenoy/VehicleCounting2021/blob/main/images/00001.jpg?raw=true)

This repository contains the code to perform Vehicle Counting as part of Track 1 of the AI City Challenge 2021. See the associated paper [Multi-Class, Multi-Movement Vehicle Counting on Traffic Camera Data](//vineetrshenoy.github.io/data/Multi_Class__Multi_Movement_Vehicle_Counting_on_Traffic_Camera_Data.pdf).

To see the results from running the code, see the videos at [this link](https://drive.google.com/drive/folders/1WRWeo71a_RwlowmFNAPZDS4wa8UGY7Xi?usp=sharing)


## Instructions to run code
The code is heavily dependent on the input data as the trajectories are generated from the input. Unfortunately, this data must be obtained directly from the AI City Challenge Organizers. If you do obtain the data, here is how to run the code.



1. Extract the data to your location of choice
2. Change the file `config/basic.ini` and `[DEFAULT][data_dir]` to point to your data location
3. Install the conda environment file
4. From the highest level of the repository, run `python src/main.py config/cam_10.ini` for camera 10, for example. Change the config as necessary.

The entire workflow is run from the configuation files in `config/`. Here you will find the hyperparameters as well. Feel free to modify it as necessary


