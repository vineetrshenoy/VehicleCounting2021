#! /usr/bin/bash
cd ./src
CUDA_VISIBLE_DEVICES=0 python demo.py tracking \
    --load_model ../models/coco_tracking.pth \
    --demo /fs/diva-scratch/nvidia_data_2020/NVIDIA_AIC_2020_Datasets/AIC20_track1/Dataset_A/cam_13.mp4 \
    --resize_video \
    --video_h 1080 \
    --video_w 1920 \
    --save_framerate 10 \
    --save_video \
    --save_results \
    --track_thresh 0.3 \
    --max_age -1
