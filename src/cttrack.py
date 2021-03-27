import sys
import configparser
import app_logger
import os

CENTERTRACK_PATH = '/fs/diva-scratch/vshenoy/VehicleCounting/src/CenterTrack/src/lib'
sys.path.insert(0, CENTERTRACK_PATH)

logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')

from detector import Detector
from opts import opts
from CenterTrack.src.demo import demo


class CTTrack:

    def __init__(self):

        self.basic = basic_config['DEFAULT']
        self.config = config['TRACKING']
        self.default = config['DEFAULT']
        self.cam_ident = self.default['cam_name']
        
        self.out_dir = os.path.join(self.basic['output_dir'], 
            self.basic['job_name'], 'tracker_output', self.cam_ident) #set output directory
        os.makedirs(self.out_dir, exist_ok=True) #Create tracker_output folder

        opt = self.build_opts()

        demo(opt)


    def build_opts(self):

        task = 'tracking'
        load_model = 'src/CenterTrack/models/coco_tracking.pth'
        demo = '/fs/diva-scratch/nvidia_data_2020/NVIDIA_AIC_2020_Datasets/AIC20_track1/Dataset_A/cam_13.mp4'
        video_h = 1080
        video_w = 1920
        save_framerate = 10
        track_thresh = 0.3
        max_age = 5

        opt_str = 'cam10.ini --task {} --load_model {} --demo {} --resize_video --video_h {} --video_w {} --save_framerate {} --save_video --save_results --track_thresh {} --max_age {}'.format(
                task,
                load_model,
                demo,
                video_h,
                video_w,
                save_framerate,
                track_thresh,
                max_age
            ).split(' ')
        
        opt = opts().init(opt_str)

        return opt
            





if __name__ == '__main__':
    
    CTTrack()

    #opt = opts().init()
    #demo(opt)