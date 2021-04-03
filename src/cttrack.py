import sys
import configparser
import app_logger
import os
import cv2
import time
import numpy as np
import pickle

CENTERTRACK_PATH = '/fs/diva-scratch/vshenoy/VehicleCounting/src/CenterTrack/src/lib'
sys.path.insert(0, CENTERTRACK_PATH)

logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')

from helper import Helper
from detector import Detector
from opts import opts
from CenterTrack.src.demo import demo
from ctdetection import CTDetection
from tracker_ct import TrackerCT


class CTTrack:

    def __init__(self):

        self.basic = basic_config['DEFAULT']
        self.det_conf = config['DETECTION']
        self.tr_conf = config['TRACKING']
        self.default = config['DEFAULT']

        self.ctdets = CTDetection()
        self.cttracker = TrackerCT()

        self.video = os.path.join(self.ctdets.basic['data_dir'], 
            self.ctdets.default['cam_name']) + '.mp4'

        
        

            
    def workflow(self):
        #CTHelper.mask_outputs()
        cap = cv2.VideoCapture(self.video)
        
        detection_dict = {}
        frame_num = 1
        print('Starting Detections')


        start_process_time = time.time()

        while (cap.isOpened()):
            
            ret, frame = cap.read() #Read the frame
            if ret == False:
                break
            
            ######################### Detection
            img = self.ctdets.get_model_input(frame)
            ret = self.ctdets.detector.run(img)
            dets, all_dets = self.ctdets.mask_outputs(ret['results'])
            detection_dict[frame_num] = dets
            #########################

            ######################### Tracking
            self.cttracker.update_trackers(ret['results'], frame_num)
            #########################

        
            print('Frame Num: {}'.format(frame_num))
            frame_num +=1



        self.cttracker.write_outputs()

        outfile = os.path.join(self.ctdets.out_dir, 
            self.default['cam_name'] + '.pkl' )

        with open(outfile, 'wb') as handle:
            pickle.dump(detection_dict, handle)



if __name__ == '__main__':
    
    CTTrack().workflow()

    #opt = opts().init()
    #demo(opt)