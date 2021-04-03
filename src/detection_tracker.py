import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pickle
import numpy as np
import cv2
import configparser
import torch
import time
import app_logger
import subprocess
from helper import Helper

from detect_detectron2 import DetectDetectron
from sort_tracker import SortTracker
from tracker_deepsort import DeepsortTracker
from tracker_mot import MOTTracker
from bezier_online import BezierOnline


logger = app_logger.get_logger('detectron')

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')

config = configparser.ConfigParser()
config.read(sys.argv[1])


class DetectionTracker:


    ##
    # Initializes the DetectionTracker object
    # 
    #
    def __init__(self):

        self.basic = basic_config['DEFAULT']
        self.default = config['DEFAULT']
        self.config = config['DETECTION']

        self.detector = DetectDetectron()
        self.tracker = MOTTracker()
        #self.tracker = DeepsortTracker()
        #self.tracker = SortTracker()
        self.counter = BezierOnline()
        self.video = os.path.join(self.detector.basic['data_dir'], 
            self.detector.default['cam_name']) + '.mp4'
        
     
    def workflow(self):

        cap = cv2.VideoCapture(self.video)
        detection_dict = {}
        frame_num = 1
        print('Starting Detections')


        start_process_time = time.time()

        while (cap.isOpened()):
            
            ret, frame = cap.read() #Read the frame
            if ret == False:
                break
            
            #####################################Detection portion
            inputs = self.detector.get_model_input(frame)
            with torch.no_grad():
                #pred = self.detector.model([inputs])
                pred, features = self.detector.inference([inputs])
            
            assert len(pred[0]['instances']) == features.shape[0]
            dets, all_dets = self.detector.mask_outputs(pred[0], features)     
            detection_dict[frame_num] = dets
            
            ######################################TrackerPortion
            self.tracker.update_trackers(all_dets, frame_num)
            
            if frame_num % 200 == 0:
                print('Frame Number {}'.format(frame_num))
                self.tracker.write_outputs()
                self.counter.workflow()
                self.tracker.flush()
                '''
                outfile = os.path.join(self.detector.out_dir, 
                    self.detector.cam_ident + '.pkl' )
                with open(outfile, 'wb') as handle:
                    pickle.dump(detection_dict, handle)
                '''
            
            frame_num += 1
        
        end_process_time = time.time()
        elapsed = end_process_time - start_process_time

        print('Elapsed: {} seconds'.format(elapsed))
        print('Num of Frames: {}'.format(frame_num))
        
        outfile = os.path.join(self.detector.out_dir, 
            self.detector.cam_ident + '.pkl' )
        with open(outfile, 'wb') as handle:
            pickle.dump(detection_dict, handle)

        self.tracker.write_outputs()
        self.counter.workflow()
        self.counter.track1txt.close()
        '''
        pid = subprocess.Popen([sys.executable, "bezier_online.py config/cam_13.ini"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            stdin=subprocess.PIPE)
        '''
        print('Done')

if __name__ == '__main__':

    DetectionTracker().workflow()
    #pid = subprocess.Popen(["python", "/fs/diva-scratch/vshenoy/VehicleCounting/src/bezier_online.py /fs/diva-scratch/vshenoy/VehicleCounting/config/cam_13.ini"]).pid
    '''
    filename = '/fs/diva-scratch/vshenoy/VehicleCounting/vc_outputs/aic/tracker_output/cam_13/bezier_idx.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    idx = data[1]
    for list1 in idx:
        for list2 in list1:
            print(list2.hits)
    '''

    print('Hello World')