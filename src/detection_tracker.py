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
from datetime import datetime

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

        self.time = time.time()

        self.detector = DetectDetectron()
        self.tracker = MOTTracker()
        #self.tracker = DeepsortTracker()
        #self.tracker = SortTracker()
        self.bs = int(self.basic['batch_size'])
        self.counter = BezierOnline()
        self.video = os.path.join(self.detector.basic['data_dir'], 
            self.detector.default['cam_name']) + '.mp4'
    

    def get_batch(self, cap):

        count = 0
        inputs = []
        while count < self.bs:

            ret, frame = cap.read() #Read the frame
            if ret == False:
                break
            
            count +=1
            inp = self.detector.get_model_input(frame)
            inputs.append(inp)

        return inputs, count
     
    def workflow(self):
        print('{}'.format(datetime.now()))
        cap = cv2.VideoCapture(self.video)
        detection_dict = {}
        rem = 0
        frame_num = 1
        print('Starting Detections')


        start_process_time = time.time()

        while (cap.isOpened()):
            

            #####################################Detection portion
            inputs, count = self.get_batch(cap)
            if count == 0:
                break
                
            with torch.no_grad():
                #pred = self.detector.model([inputs])
                assert len(inputs) != 0
                pred, features = self.detector.inference(inputs)

            len_feat = 0 
            for i in range(0, count):
                len_feat_i = len(pred[i]['instances'])
                features_i = features[len_feat:(len_feat + len_feat_i), :]
                
                assert len(pred[i]['instances']) == features_i.shape[0]
                dets, all_dets = self.detector.mask_outputs(pred[i], features_i)
                detection_dict[frame_num + i] = dets
            
                ######################################TrackerPortion
                self.tracker.update_trackers(all_dets, frame_num)
            
            frame_num += count
            if frame_num // 100 > rem:
                print('Frame Number {}'.format(frame_num))
                self.tracker.write_outputs()
                self.counter.workflow()
                self.tracker.flush()
                rem = frame_num // 100
                
                outfile = os.path.join(self.detector.out_dir, 
                    self.detector.cam_ident + '.pkl' )
                with open(outfile, 'wb') as handle:
                    pickle.dump(detection_dict, handle)
                
            
        
        end_process_time = time.time()
        elapsed = end_process_time - start_process_time

        print('Elapsed {}: {} seconds'.format(self.detector.default['cam_name'], elapsed))
        print('Num of Frames: {}'.format(frame_num))
        
        outfile = os.path.join(self.detector.out_dir, 
            self.detector.cam_ident + '.pkl' )
        with open(outfile, 'wb') as handle:
            pickle.dump(detection_dict, handle)

        self.tracker.write_outputs()
        self.counter.workflow()
        #self.counter.track1txt.close()
        self.counter.percam_txt.close()
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