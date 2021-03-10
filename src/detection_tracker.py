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
        self.tracker = SortTracker()
        self.counter = BezierOnline()
        self.video = os.path.join(self.detector.basic['data_dir'], 
            self.detector.default['cam_name']) + '.mp4'
        
        
    
    def get_model_input(self, img):
        
        height, width = img.shape[:2]
        img_pred = self.detector.aug.get_transform(img).apply_image(img)
        img_pred = img_pred.transpose(2, 0, 1)
        img_pred = torch.from_numpy(img_pred)
        inputs = {'image': img_pred, 'height':height, 'width': width}
        
        return inputs



    def workflow(self):

        cap = cv2.VideoCapture(self.video)
        detection_dict = {}
        frame_num = 1
        print('Starting Detections')
        while (cap.isOpened()):
            
            ret, frame = cap.read() #Read the frame
            if ret == False:
                break
            
            ####Detection portion
            inputs = self.get_model_input(frame)
            with torch.no_grad():
                pred = self.detector.model([inputs])[0]

            detections, all_dets = self.detector.process_outputs(pred)
            detection_dict[frame_num] = detections

            ####TrackerPortion
            self.tracker.update_trackers(all_dets, frame_num)

            if frame_num % 75 == 0:
                print('Frame Number {}'.format(frame_num))
                self.tracker.write_outputs()
                self.counter.workflow()
                self.tracker.flush()

            frame_num += 1
        
        outfile = os.path.join(self.detector.out_dir, 
            self.detector.cam_ident + '.pkl' )
        with open(outfile, 'wb') as handle:
            pickle.dump(detection_dict, handle)

        self.tracker.write_outputs()
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