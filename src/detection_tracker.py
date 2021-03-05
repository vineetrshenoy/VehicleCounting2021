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
from helper import Helper

from detect_detectron2 import DetectDetectron
from sort_tracker import SortTracker

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
        self.tracker = SortTracker()
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

            if frame_num % 20 == 0:
                print('Frame Number {}'.format(frame_num))

            frame_num += 1
        
        outfile = os.path.join(self.detector.out_dir, 
            self.detector.cam_ident + '.pkl' )
        with open(outfile, 'wb') as handle:
            pickle.dump(detection_dict, handle)

        self.tracker.write_outputs()



if __name__ == '__main__':

    DetectionTracker().workflow()


    '''
    video_name = '/fs/diva-scratch/aicity_2021/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_13.mp4'
    cap = cv2.VideoCapture(video_name)

    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imwrite('frame.png', frame)
    '''



    print('Hello World')