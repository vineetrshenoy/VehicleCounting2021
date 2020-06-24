import sys
import os
import pickle
import numpy as np
import cv2
import configparser
import time

import app_logger
from helper import Helper

from tqdm import tqdm
from PIL import Image

sys.path.append('src/automl/')
sys.path.append('src/automl/efficientdet')
sys.path.append('src/automl/efficientdet/object_detection')
import efficientdet.inference as inference

logger = app_logger.get_logger('efficientdet_detections')

config = configparser.ConfigParser()
config.read(sys.argv[1])




class EfficientDet:


    def __init__(self):
        self.default = config['DEFAULT']
        self.config = config['DETECTION']
        self.cam_ident = self.default['cam_name']
        self.out_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'detection_output', self.cam_ident)
        os.makedirs(self.out_dir, exist_ok=True)
        self.roi = Helper.get_roi(self.default['roi'])
        self.model_name = self.config['model_name']
        self.ckpt = self.config['ckpt_path']
        self.input_size = (int(self.default['width']),int(self.default['height']))
        self.driver = inference.ServingDriver(self.model_name, self.ckpt, batch_size=1)
        self.driver.build()
        
        ##Video visualization
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_name = os.path.join(self.out_dir, self.cam_ident + '.avi')
        frame_dim = (int(self.default['width']), int(self.default['height']))
        self.out_video = cv2.VideoWriter(video_name, self.fourcc, int(self.default['fps']), frame_dim)


    def get_roi(self, video_id):
        roi_list = []        
        
        with open(self.config['roi_path'], 'r') as roi_file:
            for point in roi_file.readlines():
                roi_list.append((int(point.split(',')[0]), int(point.split(',')[1])))
        
        return roi_list
    
    
    def create_roi_mask(self, roi_polygon, image_path):
        image = cv2.imread(image_path, -1)   
        image = cv2.resize(image, self.input_size, interpolation = cv2.INTER_AREA) 
        mask = np.zeros(image.shape, dtype=np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
        cv2.fillConvexPoly(mask, np.array(roi_polygon, dtype=np.int32), ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)

        # save the result
        #cv2.imwrite('image_masked.png', masked_image)
        return masked_image


    def process_predictions(self, predictions):

        N = predictions.shape[0]

        car_detections = []
        bus_detections = []
        truck_detections = []

        for i in range(0, N):

            p = predictions[i]
            
            tup = (p[1], p[2], p[3], p[4], p[5])
            cat = p[6]

            if scores[i].item() > 0.7:

                if cat == 3:
                    car_detections.append(tup)
                elif cat == 6:
                    bus_detections.append(tup)
                else:
                    truck_detections.append(tup)

        detections = [car_detections, bus_detections, truck_detections]

        return detections

    def run_predictions(self):

        detection_dict = {}
        files = sorted(os.listdir(os.path.join(self.default['data_dir'], self.cam_ident)))

        roi_list = self.get_roi(self.config['roi_path'])
        roi_polygon = np.array(roi_list)
        #Load the images
        
        for i in tqdm(range(0, len(files), int(self.config['step']))):
            file_path  = os.path.join(self.default['data_dir'], self.cam_ident, files[i])
            img = self.create_roi_mask(roi_polygon, file_path)
            
            predictions = self.driver.serve_images([img])

            detections = self.process_predictions(predictions[0])


            x = 5



        




if __name__ == '__main__':
    EfficientDet().run_predictions()
    print('Done')