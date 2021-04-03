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


from detector import Detector
from opts import opts
from helper import Helper

class CTDetection:

    def __init__(self):

        self.basic = basic_config['DEFAULT']
        self.default = config['DEFAULT']
        self.config = config['DETECTION']
                
        
        self.cam_ident = self.default['cam_name']
        self.out_dir = os.path.join(self.basic['output_dir'], self.basic['job_name'], 
            'detection_output', self.cam_ident)
        os.makedirs(self.out_dir, exist_ok=True)
        self.roi = Helper.get_roi(self.default['roi'])

        opt = self.build_opts()
        self.detector = Detector(opt)


    def build_opts(self):

        task = 'tracking'
        load_model = 'src/CenterTrack/models/coco_tracking.pth'
        demo = '/fs/diva-scratch/nvidia_data_2020/NVIDIA_AIC_2020_Datasets/AIC20_track1/Dataset_A/cam_13.mp4'
        video_h = int(self.default['height'])
        video_w = int(self.default['width'])
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


    def get_model_input(self, image):
            
        #image = cv2.imread(image_path, -1)    
        mask = np.zeros(image.shape, dtype=np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
        cv2.fillConvexPoly(mask, np.array(self.roi, dtype=np.int32),
             ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)

        # save the result
        #cv2.imwrite('image_masked.png', masked_image)
        return masked_image

    ##
    #   Creates object structure for subsequent tracking
    #   @param raw_detections The detections for a single frame 
    #   @returns detections The processed detections
    #
    
    def mask_outputs(self, dets):

        N = len(dets)
        #Initialize detection structure
        car_detections = []
        bus_detections = []
        truck_detections = []
                
        for i in range(0, N): #for each box
            
            det = dets[i]
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            cat = det['class'] #category
            feats = np.zeros(3)
            
            tup = (x1, y1, x2, y2, score, cat, feats)
            
            
            #bbPath = mplPath.Path(self.roi) #gets the ROI coordinates

            #if contains point, add to list of detections
            if True:
            #if bbPath.contains_point((centers[0].item(), centers[1].item())): #if inside the region of interest
                
                if score > 0.5: #float(self.config['score_thresh']):

                    if cat == 3:
                        car_detections.append(tup)
                        
                    elif cat == 6:
                        bus_detections.append(tup)

                    elif cat == 8:
                        truck_detections.append(tup)
                        

        #Perform NMS per-class
        #car_detections = CTHelper.perform_nms(car_detections)
        #bus_detections = CTHelper.perform_nms(bus_detections)
        #truck_detections = CTHelper.perform_nms(truck_detections)
        
        detections = car_detections + bus_detections + truck_detections
        all_dets = {'Car': car_detections, 'Bus': bus_detections, 'Truck': truck_detections}
        
        return detections, all_dets


    @staticmethod
    def perform_nms(dets):

        N = len(dets)
        boxes = np.zeros((1, 4))
        scores = np.zeros(1)
        for i in range(0, N):

            det = dets[i]
            bbox = np.array([det[0], det[1], det[2], det[3]])
            boxes = np.vstack((boxes, bbox))
            nscore = np.array([det[4]])
            scores = np.vstack((scores, nscore))


        box_tens = torch.from_numpy(boxes[1:, :])
        score_tens = torch.from_numpy(scores[1:, :])

        idx = torchvision.ops.nms(box_tens, score_tens, 0.5)

        nms_dets = list(map(lambda x: dets[x], idx))

        return nms_dets
