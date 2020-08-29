from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker 
from deep_sort.application_util import preprocessing as prep
from deep_sort.application_util import visualization
from deep_sort.deep_sort.detection import Detection
#from nanonets_object_tracking.deep_sort.siamese_net import *
from deep_sort.siamese_net import *
from tqdm import tqdm

import numpy as np
import pickle

import matplotlib.pyplot as plt

import torch
import torchvision
from scipy.stats import multivariate_normal

import sys
import os
import configparser

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')

class DeepsortTracker:

    def __init__(self):
        self.basic = basic_config['DEFAULT']
        self.config = config['TRACKING']
        self.default = config['DEFAULT']
        self.cam_ident = self.default['cam_name']
        self.out_dir = os.path.join(self.default['output_dir'], self.basic['job_name'], 'tracker_output', self.cam_ident) #set output directory
        os.makedirs(self.out_dir, exist_ok=True) #Create tracker_output folder

        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)
        #self.tracker = Tracker(metric)
    

    def get_deepsort_format(self, detections, features):

        N = len(detections)
        detection_list = []

        THRESHOLD = float(self.config['THRESHOLD'])

        for i in range(0, N):#for every detections

            xmin = detections[i][0]
            ymin = detections[i][1]
            xmax = detections[i][2]
            ymax = detections[i][3]
            
            bbox = [xmin, ymin, (xmax - xmin), (ymax - ymin)]
            score = detections[i][4]
            feature = features[i, :]

            if score > THRESHOLD:
                detection_list.append(Detection(bbox, score, feature))
        
        return detection_list #list formatted for deepsort

    

    def process_tracker_output(self, frameBox, trackBox, frameCount):

        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            X1, Y1, X2, Y2 = track.to_tlbr()
            ID = int(track.track_id)
            if ID not in trackBox: #If new car, create a tracker entry
                trackBox[ID] = []
            if frameCount not in frameBox: #If new frame, create new tracker entry
                frameBox[frameCount] = []

            #Add tracker results to dictionaryies
            frameBox[frameCount].append((ID, (X1, Y1, X2, Y2)))
            trackBox[ID].append((frameCount, (X1, Y1, X2, Y2)))

        

    def per_class_tracker(self, vehicle_dtcs, vehicle_feat):

        self.tracker = Tracker(self.metric) #Initialize sort object
        frameBox = {}
        trackBox = {}

        for frameCount in tqdm(range(0, len(vehicle_dtcs))): #For every frame

            dets = vehicle_dtcs[frameCount] #get detections for that frame
            features = vehicle_feat[frameCount]

            detection_list = self.get_deepsort_format(dets, features) #get correct format

            self.tracker.predict()
            self.tracker.update(detection_list)

            index = (frameCount + 1) * int(self.config['step']) - 1
            self.process_tracker_output(frameBox, trackBox, index)

        return frameBox, trackBox

    
    def get_det_and_feat(self, detections, features, class_id):
        
        #stores all detections for a certain class
        all_dets = []
        all_feat = []
        for frame_num in sorted(detections.keys()):
            
            frame_dets = []
            frame_feat = []
            N = len(detections[frame_num])
           
            for i in range(0, N): #for all detections in a certain frame

                if detections[frame_num][i][5] == class_id: #
                    
                    #Add detection and features to frame dictionary
                    frame_dets.append(detections[frame_num][i]) 
                    frame_feat.append(features[frame_num][i, :])
      
            
            all_dets.append(frame_dets) #add dets to all_dets[frame_num]
            try:
                frame_feat = np.stack(frame_feat) #stack if non-empty
            except:
                x = 5
                
            all_feat.append(frame_feat) #add features to all_feat[frame_num]
        
        return all_dets, all_feat

    
    def run_deepsort(self):
        
        detection_file = os.path.join(self.default['output_dir'], self.basic['job_name'], 'detection_output', self.cam_ident, self.cam_ident + '.pkl')
        with open(detection_file, 'rb') as f:
            detections = pickle.load(f)

        features_file = os.path.join(self.default['output_dir'], self.basic['job_name'], 'detection_output', self.cam_ident, self.cam_ident + '_features.pkl')
        with open(features_file, 'rb') as f:
            features = pickle.load(f)

    
        for classID, className in zip([3, 6, 8], ['Car', 'Bus', 'Truck']):
            
            dets, feat = self.get_det_and_feat(detections, features, classID)
            frameBox, trackBox = self.per_class_tracker(dets, feat)

            if len(trackBox.keys()) !=0:
                trackName = "Track_" + className + "_" + self.cam_ident + '.pkl'
                frameName = "Frame_" + className + "_" + self.cam_ident + '.pkl'
                trackName = os.path.join(self.out_dir, trackName)
                frameName = os.path.join(self.out_dir, frameName)
                pickle.dump(trackBox, open(trackName, "wb"))
                pickle.dump(frameBox, open(frameName, "wb"))

    

if __name__ == "__main__":
    dp = DeepsortTracker()
    dp.run_deepsort()