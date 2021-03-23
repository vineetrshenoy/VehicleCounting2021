import os
from sort_online import Sort
import numpy as np
import cv2
import sys
from tqdm import tqdm
import pickle
import app_logger
import configparser
import torch

from deep_sort.deep_sort.tracker import Tracker
from tracker_mot_zhong.tr_mot.multitracker import STrack, JDETracker

logger = app_logger.get_logger('tracker')

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')



class MOTTracker():

    ##
    # Initializes the Tracker object
    # 
    #
    def __init__(self):
        self.basic = basic_config['DEFAULT']
        self.config = config['TRACKING']
        self.default = config['DEFAULT']
        self.cam_ident = self.default['cam_name']
        
        self.out_dir = os.path.join(self.basic['output_dir'], 
            self.basic['job_name'], 'tracker_output', self.cam_ident) #set output directory
        os.makedirs(self.out_dir, exist_ok=True) #Create tracker_output folder
        #self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)

        self.buffer_size = 30 #int(feature_buffer_size * fps)
        max_age = 5
        max_time_lost = int(max_age * self.default['fps'])
        feat_thresh = 0.7
        min_iou = 0.2


        self.car_tracker = JDETracker(max_time_lost, feat_thresh, 1 - min_iou, 1 - min_iou/2) #Initialize sort object
        self.bus_tracker = JDETracker(max_time_lost, feat_thresh, 1 - min_iou, 1 - min_iou/2)
        self.truck_tracker = JDETracker(max_time_lost, feat_thresh, 1 - min_iou, 1 - min_iou/2)

        self.tracklets = {'Car': [], 'Bus': [], 'Truck': []}

        self.frameBox = {'Car': {}, 'Bus': {}, 'Truck': {}}
        self.trackBox = {'Car': {}, 'Bus': {}, 'Truck': {}}
        
        self.car_frameBox = {}
        self.car_trackBox = {}
        self.bus_frameBox = {}
        self.bus_trackBox = {}
        self.truck_frameBox = {}
        self.truck_trackBox = {}


    ##
    # Draws the bounding boxes on the image frame and adds to video
    # @param tracker_output Output of the sort.py code
    # @param frameBox Dictionary indexed by frame number with tracking results
    # @param trackBox Dictionary indexed by objectID with tracking results
    # @param frameNumber The frame count to put in correct order
    #
    def process_tracker_output(self, tracker_output, frameBox, trackBox, frameCount):
        
        
        for trk in tracker_output:
            
            ID = trk.track_id
            X1, Y1, X2, Y2 = trk.tlbr
            ID = int(ID)
            if ID not in trackBox: #If new car, create a tracker entry
                trackBox[ID] = []
            if frameCount not in frameBox: #If new frame, create new tracker entry
                frameBox[frameCount] = []

            #Add tracker results to dictionaryies
            frameBox[frameCount].append((ID, (X1, Y1, X2, Y2)))
            trackBox[ID].append((frameCount, (X1, Y1, X2, Y2)))

    
    
    def tracklet_update(self, tracklet, cat):
        
        indices = tracklet[:, 4]
        N = len(indices)
        for i in range(0, N):
            self.tracklets[cat].append(int(indices[i]))

    
    
    def tracker_mot_format(self, detections):

        N = len(detections)
        detection_list = []

        for i in range(0, N):#for every detections

            xmin = detections[i][0]
            ymin = detections[i][1]
            xmax = detections[i][2]
            ymax = detections[i][3]
            
            bbox = [xmin, ymin, (xmax - xmin), (ymax - ymin)]
            score = detections[i][4]
            feature = detections[i][6]
            
            strack_i = STrack(bbox, score, feature, i, self.buffer_size)
            
            detection_list.append(strack_i)
        
        return detection_list #list formatted for deepsort
    

    def update_trackers(self, dets, frame_num):

        car_dets = self.tracker_mot_format(dets[0])
        bus_dets = self.tracker_mot_format(dets[1])
        truck_dets = self.tracker_mot_format(dets[2])


        car_output = self.car_tracker.update(car_dets)
        bus_output  = self.bus_tracker.update(bus_dets)
        truck_output = self.truck_tracker.update(truck_dets)
        '''
        if car_tracklet is not None:
            self.tracklets['Car'] = self.tracklets['Car'] + car_tracklet
            #self.tracklet_update(car_tracklet, 'Car')
        if bus_tracklet is not None:
            self.tracklets['Bus'] = self.tracklets['Bus'] + bus_tracklet
            #self.tracklet_update(bus_tracklet, 'Bus')
        if truck_tracklet is not None:
            self.tracklets['Truck'] = self.tracklets['Truck'] + truck_tracklet
            #self.tracklet_update(truck_tracklet, 'Truck')
        '''
        self.process_tracker_output(car_output, self.car_frameBox, self.car_trackBox, frame_num - 1)
        self.process_tracker_output(bus_output, self.bus_frameBox, self.bus_trackBox, frame_num - 1)
        self.process_tracker_output(truck_output, self.truck_frameBox, self.truck_trackBox, frame_num - 1)

        return [car_output, bus_output, truck_output]


        



    def per_class_writer(self, frameBox, trackBox, cat):

        # Save frameBox and trackBox as a pickle file and return 
        if len(trackBox.keys()) !=0:
            trackName = "Track_" + cat + "_" + self.cam_ident + '.pkl'
            frameName = "Frame_" + cat + "_" + self.cam_ident + '.pkl'
            trackName = os.path.join(self.out_dir, trackName)
            frameName = os.path.join(self.out_dir, frameName)
            pickle.dump(trackBox, open(trackName, "wb"))
            pickle.dump(frameBox, open(frameName, "wb"))



    def write_outputs(self):
        # Save frameBox and trackBox as a pickle file and return 

        self.per_class_writer(self.car_frameBox, self.car_trackBox, 'Car')
        self.per_class_writer(self.bus_frameBox, self.bus_trackBox, 'Bus')
        self.per_class_writer(self.truck_frameBox, self.truck_trackBox, 'Truck')

        bezier_idx = {1: self.tracklets['Car'],
            3: self.tracklets['Bus'],
            2: self.tracklets['Truck']}
       
        bezier_outname = os.path.join(self.out_dir, 'bezier_idx.pkl')
        pickle.dump(bezier_idx, open(bezier_outname, "wb"))

        self.flush()

    def flush(self):
        self.tracklets['Car'] = []
        self.tracklets['Bus'] = []
        self.tracklets['Truck'] = []




    ##
    # Workflow for the tracker object
    # @param detections The Detections file from the previous step in pipeline 
    #
    def run_tracker(self): 

        detection_file = os.path.join(self.basic['output_dir'], self.basic['job_name'], 'detection_output', self.cam_ident, self.cam_ident + '.pkl')
        with open(detection_file, 'rb') as f:
            detections = pickle.load(f)
        
        
        car_dets = [list(filter(lambda x: x[5] == 3, detections[frame])) for frame in sorted(detections.keys())]
        bus_dets = [list(filter(lambda x: x[5] == 6, detections[frame])) for frame in sorted(detections.keys())]
        truck_dets = [list(filter(lambda x: x[5] == 8, detections[frame])) for frame in sorted(detections.keys())]
        
        frame_num = sorted(detections.keys())
        for frame in frame_num:

            car_dets_frame = self.tracker_mot_format(car_dets[frame])
            bus_dets_frame = self.tracker_mot_format(bus_dets[frame])
            truck_dets_frame = self.tracker_mot_format(truck_dets[frame])


            car_tracklet = self.car_tracker.update(car_dets_frame)
            bus_tracklet  = self.bus_tracker.update(bus_dets_frame)
            truck_tracklet = self.truck_tracker.update(truck_dets_frame)

    


if __name__ == '__main__':

    MOTTracker().run_tracker()