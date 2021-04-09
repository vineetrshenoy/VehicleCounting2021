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


from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker 
from deep_sort.application_util import preprocessing as prep
from deep_sort.application_util import visualization
from deep_sort.deep_sort.detection import Detection

logger = app_logger.get_logger('tracker')

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')



class DeepsortTracker():

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
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)


        self.trackers = {
            'Car' : Tracker(self.metric),
            'Bus' : Tracker(self.metric),
            'Truck': Tracker(self.metric)
        }
        
        self.tracklets = {'Car': set(), 'Bus': set(), 'Truck': set()}

        self.frameBox = {'Car': {}, 'Bus': {}, 'Truck': {}}
        self.trackBox = {'Car': {}, 'Bus': {}, 'Truck': {}}

    ##
    # Draws the bounding boxes on the image frame and adds to video
    # @param tracker_output Output of the sort.py code
    # @param frameBox Dictionary indexed by frame number with tracking results
    # @param trackBox Dictionary indexed by objectID with tracking results
    # @param frameNumber The frame count to put in correct order
    #
    def process_tracker_output(self, tracker_output, frameBox, trackBox, frameCount):
        
        if frameCount not in frameBox: #If new frame, create new tracker entry
            frameBox[frameCount] = []
        
        for trk in tracker_output.tracks:

            X1, Y1, X2, Y2 = trk.to_tlbr()
            ID = trk.track_id
            if ID not in trackBox: #If new car, create a tracker entry
                trackBox[ID] = []
            if frameCount not in frameBox: #If new frame, create new tracker entry
                frameBox[frameCount] = []

            #Add tracker results to dictionaryies
            frameBox[frameCount].append((ID, (X1, Y1, X2, Y2)))
            trackBox[ID].append((frameCount, (X1, Y1, X2, Y2)))

   
    
    def deepsort_format(self, detections):

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

            
            detection_list.append(Detection(bbox, score, feature))
        
        return detection_list #list formatted for deepsort
    

    def update_trackers(self, dets, frame_num):

        for cname  in ['Car', 'Bus', 'Truck']:

            clsdets = self.deepsort_format(dets[cname])
            self.trackers[cname].predict()
            cls_removed_ids = self.trackers[cname].update(clsdets)
            self.tracklets[cname].update(cls_removed_ids)
        
            self.process_tracker_output(self.trackers[cname], self.frameBox[cname], 
                    self.trackBox[cname], frame_num - 1)
       
        
        #return [car_output, bus_output, truck_output]


##
    # Helper function to the write_outputs function
    # 
    #
    def per_class_writer(self, frameBox, trackBox, cat):

        # Save frameBox and trackBox as a pickle file and return 
        #if len(trackBox.keys()) !=0:
        trackName = "Track_" + cat + "_" + self.cam_ident + '.pkl'
        frameName = "Frame_" + cat + "_" + self.cam_ident + '.pkl'
        trackName = os.path.join(self.out_dir, trackName)
        frameName = os.path.join(self.out_dir, frameName)
        pickle.dump(trackBox, open(trackName, "wb"))
        pickle.dump(frameBox, open(frameName, "wb"))

    ##
    # Writes the outputs of the tracker to disk periodically
    # How often this is run is determiend by the detection_tracker module
    #
    def write_outputs(self):
        # Save frameBox and trackBox as a pickle file and return 

        
        for cname  in ['Car', 'Bus', 'Truck']:
            self.per_class_writer(self.frameBox[cname], self.trackBox[cname], cname)
    
        bezier_idx = {'Car': self.tracklets['Car'],
            'Bus': self.tracklets['Bus'],
            'Truck': self.tracklets['Truck']}
       
        bezier_outname = os.path.join(self.out_dir, 'bezier_idx.pkl')
        pickle.dump(bezier_idx, open(bezier_outname, "wb"))

        self.flush()

    ##
    # Removes the dead tracklets so the counting doesn't happen twice
    # 
    #
    def flush(self):
        
        for cname  in ['Car', 'Bus', 'Truck']:
            self.tracklets[cname] = set()
    