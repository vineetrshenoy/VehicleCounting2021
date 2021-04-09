import os
from sort_online import Sort
import numpy as np
import cv2
import sys
from tqdm import tqdm
import pickle
import app_logger
import configparser

logger = app_logger.get_logger('tracker')

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')



class SortTracker():

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
                
        self.trackers = {
            'Car' : Sort(),
            'Bus' : Sort(),
            'Truck': Sort()
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
        
        for bbox in tracker_output:

            X1, Y1, X2, Y2, ID = bbox
            ID = int(ID)
            if ID not in trackBox: #If new car, create a tracker entry
                trackBox[ID] = []
            if frameCount not in frameBox: #If new frame, create new tracker entry
                frameBox[frameCount] = []

            #Add tracker results to dictionaryies
            frameBox[frameCount].append((ID, (X1, Y1, X2, Y2)))
            trackBox[ID].append((frameCount, (X1, Y1, X2, Y2)))

    

    def update_trackers(self, detections, frame_num):


        for cname  in ['Car', 'Bus', 'Truck']:

            cls_detections = np.array(detections[cname])
            cls_output, cls_tracklet = self.trackers[cname].update(cls_detections)
            if cls_tracklet is not None:
                self.tracklets[cname].update(cls_tracklet)
        
            self.process_tracker_output(cls_output, self.frameBox[cname], 
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

    