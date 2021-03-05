import os
from sort import Sort
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
                
        self.car_tracker = Sort() #Initialize sort object
        self.bus_tracker = Sort()
        self.truck_tracker = Sort()
        
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

        car_detections = np.array(detections[0])
        bus_detections = np.array(detections[1])
        truck_detections = np.array(detections[2])

        car_output = self.car_tracker.update(car_detections)
        bus_output = self.bus_tracker.update(bus_detections)
        truck_output = self.truck_tracker.update(truck_detections)

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

    