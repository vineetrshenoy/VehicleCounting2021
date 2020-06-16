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


class Tracker():

    ##
    # Initializes the Tracker object
    # 
    #
    def __init__(self):
        self.config = config['TRACKING']
        self.default = config['DEFAULT']
        self.cam_ident = self.default['cam_name']

        
        self.out_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'tracker_output', self.cam_ident) #set output directory
        os.makedirs(self.out_dir, exist_ok=True) #Create tracker_output folder
        print()


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
            
    
    ##
    # Performs tracking on a per-class basis
    # @param vehicle dtcs List of lists, where nested list are detections
    # and each list corresponds to a frame (arranged in order)
    # @returns frameBox, trackBox Tracked objects
    #
    def per_class_tracker(self, vehicle_dtcs):
        
        objectTracker = Sort() #Initialize sort object
        frameBox = {}
        trackBox = {}

        THRESHOLD = float(self.config['THRESHOLD'])

        for frameCount in tqdm(range(0, len(vehicle_dtcs))): #For every frame

            frame = vehicle_dtcs[frameCount] #get detections for that frame
            
            NMS = list(filter(lambda detect: detect[4] > THRESHOLD, frame)) #Get all detections that exceed THRESHOLD
            NMS = np.array(NMS)
            tracker_output = objectTracker.update(NMS) #Update the tracker
            index = (frameCount + 1) * int(self.config['step']) - 1
            self.process_tracker_output(tracker_output, frameBox, trackBox, index) #Get results in format for additional consumption

        objectTracker.reset()
        return frameBox, trackBox




    ##
    # Workflow for the tracker object
    # @param detections The Detections file from the previous step in pipeline 
    #
    def run_tracker(self, detections): 

        
        for classID, className in zip([0, 1, 2], ['Car', 'Bus', 'Truck']):

            vehicle_detections = [detections[frame][classID] for frame in sorted(detections.keys())]
            frameBox, trackBox = self.per_class_tracker(vehicle_detections)

            # Save frameBox and trackBox as a pickle file and return 
            if len(trackBox.keys()) !=0:
                trackName = "Track_" + className + "_" + self.cam_ident + '.pkl'
                frameName = "Frame_" + className + "_" + self.cam_ident + '.pkl'
                trackName = os.path.join(self.out_dir, trackName)
                frameName = os.path.join(self.out_dir, frameName)
                pickle.dump(trackBox, open(trackName, "wb"))
                pickle.dump(frameBox, open(frameName, "wb"))
            
            



if __name__=='__main__':

    filepath = 'src/vc_outputs/aicity/detection_output/cam_1/cam_1.pkl'
    #filepath2 = 'cam_10.pkl'
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    tr = Tracker()
    tr.run_tracker(data)

    