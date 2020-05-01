import os
from sort import Sort
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import app_logger
import configparser

logger = app_logger.get_logger('tracker')

config = configparser.ConfigParser()
config.read('config/basic.ini')


class Tracker():

    ##
    # Workflow for the tracker object
    # @returns 
    #
    def __init__(self):
        self.config = config['TRACKING']
        print()


    ##
    # Workflow for the tracker object
    # @returns 
    #
    def process_tracker_output(self, tracker_output, frameBox, trackBox, frameCount):
        
        
        for bbox in tracker_output:

            X1, Y1, X2, Y2, ID = bbox
            ID = int(ID)
            if ID not in trackBox:
                trackBox[ID] = []
            if frameCount not in frameBox:
                frameBox[frameCount] = []

            frameBox[frameCount].append((ID, (X1, Y1, X2, Y2)))
            trackBox[ID].append((frameCount, (X1, Y1, X2, Y2)))
            
    
    ##
    # Workflow for the tracker object
    # @returns 
    #
    def per_vehicle_tracker(self, vehicle_dtcs):
        
        objectTracker = Sort()
        frameBox = {}
        trackBox = {}

        for frameCount in tqdm(range(0, len(vehicle_dtcs))):

            frame = vehicle_dtcs[frameCount]
            THRESHOLD = float(self.config['THRESHOLD'])
            NMS = list(filter(lambda detect: detect[4] > THRESHOLD, frame))
            NMS = np.array(NMS)
            
            tracker_output = objectTracker.update(NMS)
            self.process_tracker_output(tracker_output, frameBox, trackBox, frameCount)


        return frameBox, trackBox




    ##
    # Workflow for the tracker object
    # @returns 
    #
    def run_tracker(self, detections): 

        
        for classID, className in zip([0, 1, 2], ['Car', 'Bus', 'Truck']):

            vehicle_detections = [detections[frame][classID] for frame in sorted(detections.keys())]
            frameBox, trackBox = self.per_vehicle_tracker(vehicle_detections)

            # Save frameBox and trackBox as a pickle file and return 
            '''
            if len(trackBox.keys()) !=0:
                basename = os.path.basename('/vulcan/scratch/vshenoy/aicity2020/other_detection_files/cam_5_rain.pkl').split('.')[0]
                pickle.dump(trackBox, open("/vulcan/scratch/vshenoy/aicity2020/tracker_output/" + "Track_" + className + "_" + basename, "wb"))
                pickle.dump(frameBox, open("/vulcan/scratch/vshenoy/aicity2020/tracker_output/" + "Frame_" + className + "_" + basename, "wb"))
            '''
            return frameBox, trackBox



if __name__=='__main__':


    with open('/vulcan/scratch/vshenoy/aicity2020/other_detection_files/cam_5_rain.pkl', 'rb') as f:
        data = pickle.load(f)

    tr = Tracker()
    tr.run_tracker(data)

    