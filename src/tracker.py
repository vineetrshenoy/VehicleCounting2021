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
    # Initializes the Tracker object
    # 
    #
    def __init__(self, cam_ident):
        self.config = config['TRACKING']
        self.cam_ident = cam_ident

        os.makedirs(os.path.join('src', 'vc_outputs', 'tracker_output', self.cam_ident), exist_ok=True)
        self.out_dir = os.path.join('src', 'vc_outputs', 'tracker_output', self.cam_ident)
        print()


    ##
    # Performs manipulations on tracker output and prepares object for next 
    # step in Pipelline
    # 
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
    # Performs tracking on a per-class basis
    # @param vehicle dtcs List of lists, where nested list are detections
    # and each list corresponds to a frame (arranged in order)
    # @returns frameBox, trackBox Tracked objects
    #
    def per_class_tracker(self, vehicle_dtcs):
        
        objectTracker = Sort()
        frameBox = {}
        trackBox = {}

        for frameCount in tqdm(range(0, len(vehicle_dtcs))):

            frame = vehicle_dtcs[frameCount]
            THRESHOLD = float(self.config['THRESHOLD'])
            NMS = list(filter(lambda detect: detect[4] > THRESHOLD, frame))
            NMS = np.array(NMS)
            
            tracker_output = objectTracker.update(NMS)
            self.process_tracker_output(tracker_output, frameBox, trackBox, frameCount + 1)

        objectTracker.reset()
        return frameBox, trackBox




    ##
    # Workflow for the tracker object
    # @returns 
    #
    def run_tracker(self, detections): 

        
        for classID, className in zip([0, 1, 2], ['Car', 'Bus', 'Truck']):

            vehicle_detections = [detections[frame][classID] for frame in sorted(detections.keys())]
            frameBox, trackBox = self.per_class_tracker(vehicle_detections)

            # Save frameBox and trackBox as a pickle file and return 
            #TODO: Fix Paths
            if len(trackBox.keys()) !=0:
                trackName = "Track_" + className + "_" + self.cam_ident + '.pkl'
                frameName = "Frame_" + className + "_" + self.cam_ident + '.pkl'
                trackName = os.path.join(self.out_dir, trackName)
                frameName = os.path.join(self.out_dir, frameName)
                pickle.dump(trackBox, open(trackName, "wb"))
                pickle.dump(frameBox, open(frameName, "wb"))
            
            #return frameBox, trackBox



if __name__=='__main__':


    with open('/vulcan/scratch/vshenoy/aicity2020/other_detection_files/cam_10.pkl', 'rb') as f:
        data = pickle.load(f)

    tr = Tracker('cam_10')
    tr.run_tracker(data)

    