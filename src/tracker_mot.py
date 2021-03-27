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

        self.trackers = {
            'Car' : JDETracker(max_time_lost, feat_thresh, 1 - min_iou, 1 - min_iou/2),
            'Bus' : JDETracker(max_time_lost, feat_thresh, 1 - min_iou, 1 - min_iou/2),
            'Truck': JDETracker(max_time_lost, feat_thresh, 1 - min_iou, 1 - min_iou/2)
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

        for trk in tracker_output:
            
            ID = trk.track_id
            X1, Y1, X2, Y2 = trk.tlbr
            ID = int(ID)
            if ID not in trackBox: #If new car, create a tracker entry
                trackBox[ID] = []
            

            #Add tracker results to dictionaryies
            frameBox[frameCount].append((ID, (X1, Y1, X2, Y2)))
            trackBox[ID].append((frameCount, (X1, Y1, X2, Y2)))

    
    ##
    # Creates the correct format for ingestion by the MOT Tracker
    # @param detections Output detection module, with feature numpy array
    # 
    #
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
    


    ##
    # Updates the per-class MOT Trackers
    # @param dets Concatenated (for all classes) detections from detection module
    # @param frame_num The frame number corresponding to this update
    #
    def update_trackers(self, dets, frame_num):

        x = 5

        for cname  in ['Car', 'Bus', 'Truck']:

            cdets = self.tracker_mot_format(dets[cname])
            class_output = self.trackers[cname].update(cdets)
            
            cls_tracklets = list(filter(lambda x: x.track_id in 
                self.trackers[cname].tracked_ids, self.trackers[cname].lost_stracks))
            '''
            if frame_num == 100:
                import pdb; pdb.set_trace()
            '''
            cls_removed_ids = [trk.track_id for trk in cls_tracklets]
            self.tracklets[cname].update(cls_removed_ids)

            self.process_tracker_output(class_output, self.frameBox[cname], 
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
        
        dets = {'Car': car_dets, 'Bus': bus_dets, 'Truck': truck_dets}
        frame_list = sorted(detections.keys())
        
        
        for frame_num in tqdm(frame_list):

            frame_dets = {'Car': car_dets[frame_num - 1], 'Bus': bus_dets[frame_num - 1], 
                'Truck': truck_dets[frame_num - 1]}
            self.update_trackers(frame_dets, frame_num)
            
            
            '''
            for cname in ['Car', 'Bus', 'Truck']:

                cdets_frame = self.tracker_mot_format(dets[cname][frame_num - 1])
                class_output = self.trackers[cname].update(cdets_frame)
            
                cls_tracklets = list(filter(lambda x: x.track_id in 
                    self.trackers[cname].tracked_ids, self.trackers[cname].lost_stracks))
                
                
                if frame_num == 500:
                    import pdb; pdb.set_trace()
                

                cls_removed_ids = [trk.track_id for trk in cls_tracklets]
                self.tracklets[cname].update(cls_removed_ids)

                self.process_tracker_output(class_output, self.frameBox[cname], 
                    self.trackBox[cname], frame_num - 1)
            '''
        
        self.write_outputs()
    


if __name__ == '__main__':

    MOTTracker().run_tracker()