import os
import app_logger
import configparser
import cv2
import pickle
from tqdm import tqdm

logger = app_logger.get_logger('tracker')

config = configparser.ConfigParser()
config.read('config/basic.ini')

'''
    1. Need images directory
    2. Camera FPS
    3. Tracker files

'''
##
# Workflow for the tracker object
# @returns 
#
class VisualizeTracker():

    def __init__(self, cam_ident):
        self.default = config['DEFAULT']
        self.config = config['TRACKING']
        self.cam_ident = cam_ident
        self.out_dir = os.path.join('src', 'vc_outputs', 'tracker_output', self.cam_ident)
        
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_name = os.path.join(self.out_dir, self.cam_ident + '.avi')
        frame_dim = (self.default['frame_width'], self.default['frame_height'])
        self.out_video = cv2.VideoWriter(video_name, self.fourcc, 10, (1920, 1080))
        
        print()


    def load_files(self):

        framepkl = {}
        trackpkl = {}

        for vehicleType in ['Car', 'Bus', 'Truck']:

            frameName = "Frame_" + vehicleType + "_" + self.cam_ident + ".pkl"
            trackName = "Track_" + vehicleType + "_" + self.cam_ident + ".pkl"
            if os.path.exists(os.path.join(self.out_dir, frameName)):
                framepkl[vehicleType] = pickle.load(open(os.path.join(self.out_dir, frameName), 'rb'))
                trackpkl[vehicleType] = pickle.load(open(os.path.join(self.out_dir, trackName), 'rb'))

        return framepkl, trackpkl

    
    def write_video_frame(self, img, framepkl, trackpkl, frameNumber):


        for vehicleType in framepkl.keys():

            if frameNumber in framepkl[vehicleType].keys():
                
                for detection in framepkl[vehicleType][frameNumber]:

                    ID, boundingBox = detection

                    if len(trackpkl[vehicleType][ID]) > int(self.config['TRACKLENGTH']):

                        X1, Y1, X2, Y2 = boundingBox
                        X1 = int(X1)
                        Y1 = int(Y1)
                        X2 = int(X2)
                        Y2 = int(Y2)
                        X = int(X1 + 0.5 * (X2 - X1))
                        Y = int(Y1 + 0.5 * (Y2 - Y1))

                        cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
                        cv2.putText(img, vehicleType + ": " + str(ID), (X, Y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))

        self.out_video.write(img)

    def run_visualizations(self):

        framepkl, trackpkl = self.load_files()

        images = os.listdir(os.path.join(self.default['data_dir'], self.cam_ident))

        for img in tqdm(sorted(images)):

            frameNum = int(img.replace(".jpg", ""))
            img = cv2.imread(os.path.join(self.default['data_dir'], self.cam_ident, img))
            self.write_video_frame(img, framepkl, trackpkl, frameNum)

        self.out_video.release()

if __name__=='__main__':
    
    vt = VisualizeTracker('cam_10')
    vt.run_visualizations()