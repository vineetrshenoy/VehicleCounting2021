import sys
import os
import app_logger
import configparser
import cv2
import pickle
from tqdm import tqdm

logger = app_logger.get_logger('tracker_visualization')

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')

##
# Workflow for the Visualizing Tracking
# 
#
class VisualizeTracker():

    def __init__(self):
        self.basic = basic_config['DEFAULT']
        self.default = config['DEFAULT']
        self.config = config['TRACKING']
        self.cam_ident = self.default['cam_name']
        self.out_dir = os.path.join(self.default['output_dir'], self.basic['job_name'], 'tracker_output', self.cam_ident)
        
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_name = os.path.join(self.out_dir, self.cam_ident + '.avi')
        frame_dim = (int(self.default['width']), int(self.default['height']))
        self.out_video = cv2.VideoWriter(video_name, self.fourcc, int(self.default['fps']), frame_dim) 
        
        

    ##
    # Loads the tracker files needed for visualization
    # @returns framepkl, trackpkl
    #
    def load_files(self):

        framepkl = {}
        trackpkl = {}

        for vehicleType in ['Car', 'Bus', 'Truck']: #Usually Track_* and Frame_* for each vehicle

            frameName = "Frame_" + vehicleType + "_" + self.cam_ident + ".pkl"
            trackName = "Track_" + vehicleType + "_" + self.cam_ident + ".pkl"
            if os.path.exists(os.path.join(self.out_dir, frameName)): #If the files exist, load them
                framepkl[vehicleType] = pickle.load(open(os.path.join(self.out_dir, frameName), 'rb'))
                trackpkl[vehicleType] = pickle.load(open(os.path.join(self.out_dir, trackName), 'rb'))

        return framepkl, trackpkl

    ##
    # Draws the bounding boxes on the image frame and adds to video
    # @param img The cv2-loaded image
    # @param framepkl Dictionary indexed by frame number with tracking results
    # @param trackpkl Dictionary indexed by objectID with tracking results
    # @param frameNumber The frame count to put in correct order
    #
    def write_video_frame(self, img, framepkl, trackpkl, frameNumber):
        #TODO: Write this code more elegantly

        for vehicleType in framepkl.keys(): #Car, Truck, or Bus

            #May not be a detection for that class in every frame
            if frameNumber in framepkl[vehicleType].keys():

                for detection in framepkl[vehicleType][frameNumber]: #Get the trackingID + detection

                    ID, boundingBox = detection

                    if len(trackpkl[vehicleType][ID]) > int(self.config['TRACKLENGTH']):

                        X1, Y1, X2, Y2 = boundingBox
                        X1 = int(X1)
                        Y1 = int(Y1)
                        X2 = int(X2)
                        Y2 = int(Y2)
                        X = int(X1 + 0.5 * (X2 - X1))
                        Y = int(Y1 + 0.5 * (Y2 - Y1))

                        cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 0, 255), 2) #write bounding box onto video
                        cv2.putText(img, vehicleType + ": " + str(ID), (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        
        self.out_video.write(img)
        return img

    ##
    # Workflow for the visualizations. Writes per-image tracking results
    # @returns framepkl, trackpkl
    #
    def run_visualizations(self):

        framepkl, trackpkl = self.load_files() #loads tracking files

        images = os.listdir(os.path.join(self.default['data_dir'], self.cam_ident)) #gets the images
        images = sorted(images)
        for i in tqdm(range(0, len(images), int(self.config['step']))):
            imgName = images[i]
            frameNum = int(imgName.replace(".jpg", ""))
            img = cv2.imread(os.path.join(self.default['data_dir'], self.cam_ident, imgName)) #read the images
            img = self.write_video_frame(img, framepkl, trackpkl, i) #write the images
            cv2.imwrite(os.path.join(self.out_dir, imgName), img)

        self.out_video.release() #release the video

if __name__=='__main__':
    
    VisualizeTracker().run_visualizations()