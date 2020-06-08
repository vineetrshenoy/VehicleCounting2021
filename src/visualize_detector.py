import sys
import os
import app_logger
import configparser
import cv2
import pickle
from tqdm import tqdm

logger = app_logger.get_logger('detector_visualization')

config = configparser.ConfigParser()
config.read(sys.argv[1])

'''
    1. Load detection file
    2. load each image one by one
    3. write detections
'''


##
# Workflow for the tracker object
# @returns 
#
class VisualizeDetector():

    def __init__(self, cam_ident, fps, size_tup):
        self.default = config['DEFAULT']
        self.config = config['DETECTION']
        self.cam_ident = cam_ident
        self.out_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'detection_output', self.cam_ident)
        
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_name = os.path.join(self.out_dir, self.cam_ident + '.avi')
        frame_dim = (self.default['frame_width'], self.default['frame_height'])
        self.out_video = cv2.VideoWriter(video_name, self.fourcc, fps, size_tup) #TODO: CAN NOT HARDCODE
        


    ##
    # Draws the bounding boxes on the image frame and adds to video
    # @param img The cv2-loaded image
    # @param framepkl Dictionary indexed by frame number with tracking results
    # @param trackpkl Dictionary indexed by objectID with tracking results
    # @param frameNumber The frame count to put in correct order
    #
    def write_video_frame(self, img, detections, file_name):
        
        for vehicle in detections:

            for bbox in vehicle:

                top_left = (int(bbox[0]), int(bbox[1]))
                bottom_right = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
                cv2.imwrite(file_name, img)


        self.out_video.write(img)
        return img

    ##
    # Workflow for the visualizations. Writes per-image tracking results
    # @returns framepkl, trackpkl
    #
    def run_visualizations(self):

        with open(os.path.join(self.out_dir, self.cam_ident + '.pkl'), 'rb') as f:
            detections = pickle.load(f)

        images = os.listdir(os.path.join(self.default['data_dir'], self.cam_ident)) #gets the images
        images = sorted(images)
        for i in tqdm(range(0, len(images), int(self.config['step']))):
            imgName = images[i]
            frameNum = int(imgName.replace(".jpg", ""))
            img = cv2.imread(os.path.join(self.default['data_dir'], self.cam_ident, imgName)) #read the images
            img = self.write_video_frame(img, detections[frameNum], os.path.join(self.out_dir, imgName)) #write the images
            

        self.out_video.release() #release the video

if __name__=='__main__':
    
    VisualizeDetector('cam_9', 10, (1920, 1080)).run_visualizations()