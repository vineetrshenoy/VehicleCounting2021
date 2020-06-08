import os
import sys
import cv2
import numpy as np
import configparser
import glob
from tqdm import tqdm
from helper import Helper

config = configparser.ConfigParser()
config.read(sys.argv[1])

class VisualizeCounting():

    def __init__(self):
        self.default = config['DEFAULT']
        self.cam_ident = self.default['cam_name']
        self.out_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'counting_output') #set output directory
        self.roi = Helper.get_roi(self.default['roi'])
        self.track1txt = self.default['counting_file']
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_name = os.path.join(self.default['output_dir'], self.default['job_name'], 'counting_output', self.default['job_name'] + '.avi') 
        frame_dim = (int(self.default['width']), int(self.default['height']))
        self.out_video = cv2.VideoWriter(video_name, self.fourcc, int(self.default['fps']), frame_dim) 
        
    ##
    # Reads and sorts the counting file results
    # 
    # @returns DefaultPredictor, cfg object
    #
    def read_counting_file(self):

        with open(self.track1txt, 'rb') as f:
            data = f.readlines()

        data = list(map(lambda x: x.decode('utf-8').strip('\n'), data))
        data = list(map(lambda x: x.split(' '), data))
        data = list(map(lambda x: list(map(lambda y: int(y), x)), data))

        data = np.array(data)
        data = data[data[:,1].argsort()]
        
        return data

    ##
    #   Processes a filename
    #   @param filename filepath for the image
    #   @returns frame Frame number for certain file
    #
    def write_on_frame(self, img, mvts):

        N = mvts.shape[0]

        mvt_dict = {1: (100,300), 2: (1000, 1000), 3: (1650, 925)}
        cat_dict = {1: 'CAR', 2: 'TRUCK'}

        for j in range(0, N):

            mvt = mvts[j, :]
            text = '{}-{}-{}'.format(cat_dict[mvt[3]],str(mvt[2]), str(mvt[4]))
            loc = mvt_dict[mvt[2]]
            cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)

        return img

    ##
    #   Processes a filename
    #   @param filename filepath for the image
    #   @returns frame Frame number for certain file
    #
    def get_frame_number(self, filename: str) -> int:
	
        split_string = filename.split('/')
        frame = split_string[-1].split('.')[0]
        return int(frame)

    ##
    # Reads and sorts the counting file results
    # 
    # @returns DefaultPredictor, cfg object
    #
    def workflow(self):
        
        
        results = self.read_counting_file()
        imgLoc = 'src/vc_outputs/aicity/tracker_output'
        images = sorted(glob.glob(os.path.join(imgLoc, self.cam_ident, '*.jpg')))
        N = len(images)

        
        imgLoc = 'src/vc_outputs/aicity/tracker_output'
        for i in tqdm(range(0, N)):

            imageName = os.path.join(imgLoc, self.cam_ident, os.path.basename(images[i])) # image i+1.jpg
            img = cv2.imread(imageName)
            imgNum = i + 1
            cv2.polylines(img, np.int32([self.roi]), 1, (0, 255, 0), 1, cv2.LINE_AA)


            if np.sum(np.array(results[:, 1] == imgNum)) > 0:
                
                indices = np.array(results[:, 1] == imgNum)
                indices = np.where(indices)[0]

                img = self.write_on_frame(img, results[indices, :])

                #i = i + len(indices)
                
            
            outfile = os.path.join(self.out_dir, os.path.basename(imageName))
            cv2.imwrite(outfile, img)
            self.out_video.write(img)

        

if __name__=='__main__':

    vc = VisualizeCounting()
    vc.workflow()