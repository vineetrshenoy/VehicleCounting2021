import sys
import os
import pickle
import numpy as np
import cv2
import configparser
import torch
torch.cuda.current_device()
import app_logger

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import BoxMode
from detectron2.layers import nms

from tqdm import tqdm
import matplotlib.path as mplPath

logger = app_logger.get_logger('detect_detectron')

config = configparser.ConfigParser()
config.read('config/basic.ini')



cam_1_path = mplPath.Path(np.array([[1, 150], [844, 96], [1277, 277], [1277, 750], [2, 683]]))
cam_2_path = mplPath.Path(np.array([[1, 26], [751, 24], [1277, 149], [1278, 718], [2, 717]]))
cam_3_path = mplPath.Path(np.array([[1, 191], [1279, 86], [1278, 623], [1, 629]]))
cam_4_path = mplPath.Path(np.array([[1, 198], [342, 116], [750, 81], [1279, 264], [1278, 956], [0, 956]]))
cam_5_path = mplPath.Path(np.array([[1, 277], [450, 99], [599, 96], [1172, 280], [960, 955], [0, 956]]))
cam_6_path = mplPath.Path(np.array([[1, 265], [496, 127], [793, 162], [1278, 453], [1279, 957], [0, 956]]))
cam_7_path = mplPath.Path(np.array([[2, 415], [421, 386], [1277, 468], [1279, 716], [2, 596]]))
cam_8_path = mplPath.Path(np.array([[3, 300], [1031, 2], [1567, 177], [1913, 576], [1917, 1074], [1, 1076]]))
cam_9_path = mplPath.Path(np.array([[4, 270], [1036, 132], [1916, 280], [1918, 1076], [0,1074]]))
cam_10_path = mplPath.Path(np.array([[2, 228], [1916, 300], [1916, 1076], [2, 1074]]))
cam_11_path = mplPath.Path(np.array([[2,296], [1916, 306], [1918, 1076], [0, 1072]]))
cam_12_path = mplPath.Path(np.array([[4, 172], [1522, 198], [1916, 312], [1916, 1076], [2, 1074]]))
cam_13_path = mplPath.Path(np.array([[2, 126], [1916, 88], [1918, 1076], [2, 1074]]))
cam_14_path = mplPath.Path(np.array([[2, 870], [2554, 1108], [2556, 1912], [2, 1912]]))
cam_15_path = mplPath.Path(np.array([[4, 282], [858, 200], [1918, 360], [1916, 1074], [2, 1074]]))
cam_16_path = mplPath.Path(np.array([[2, 198], [1918, 216], [1916, 1074], [2, 1072]]))
cam_17_path = mplPath.Path(np.array([[0, 224], [1916, 222], [1916, 1076], [2, 1972]]))
cam_18_path = mplPath.Path(np.array([[4, 276], [1920, 280], [1918, 1078], [2, 1072]]))
cam_19_path = mplPath.Path(np.array([[2, 364], [1914, 374], [1916, 1074], [4, 1072]]))
cam_20_path = mplPath.Path(np.array([[4, 512], [1918, 454], [1916, 1076], [2, 1074]]))

paths = [None, cam_1_path, cam_2_path, cam_3_path, cam_4_path, cam_5_path, cam_6_path,
cam_7_path, cam_8_path, cam_9_path, cam_10_path, cam_11_path, cam_12_path, cam_13_path,
cam_14_path, cam_15_path, cam_16_path, cam_17_path, cam_18_path, cam_19_path, cam_20_path]


paths = {
	'cam_1': cam_1_path,
	'cam_1_dawn': cam_1_path,
	'cam_1_rain': cam_1_path,
	'cam_2': cam_2_path,
	'cam_2_rain': cam_2_path,
	'cam_3': cam_3_path,
	'cam_3_rain': cam_3_path,
	'cam_4': cam_4_path,
	'cam_4_dawn': cam_4_path,
	'cam_4_rain': cam_4_path,
	'cam_5': cam_5_path,
	'cam_5_dawn': cam_5_path,
	'cam_5_rain': cam_5_path,
	'cam_6': cam_6_path,
	'cam_6_snow': cam_6_path,
	'cam_7': cam_7_path,
	'cam_7_dawn': cam_7_path,
	'cam_7_rain': cam_7_path,
	'cam_8': cam_8_path,
	'cam_9': cam_9_path,
	'cam_10': cam_10_path,
	'cam_11': cam_11_path,
	'cam_12': cam_12_path,
	'cam_13': cam_13_path,
	'cam_14': cam_14_path,
	'cam_15': cam_15_path,
	'cam_16': cam_16_path,
	'cam_17': cam_17_path,
	'cam_18': cam_18_path,
	'cam_19': cam_19_path,
	'cam_20': cam_20_path
}

##
#   DetectDetectron class for performing detections using detectron2
#
class DetectDetectron:

    def __init__(self, cam_ident, fps, size_tup):
        self.paths = paths
        self.default = config['DEFAULT']
        self.config = config['DETECTION']
        predictor, cfg = self.load_model()
        self.predictor = predictor
        self.cfg = cfg
        self.cam_ident = cam_ident
        self.out_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'detection_output', self.cam_ident)
        
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_name = os.path.join(self.out_dir, self.cam_ident + '.avi')
        #frame_dim = (self.default['frame_width'], self.default['frame_height'])
        self.out_video = cv2.VideoWriter(video_name, self.fourcc, fps, size_tup) #TODO: CAN NOT HARDCODE
        os.makedirs(self.out_dir, exist_ok=True)
    ##
    # Loads a model for inference
    # @returns DefaultPredictor, cfg object
    #
    def load_model(self):
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config['config_file']))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config['config_file'])
        cfg.MODEL.DEVICE = 'cuda:0'
        predictor = DefaultPredictor(cfg)

        return predictor, cfg

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
    #   Creates object structure for subsequent tracking
    #   @param output 
    #
    def process_outputs(self, raw_detections: dict) -> list:

        boxes = raw_detections['instances'].get_fields()['pred_boxes']	
        scores = raw_detections['instances'].get_fields()['scores']
        classes = raw_detections['instances'].get_fields()['pred_classes'] 
        
        car_indices = np.where(classes.cpu() == 2)
        bus_indices = np.where(classes.cpu() == 5)
        truck_indices = np.where(classes.cpu() == 7)

        indices = np.append(car_indices[0], [bus_indices[0]])
        indices = np.append(indices, truck_indices[0])	#indices of detections only including car, bus, truck
        boxes = boxes[indices] 
        scores = scores[indices]
        classes = classes[indices]
        
        N = len(boxes)
        #Initialize detection structure
        car_detections = []
        bus_detections = []
        truck_detections = []
                
        for i in range(0, N): #for each box

            #box = BoxMode.convert(boxes[i].tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)[0]
            centers = boxes[i].get_centers()[0]
                        
            box = boxes[i].tensor[0]
            x1 = box[0].item()
            y1 = box[1].item()
            x2 = box[2].item()
            y2 = box[3].item()
            
            tup = (x1, y1, x2, y2, scores[i].item())
            
            cat = classes[i].item() #category
            bbPath = self.paths[self.cam_ident] #gets the ROI coordinates

            #if contains point, add to list of detections
            if bbPath.contains_point((centers[0].item(), centers[1].item())):
                
                if scores[i].item() > float(self.config['score_thresh']):

                    if cat == 2:
                        car_detections.append(tup)
                    elif cat == 5:
                        bus_detections.append(tup)
                    else:
                        truck_detections.append(tup)

        car_detections = self.perform_nms(car_detections)
        bus_detections = self.perform_nms(bus_detections)
        truck_detections = self.perform_nms(truck_detections)
        detections = [car_detections, bus_detections, truck_detections]
        return detections

    ##
    #   The detections for a certain class
    #   @param detections
    #   @returns 
    #
    def perform_nms(self, detections):

        N = len(detections)
        scores = np.array([box[4] for box in detections])
        boxes = np.zeros((N, 4))
        for i in range(0, N):
            boxes[i, 0] = detections[i][0]
            boxes[i, 1] = detections[i][1]
            boxes[i, 2] = detections[i][2]
            boxes[i, 3] = detections[i][3]
        
        indices = nms(torch.from_numpy(boxes), torch.from_numpy(scores), float(self.config['iou']))
        indices = indices.numpy().tolist()
        return [detections[i] for i in indices]
    ##
    #   Runs the detection workflow
    #   @param img Input image
    #   @param detections Post-processed detections
    #   @returns 
    #
    def visualize_detections(self, img, detections, file_name):
        
        for vehicle in detections:

            for bbox in vehicle:

                top_left = (int(bbox[0]), int(bbox[1]))
                bottom_right = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
                cv2.imwrite(file_name, img)

        self.out_video.write(img)
        return img

    ##
    #   Runs the detection workflow
    #   @param filepath Folder containing extracted images 
    #
    def run_predictions(self):

        detection_dict = {}
        files = sorted(os.listdir(os.path.join(self.default['data_dir'], self.cam_ident)))

        for i in tqdm(range(0, len(files), int(self.config['step']))): #for every camera frame

            img = cv2.imread(os.path.join(self.default['data_dir'], self.cam_ident, files[i])) #read the frame

            outputs = self.predictor(img) #generate detections on image
            detections = self.process_outputs(outputs)
            frame = self.get_frame_number(os.path.join(self.default['data_dir'], self.cam_ident, files[i]))
            
            if int(self.config['visualize']) == 1:

                file_name = os.path.join(self.out_dir, files[i])
                self.visualize_detections(img, detections, file_name)
                
            detection_dict[frame] = detections
        
        if (int(self.config['visualize'])) == 1:
            self.out_video.release() #release the video

        with open(os.path.join(self.out_dir, self.cam_ident + '.pkl' ), 'wb') as handle:
            pickle.dump(detection_dict, handle)           

        return detection_dict

if __name__ == '__main__':

    #filepath = sys.argv[1]
    dt = DetectDetectron('cam_9', 10, (1920, 1080))
    dt.run_predictions()
    print('Hello World')