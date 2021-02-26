import os
import sys
import numpy as np
import time
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali import fn
import argparse
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.types import DALIImageType
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from PIL import Image
import cv2
import numpy as np
import torch
import configparser
import pickle
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
'''
logger = app_logger.get_logger('detect_detectron')
'''
config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')


##
#   DetectDetectron class for performing detections using detectron2
#
class DetectDali:

    def __init__(self):
        self.basic = basic_config['DEFAULT']
        self.default = config['DEFAULT']
        self.config = config['DETECTION']
        

        self.load_model()

        self.cam_ident = self.default['cam_name']
        self.out_dir = os.path.join(
            self.basic['output_dir'],
            self.basic['job_name'],
            'detection_output',
            self.cam_ident)
        
        os.makedirs(self.out_dir, exist_ok=True)
        #self.roi = Helper.get_roi(self.default['roi'])


    ##
    # Loads a model for inference
    # @returns DefaultPredictor, cfg object
    #

    def load_model(self):

        self.cfg = get_cfg()
        model_file = self.basic['model_file']
        self.cfg.merge_from_file(model_zoo.get_config_file(model_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        #self.cfg.MODEL.DEVICE='cpu'
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )

    
    ##
    #   Creates object structure for subsequent tracking
    #   @param raw_detections The detections for a single frame
    #   @returns detections The processed detections
    #
    def process_outputs(self, raw_detections: dict) -> list:

        boxes = raw_detections['instances'].get_fields()['pred_boxes']
        scores = raw_detections['instances'].get_fields()['scores']
        classes = raw_detections['instances'].get_fields()['pred_classes']

        # get detections only for these classes
        car_indices = np.where(classes.cpu() == 2)
        bus_indices = np.where(classes.cpu() == 5)
        truck_indices = np.where(classes.cpu() == 7)

        indices = np.append(car_indices[0], [bus_indices[0]])
        # indices of detections only including car, bus, truck
        indices = np.append(indices, truck_indices[0])
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]

        N = len(boxes)
        # Initialize detection structure
        car_detections = []
        bus_detections = []
        truck_detections = []

        for i in range(0, N):  # for each box

            centers = boxes[i].get_centers()[0]

            box = boxes[i].tensor[0]
            x1 = box[0].item()
            y1 = box[1].item()
            x2 = box[2].item()
            y2 = box[3].item()
            cat = classes[i].item()  # category

            tup = (x1, y1, x2, y2, scores[i].item(), cat + 1)

            bbPath = mplPath.Path(self.roi)  # gets the ROI coordinates

            # if contains point, add to list of detections
            # if True:
            if bbPath.contains_point(
                    (centers[0].item(), centers[1].item())):  # if inside the region of interest

                if scores[i].item() > float(self.config['score_thresh']):

                    if cat == 2:
                        car_detections.append(tup)
                    elif cat == 5:
                        bus_detections.append(tup)
                    else:
                        truck_detections.append(tup)

        # Perform NMS per-class
        car_detections = self.perform_nms(car_detections)
        bus_detections = self.perform_nms(bus_detections)
        truck_detections = self.perform_nms(truck_detections)
        detections = car_detections + bus_detections + truck_detections
        return detections

    ##
    #   Performs non-maximal supression on the bounding boxes
    #   @param detections
    #   @returns detections The detections after performing non-maximal supression
    #
    def perform_nms(self, detections: list) -> list:

        N = len(detections)
        scores = np.array([box[4] for box in detections])
        boxes = np.zeros((N, 4))
        for i in range(0, N):
            boxes[i, 0] = detections[i][0]
            boxes[i, 1] = detections[i][1]
            boxes[i, 2] = detections[i][2]
            boxes[i, 3] = detections[i][3]

        indices = nms(
            torch.from_numpy(boxes),
            torch.from_numpy(scores),
            float(
                self.config['iou']))
        indices = indices.numpy().tolist()
        return [detections[i] for i in indices]

    
    ##
    #   Runs the detection workflow
    #
    #
    def run_predictions(self):

        detection_dict = {}
        feature_dict = {}
        files = sorted(
            os.listdir(
                os.path.join(
                    self.default['data_dir'],
                    self.cam_ident)))

        frame_times = np.zeros((len(files),))
        start_process_time = time.process_time()

        #dst = DeepsortTracker()
        for i in tqdm(range(0, len(files), int(
                self.config['step']))):  # for every camera frame

            img = cv2.imread(
                os.path.join(
                    self.default['data_dir'],
                    self.cam_ident,
                    files[i]))  # read the frame

            start_frame_time = time.process_time()
            outputs = self.predictor(img)  # generate detections on image
            end_frame_time = time.process_time()

            frame_times[i] = end_frame_time - start_frame_time

            detections = self.process_outputs(outputs)
            #features = None
            #features = self.feature_extractor.workflow(img, detections)

            frame = self.get_frame_number(
                os.path.join(
                    self.default['data_dir'],
                    self.cam_ident,
                    files[i]))

            if int(self.basic['visualize']) == 1:

                file_name = os.path.join(self.out_dir, files[i])
                self.visualize_detections(img, detections, file_name)

            detection_dict[frame] = detections
            #feature_dict[frame] = features

        end_process_time = time.process_time()

        logger.info(
            self.default['job_name'] +
            ' || ' +
            self.cam_ident +
            ' -----------------')
        logger.info('Mean: ' + str(np.mean(frame_times)))
        logger.info('Median: ' + str(np.median(frame_times)))
        logger.info('std: ' + str(np.std(frame_times)))
        logger.info('max: ' + str(np.max(frame_times)))
        logger.info('min: ' + str(np.min(frame_times)))
        logger.info('Total: ' + str(end_process_time - start_process_time))

        #np.save(self.default['job_name'] +'_' + self.cam_ident + '.npy', frame_times)

        if (int(self.config['visualize'])) == 1:
            self.out_video.release()  # release the video

        with open(os.path.join(self.out_dir, self.cam_ident + '.pkl'), 'wb') as handle:
            pickle.dump(detection_dict, handle)

        with open(os.path.join(self.out_dir, self.cam_ident + '_features.pkl'), 'wb') as handle:
            pickle.dump(feature_dict, handle)
        return detection_dict


if __name__ == '__main__':

    dd = DetectDali()
