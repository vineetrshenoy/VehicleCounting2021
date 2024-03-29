import os
import sys
import numpy as np
import time
import app_logger
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
import matplotlib.path as mplPath
from helper import Helper
from PIL import Image
import cv2
import torch
import configparser
import pickle
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.layers import nms

logger = app_logger.get_logger('detectron')

config = configparser.ConfigParser()
config.read(sys.argv[1])

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')

class VideoPipe(Pipeline):
    
    def __init__(self, batch_size, num_threads, device_id, data, shuffle, sequence_length=1):       
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=16)        
        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=sequence_length,
                                     shard_id=0, num_shards=1,
                                     random_shuffle=shuffle, initial_fill=10)
        self.colorspace = ops.ColorSpaceConversion(
            image_type=DALIImageType.RGB, 
            output_type=DALIImageType.BGR,
            device='gpu'
        )
            
        self.resize = ops.Resize(resize_shorter=800, max_size=1333, device="gpu")
        self.transpose = ops.Transpose(perm=[0, 3, 1, 2], device='gpu')

    
    def define_graph(self):
        
        output = self.input(name="Reader")
        #output = self.colorspace(output)
        output = self.resize(output)
        output = self.transpose(output)
        return output

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
        self.roi = Helper.get_roi(self.default['roi'])


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
    #  Creates the pipeline to load images
    #  @returns dali_iter An iterator for loading images
    #
    def build_dali_pipeline(self):


        video = os.path.join(self.basic['data_dir'], self.default['cam_name']) + '.mp4'
        bs = int(self.basic['batch_size'])
        sl = int(self.basic['sequence_length'])

        pipe = VideoPipe(batch_size=bs, num_threads=1, device_id=0, data=video, sequence_length=sl, shuffle=False)
        pipe.build()
        dali_iter = DALIGenericIterator(pipe, ['data'], pipe.epoch_size("Reader"), fill_last_batch=False)

        return dali_iter

    ##
    #   Creates the input dictionary for the model
    #   @param images The images from dali
    #
    def create_model_input(self, images):
        
        img_pred = images[:, [2, 1, 0], :]
        height = int(self.default['height'])
        width = int(self.default['width'])

        N = images.shape[0]
        input_list = []
        for i in range(0, N):
            inputs = {'image': img_pred[i, :], 'height':height, 'width': width}
            input_list.append(inputs)

        return input_list
    
    ##
    #   Runs the detection workflow
    #
    #
    def run_predictions(self):

        dali_iter = self.build_dali_pipeline()
        start_process_times = time.process_time()
        detection_dict = {}
        for i, data in tqdm(enumerate(dali_iter)):
            
            
            img = data[0]['data'][0, :, :, :, :]
            inputs = self.create_model_input(img)

            with torch.no_grad():
                outputs = self.model(inputs)

            N = len(outputs)
            sl = int(self.basic['sequence_length'])
            for j in range(0, N):
                
                detections = self.process_outputs(outputs[j])
                idx = i * sl + j + 1
                detection_dict[idx] = detections
            
        end_process_time = time.process_time()
        t = end_process_time - start_process_times
        logger.info('Detection time: {}'.format(t))
        
        with open(os.path.join(self.out_dir, self.cam_ident + '.pkl' ), 'wb') as handle:
            pickle.dump(detection_dict, handle)
        
        


if __name__ == '__main__':

    dd = DetectDali()
    dd.run_predictions()
    print('hello world')
