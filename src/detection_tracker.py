import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pickle
import numpy as np
import cv2
import configparser
import torch
import time
import app_logger
import subprocess
from helper import Helper
from datetime import datetime

from detect_detectron2 import DetectDetectron
from sort_tracker import SortTracker
from tracker_deepsort import DeepsortTracker
from tracker_mot import MOTTracker
from bezier_online import BezierOnline

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali import fn
import argparse
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.types import DALIImageType

from tqdm import tqdm

logger = app_logger.get_logger('detectron')

basic_config = configparser.ConfigParser()
basic_config.read('config/basic.ini')

config = configparser.ConfigParser()
config.read(sys.argv[1])


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



class DetectionTracker:


    ##
    # Initializes the DetectionTracker object
    # 
    #
    def __init__(self):

        self.basic = basic_config['DEFAULT']
        self.default = config['DEFAULT']
        self.config = config['DETECTION']

        self.time = time.time()

        self.detector = DetectDetectron()
        self.tracker = MOTTracker()
        #self.tracker = DeepsortTracker()
        #self.tracker = SortTracker()
        self.bs = int(self.basic['batch_size'])
        self.counter = BezierOnline()
        self.video = os.path.join(self.detector.basic['data_dir'], 
            self.detector.default['cam_name']) + '.mp4'
    

    def get_batch(self, cap):

        count = 0
        inputs = []
        while count < self.bs:

            ret, frame = cap.read() #Read the frame
            if ret == False:
                break
            
            count +=1
            inp = self.detector.get_model_input(frame)
            inputs.append(inp)

        return inputs, count


    def dali_batch(self, batch):

        dims = batch.shape     
        inputs = []   

        for i in range(0, dims[0]):

            img = batch[i]
            img_pred = img[[2, 1, 0], :]
            height = int(self.default['height'])    
            width = int(self.default['width'])


            inp = {'image': img, 'height':height, 'width': width}

            inputs.append(inp)
        
        return inputs



    def workflow_dali(self):

        pipe = VideoPipe(batch_size=1, num_threads=1, device_id=0, data=self.video, sequence_length=16, shuffle=False)
        pipe.build()    
        dali_iter = DALIGenericIterator(pipe, ['data'], pipe.epoch_size("Reader"), fill_last_batch=False)
        frame_num = 1
        count = 0
        
        detection_dict = {}
        rem = 0
        start_process_time = time.time()
        
        for i, data in tqdm(enumerate(dali_iter)):
            batch = data[0]['data'][0, :, :, :, :]
            img_batch = self.dali_batch(batch)

            with torch.no_grad():
                #pred = self.detector.model([inputs])
                assert len(img_batch) != 0
                pred, features = self.detector.inference(img_batch)

            count = len(img_batch)
            len_feat = 0
            for i in range(0, count):
                len_feat_i = len(pred[i]['instances'])
                features_i = features[len_feat:(len_feat + len_feat_i), :]
                
                assert len(pred[i]['instances']) == features_i.shape[0]
                dets, all_dets = self.detector.mask_outputs(pred[i], features_i)
                detection_dict[frame_num + i] = dets
            
                ######################################TrackerPortion
                self.tracker.update_trackers(all_dets, (frame_num +i))


            frame_num += count
            if frame_num // 200 > rem:
                print('Frame Number {}'.format(frame_num))
                self.tracker.write_outputs()
                self.counter.workflow()
                self.tracker.flush()
                rem = frame_num // 200
                
                outfile = os.path.join(self.detector.out_dir, 
                    self.detector.cam_ident + '.pkl' )
                with open(outfile, 'wb') as handle:
                    pickle.dump(detection_dict, handle)
                    
                
            
        end_process_time = time.time()
        elapsed = end_process_time - start_process_time

        print('Elapsed {}: {} seconds'.format(self.detector.default['cam_name'], elapsed))
        print('Num of Frames: {}'.format(frame_num))
        
        outfile = os.path.join(self.detector.out_dir, 
            self.detector.cam_ident + '.pkl' )
        with open(outfile, 'wb') as handle:
            pickle.dump(detection_dict, handle)

        self.tracker.write_outputs()
        self.counter.workflow()
        self.counter.track1txt.close()


     
    def workflow(self):
        print('{}'.format(datetime.now()))
        cap = cv2.VideoCapture(self.video)
        detection_dict = {}
        rem = 0
        frame_num = 1
        print('Starting Detections')


        start_process_time = time.time()

        while (cap.isOpened()):
            

            #####################################Detection portion
            inputs, count = self.get_batch(cap)
            if count == 0:
                break
                
            with torch.no_grad():
                #pred = self.detector.model([inputs])
                assert len(inputs) != 0
                pred, features = self.detector.inference(inputs)

            len_feat = 0 
            for i in range(0, count):
                len_feat_i = len(pred[i]['instances'])
                features_i = features[len_feat:(len_feat + len_feat_i), :]
                
                assert len(pred[i]['instances']) == features_i.shape[0]
                dets, all_dets = self.detector.mask_outputs(pred[i], features_i)
                detection_dict[frame_num + i] = dets
            
                ######################################TrackerPortion
                self.tracker.update_trackers(all_dets, frame_num)
            
            frame_num += count
            if frame_num // 100 > rem:
                print('Frame Number {}'.format(frame_num))
                self.tracker.write_outputs()
                self.counter.workflow()
                self.tracker.flush()
                rem = frame_num // 100
                
                outfile = os.path.join(self.detector.out_dir, 
                    self.detector.cam_ident + '.pkl' )
                with open(outfile, 'wb') as handle:
                    pickle.dump(detection_dict, handle)
                
            
        
        end_process_time = time.time()
        elapsed = end_process_time - start_process_time

        print('Elapsed {}: {} seconds'.format(self.detector.default['cam_name'], elapsed))
        print('Num of Frames: {}'.format(frame_num))
        
        outfile = os.path.join(self.detector.out_dir, 
            self.detector.cam_ident + '.pkl' )
        with open(outfile, 'wb') as handle:
            pickle.dump(detection_dict, handle)

        self.tracker.write_outputs()
        self.counter.workflow()
        #self.counter.track1txt.close()
        self.counter.percam_txt.close()
        '''
        pid = subprocess.Popen([sys.executable, "bezier_online.py config/cam_13.ini"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            stdin=subprocess.PIPE)
        '''
        print('Done')

if __name__ == '__main__':

    #DetectionTracker().workflow()
    DetectionTracker().workflow_dali()

    print('Hello World')