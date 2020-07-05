""" Uses EfficientDet to product inference pickle files for AI City Challenge Track 1 """
""" Install the automl efficientdet module from https://github.com/google/automl/tree/master/efficientdet, 
    1. set up model: create a tmp folder and download the model file 
    2. set up dataset: I created a symbolic link to the original dataset folder 
                    mkdir -p datasets/aicity
                    ln -s /fs/diva-scratch/nvidia_data_2020/NVIDIA_AIC_2020_Datasets/AIC20_track1 datasets/aicity/AIC20_track1
    3.                update "video_dataset_pattern" path in config file
    4. setup output folders:
                    update "output_dataset_pattern" path in config file

    Run the script using the following command:
    python detect_cars.py
"""

from PIL import Image, ImageDraw
import cv2
#from imutils.video import FPS

import os
import pickle
import glob 
import sys
from tqdm import tqdm

sys.path.append('src/automl/')
sys.path.append('src/automl/efficientdet')
sys.path.append('src/automl/efficientdet/object_detection')

import tensorflow as tf
import numpy as np
import copy


import tqdm
import configparser
import time

import automl.efficientdet.inference as inference
import hparams_config


# ROI detection imports
#from matplotlib.path import Path
#from matplotlib import pyplot as plt


#num_classes = hparams_config.get_detection_config(model_name).num_classes
#enable_ema = hparams_config.get_detection_config(model_name).enable_ema

#tf.saved_model.load(sess, ['serve'], ckpt_path)

# Dict to map COCO class for AI CITY Evaluation
class_map = {3:0, 6:0, 8:1}

class EfficientDet:

    def __init__(self, config, driver):
        self.default = config['DEFAULT']
        self.config = config['DETECTION']
        #self.video_name = self.config['video_name']
        self.frame_rate = self.default['fps']
        self.input_size = (int(self.config['width']),int(self.config['height'])) 
        self.batch_size = int(self.config['batch_size'])
        self.model_name = self.config['model_name']
        self.ckpt_path = self.config['ckpt_path']

        self.cam_ident = self.default['cam_name']
        self.out_dir = os.path.join(self.default['output_dir'], self.default['job_name'], 'detection_output', self.cam_ident)
        os.makedirs(self.out_dir, exist_ok=True)

        print(self.model_name)
        self.driver = driver
        #self.driver.build()

    """ process video_ids and video_names and store in list """
    def create_video_id_dict(self, video_id_list_path):
        video_id = {}
        f = open(video_id_list_path,'r')
        for  line in f.readlines():
            l = line.split()
            video_name = l[1].split('.')[0]
            video_id[video_name] = {}
            video_id[video_name]['id'] = int(l[0])
            #video_id[video_name]['total_frames'] = int(re.search(r"\(([A-Za-z0-9_]+)\)", l[2]).group(1))
        return video_id

    """ process ROI file """
    def get_roi(self, video_id):
        roi_list = []        
        
        with open(self.config['roi_path'], 'r') as roi_file:
            for point in roi_file.readlines():
                roi_list.append((int(point.split(',')[0]), int(point.split(',')[1])))
        
        return roi_list

    def create_roi_mask(self, roi_polygon, image_path):
        image = cv2.imread(image_path, -1)   
        image = cv2.resize(image, self.input_size, interpolation = cv2.INTER_AREA) 
        mask = np.zeros(image.shape, dtype=np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
        cv2.fillConvexPoly(mask, np.array(roi_polygon, dtype=np.int32), ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)

        # save the result
        #cv2.imwrite('image_masked.png', masked_image)
        return masked_image
    '''
    def run_predictions(self):
        dict_list = []
        if self.video_name == 'all':
            folder_list = [f for f in glob.glob(config['DETECTION']['video_dataset_pattern'] + '*')]
        else:
            folder_list = [config['DETECTION']['video_dataset_pattern'] + self.video_name]
        print(" Generating detections for: ", folder_list)

        for folder in folder_list:
            video_name = folder.split('/')[-1]
            fps = FPS().start()
            video_number = video_name.split('_')[1]
            print("Processing " + video_name)

            roi_list = self.get_roi(video_number)
            roi_polygon = np.array(roi_list)

            master_dict = {}

            image_detections = []
            
            # get list of all images
            image_list = glob.glob(folder + "/*.jpg")

            start_detection = time.time()

            # iterate over batch_size images and append to predictions
            for i in range(0, len(image_list), self.batch_size):

                imgs = []
                image_batch = image_list[i: i + self.batch_size]
                for image_path in image_batch:
                    imgs.append(self.create_roi_mask(roi_polygon, image_path))

                predictions = self.driver.serve_images(imgs)
                image_detections.append(predictions.tolist())
            done_detection = start_detection - time.time()

            print("done with all predictions in ", done_detection)

            start_processing = time.time()

            # iterate over all predictions to store in required format
            for index, prediction in enumerate(image_detections):
                image_path = image_list[index]
                frameCounter = int(image_path.split('/')[-1].split('.')[0])
                master_dict[frameCounter] = []
                
                # format of predictions: 
                # boxes = predictions[0][1:5] ([x, y, width, height])
                # classes = predictions[0][6] 
                # scores = predictions[0][5]

                pred = [item for sublist in prediction for item in sublist]
                for detection in pred:
                    class_id = copy.copy(detection[6])
                    bbox = copy.copy(detection[1:5])
                    score = copy.copy(detection[5])
                    bbox_list = []

                    # Keep only car, bus and truck detections, discard the rest
                    if class_id == 3 or class_id == 6 or class_id == 8:

                        # convert [x, y, width, height] to [xmin, ymin, xmax, ymax]
                        #bbox[2:4] += bbox[0:2]

                        class_id = class_map[class_id]

                        # convert [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
                        bbox_list = [bbox[1],bbox[0],bbox[3],bbox[2]]

                        bbox_list.extend([score, class_id]) 
                

                        #if class_id not in master_dict[frameCounter]:
                        #    master_dict[frameCounter][class_id] = []
                        #master_dict[frameCounter][class_id].append(tuple(bbox_list))
                        master_dict[frameCounter].append(tuple(bbox_list))

                #out_image = self.driver.visualize(np.array(img_frame), predictions[0])       
                #img = np.array(out_image)
                #cv2.imwrite("Detection_"+index, img)
                #k = cv2.waitKey(1)
                #if k == 27:
                #    break
                done_processing = start_processing - time.time()
                print("Done with processing in ", done_processing)

            fps.stop()
            print("[INFO] elapsed time: {:.2f} for video: {}".format(fps.elapsed(),video_name))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            pickle.dump(master_dict, open(config['DETECTION']['output_dataset_pattern'] + video_name + ".pkl", "wb"))
            dict_list.append(master_dict)
        return dict_list
    '''

    ##
    #   Displays the detections on the images
    #   @param img Input image
    #   @param detections Post-processed detections
    #   @param file_name The name of the image to save
    #   @returns img Image with overlayed detections
    #
    def visualize_detections(self, img, detections: list, file_name: str):
        
        
        #cv2.polylines(img, np.int32([self.roi]), 1, (0, 255, 0), 1, cv2.LINE_AA)
        cat_dict = {3: 'Car', 6: 'Bus', 8: 'Truck'}

        
        for category in [3, 6, 8]:
            
            vehicle = list(filter(lambda x: x[5] == category, detections))

            for bbox in vehicle:

                top_left = (int(bbox[0]), int(bbox[1]))
                bottom_right = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

                text = '{}-{}'.format(cat_dict[category], np.around(bbox[4], 3))
                cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite(file_name, img)

        self.out_video.write(img)
        return img


    def process_detections(self, detections):

        master_dict = {}
        for index, prediction in enumerate(detections):
            
            frameCounter = index + 1
            master_dict[frameCounter] = []
            pred = [item for sublist in prediction for item in sublist]

            for detection in pred:

                class_id = copy.copy(detection[6])
                bbox = copy.copy(detection[1:5])
                score = copy.copy(detection[5])
                bbox_list = []

                # Keep only car, bus and truck detections, discard the rest
                if class_id == 3 or class_id == 6 or class_id == 8:

                    # convert [x, y, width, height] to [xmin, ymin, xmax, ymax]
                    #bbox[2:4] += bbox[0:2]

                    #class_id = class_map[class_id]

                    # convert [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
                    bbox_list = [bbox[1],bbox[0],bbox[3],bbox[2]]

                    bbox_list.extend([score, class_id]) 
            

                    #if class_id not in master_dict[frameCounter]:
                    #    master_dict[frameCounter][class_id] = []
                    #master_dict[frameCounter][class_id].append(tuple(bbox_list))
                    master_dict[frameCounter].append(tuple(bbox_list))
        
        
        
        return master_dict



    def generate_predictions(self):

        
        image_list = sorted(os.listdir(os.path.join(self.default['data_dir'], self.cam_ident)))

        roi_list = self.get_roi(self.config['roi_path'])
        roi_polygon = np.array(roi_list)
        image_detections = []

        for i in tqdm.tqdm(range(0, len(image_list), self.batch_size)):

            imgs = []
            image_batch = image_list[i: i + self.batch_size]
            for image_path in image_batch:
                full_path = os.path.join(self.default['data_dir'], self.cam_ident, image_path)
                imgs.append(self.create_roi_mask(roi_polygon, full_path))

            predictions = self.driver.serve_images(imgs)
            image_detections.append(predictions.tolist())
        
        master_dict = self.process_detections(image_detections)
        '''
        if int(self.config['visualize']) == 1 and self.batch_size == 1:
            full_path = os.path.join(self.default['data_dir'], self.cam_ident, image_path)
            img = cv2.imread(full_path)
            file_name = os.path.join(self.out_dir, image_list[i])
            self.visualize_detections(img, predictions, file_name)

        '''
        with open(os.path.join(self.out_dir, self.cam_ident + '.pkl' ), 'wb') as handle:
            pickle.dump(master_dict, handle)
        

if __name__ == "__main__":
    
    
    cameras = sorted(os.listdir('/mnt/dataset_A_frames'))



    driver = inference.ServingDriver('efficientdet-d6','src/checkpoints/efficientdet-d6', batch_size=1)
    driver.build()
    for cam in cameras:
        
        config_file = os.path.join('config', cam + '.ini')
        print('###### PROCESSSING ' + cam + ' ########')
        config = configparser.ConfigParser()
        config.read(config_file)

        dt = EfficientDet(config, driver)
        detection_dict_list = dt.generate_predictions()



