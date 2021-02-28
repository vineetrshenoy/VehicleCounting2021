import sys
import torch
import numpy as np
import os
import os.path as osp
import argparse
from PIL import Image
import torchvision.transforms as T
import saver
import warnings
import cv2

#from deep_sort.siamese_net import *
'''
import configparser

config = configparser.ConfigParser()
config.read(sys.argv[1])
'''

class SaverExtractor:

    def __init__(self):

        self.model = self.build_model()
        self.transforms = self.build_transform()
        self.count = 1

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.model = self.model.to(device)

            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(model)


    def build_model(self):
        model = saver.Baseline(num_classes=333, last_stride=1, neck_feat='after')
        model = saver.SAVER(baseline=model)
        model.eval()
        return model


    def build_transform(self):
        transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize([256] * 2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
                    ])
        return transform


    def read_image(self, img_path):
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        else:
            image = Image.open(img_path).convert('RGB')
        return image


    def get_crops(self, img, detections):

        crops = []
        for detect in detections:

            xmin = int(detect[0])
            ymin = int(detect[1])
            xmax = int(detect[2])
            ymax = int(detect[3])

            crop = img[ymin:ymax, xmin:xmax, :]
            
            #cv2.imwrite(str(self.count) + '.jpg', crop)
            #self.count += 1
            
            crop = self.transforms(crop)
            crops.append(crop)

        #crops = torch.stack(crops)
        return crops

    def get_input(self, img_list):
        if len(img_list) == 1:
            image = self.transforms(img_list[0])
            batch = image.unsqueeze(0)
            assert len(batch.shape) == 4
        else:
            batch = [self.transforms(image_path) for image_path in img_list]
            batch = torch.stack(batch, dim=0)
            assert len(batch.shape) == 4
        return batch


    def workflow(self, img, detections):
        '''
        1. Crop the images
        2. Perform transforms
        '''
        if len(detections) == 0:
            return []

        crops = self.get_crops(img, detections)
        batch = self.get_input(crops)


        if torch.cuda.is_available():
            batch = batch.cuda()


        with torch.no_grad():

            features = self.model(batch)

        return features.cpu().numpy()

if __name__ == "__main__":
    print('Hello World')




'''
class SiameseExtractor:

    def __init__(self):

        self.config = config['TRACKING']
        self.default = config['DEFAULT']

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.ToTensor()
        ])

        self.net = SiameseNetwork()
        self.net.load_state_dict(torch.load(self.config['deepsort_weights']))


    
    def get_crops(self, img, detections):

        crops = []
        for detect in detections:

            xmin = int(detect[0])
            ymin = int(detect[1])
            xmax = int(detect[2])
            ymax = int(detect[3])

            crop = img[ymin:ymax, xmin:xmax, :]
            crop = self.transforms(crop)
            crops.append(crop)

        crops = torch.stack(crops)
        return crops

    
    def get_features(self, img, detections):

        crops = self.get_crops(img, detections)
        features = self.net.forward_once(crops)
        features = features.detach().cpu().numpy()

        return features

'''