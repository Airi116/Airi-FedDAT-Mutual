import sys
import os
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from PIL import Image


class MSCOCOImagesDataset(Dataset):

    def __init__(self, coco_dir: str, data_dir: str, visual_input_type: str, task_key: str, image_size=(384, 640), transform=None):

        '''
        Initializes an MSCOCOImagesDataset instance that handles image-side processing for VQA and other tasks that use MS-COCO images
        coco_dir: directory that contains MS-COCO data (images within 'images' folder)
        visual_input_type: format of visual input to model
        image_size: tuple indicating size of image input to model
        '''

        self.images_dir = [os.path.join(coco_dir, dir) for dir in data_dir]
        self.image_size = image_size

        self.visual_input_type = visual_input_type
        assert visual_input_type in ['pil-image', 'raw', 'fast-rcnn']

        if task_key in ['art', 'med']:
            image_filenames = os.listdir(self.images_dir[0])
        elif task_key in ['pvqa']:
            image_filenames = os.listdir(self.images_dir[0]) + os.listdir(self.images_dir[1]) + os.listdir(self.images_dir[2])
        elif task_key in ['toronto', 'abstract']:
            image_filenames = os.listdir(self.images_dir[0]) + os.listdir(self.images_dir[1])
        self.imageid2filename = {}
        for fn in image_filenames:
            if task_key in ['abstract']:
                image_id = int(fn.strip('.png').split('_')[-1])
            elif task_key in ['toronto']:
                image_id = int(fn.strip('.jpg').split('_')[-1])
            elif task_key in ['pvqa']:
                image_id = fn.strip('.jpg')
            elif task_key in ['med']:
                image_id = fn.strip('.jpg').split('/')[-1]