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
from src.utils.image_utils import resize_image


class vizwizImagesDataset(Dataset):

    def __init__(self, coco_dir: str, data_dir: str, visual_input_type: str, task_key: str, image_size=(384, 640), transform=None):

        '''
        Initializes an MSCOC