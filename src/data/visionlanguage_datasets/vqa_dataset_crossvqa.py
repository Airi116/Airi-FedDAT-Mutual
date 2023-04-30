import copy
import sys
import os
import time
import json
import logging
import random
import re
import glob
import base64
from tqdm import tqdm
from collections import defaultdict, Counter
from torchvision import transforms
import pickle as pkl
import pdb
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from PIL import Image
from src.utils.image_utils import resize_image
from src.utils.vqa_utils import get_score, target_tensor

from src.data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from src.data.image_collation import image_collate


class VQADataset(Dataset):
    def __init__(
        self,
        logger,
        data_dir: str,
        images_dataset: MSCOCOImagesDataset,
        split: str,
        task_key: str,
        encoder_type,
        transform=None,
        **kwargs
    ):

        """
        Initiates the VQADataset - loads all the questions (and converts to input IDs using the tokenizer, if provided)
        and answers (including converting each to a numeric label, and a score based on occurence from annotators)
        Every item in self.data corresponds to a single QA pair, with a corresponding image

        Args:
        data_dir : path containing VQA questions and annotations. Also contains mapping from each answer in set of possible answers to a numerical label
        images_dataset : instance of MSCOCOImagesDataset, that is used to retrieve the MS-COCO image fo