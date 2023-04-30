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
        images_dataset : instance of MSCOCOImagesDataset, that is used to retrieve the MS-COCO image for each question
        split: either train/val split

        Returns:
        Loads all annotations into self.data, where each item is a single VQA pair
        """

        self.images_dataset = images_dataset
        if transform:
            self.images_dataset.pil_transform = transform
            self.images_dataset.use_albef = True
        self.data_dir = data_dir
        self.encoder_type = encoder_type
        if split=='test':
            split = 'test_small'
        self.split = split
        self.task_key = task_key

        file_root =  "./data/"

        self.tokenizer = kwargs["tokenizer"] if "tokenizer" in kwargs else None
        self.label2idxs = {}
        if "abstract" in task_key:
            self.questions_file = os.path.join(
                data_dir, "abstract_{}.json".format(split)
            )
            self.annotations_file = os.path.join(
                data_dir, "abstract_v002_val2015_annotations.json".format(split)
            )
            self.ans2label_file = os.path.join(file_root, "abstract/ans2label.pkl")
        elif "toronto" in task_key:
            self.annotations_file = os.path.join(
                data_dir, "toronto_{}.json".format(split)
            )
            self.questions_file = os.path.join(
                data_dir, "toronto_{}.json".format(split)
            )
            self.ans2label_file = os.path.join(file_root, "toronto/ans2label.pkl".format(split))
        elif "art" in task_key:
            self.annotations_file = os.path.join(file_root, "art/art_{}.json".format(split))
            self.questions_file = os.path.join(file_root, "art/art_{}.json".format(split))
            self.ans2label_file = os.path.join(file_root, "art/ans2label_small.pkl".format(split))
        elif "gqa" in task_key:
            self.ans2labe