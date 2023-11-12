import copy
import os
import sys
import logging
from accelerate.logging import get_logger
import itertools
import pdb
import time
from PIL import Image
from typing import List, Dict
from typing_extensions import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer, BertModel
from transformers import ViltConfig, ViltProcessor, ViltModel
from transformers import BertTokenizerFast
from transformers import logging as transformers_logging

from src.modeling.continual_learner import EncoderWrapper, ContinualLearner
from src.modeling.adaptered_output import Adaptered_ViltOutput

class ViltEncoderWrapper(EncoderWrapper):

    def __init__(self,
                 processor: ViltProcessor,
                 vilt: ViltModel,
                 device: torch.device):
        """
        Wrapper around Vilt model from huggingface library
        this is the class that gets saved during checkpointing for continual learning
        args:
        processor - instance of ViltProcessor
        vilt - instance of ViltModel class
        device - gpu/cuda
        """

        super().__init__()
        self.processor = processor
        self.vilt = vilt
        self.device = device
        # Yao: original:
        # self.processor.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # Yao: changed to the following:
        BERT_LOCAL_PATH = './models/bert-base-uncased'
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(BERT_LOCAL_PATH, local_files_only=True)

        self.max_text_length = self.vilt.config.max_position_embeddings
        self.encoder_dim = self.vilt.config.hidden_size

        self.expand_modality_type_embeddings()

    def reset_processor(self, max_text_length: int, img_size: tuple):
        self.max_text_length = max_text_length
        self.processor.feature_extractor.size = img_size

    def reallocate_text_image(self, pretrained_