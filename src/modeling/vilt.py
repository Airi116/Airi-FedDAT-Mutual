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

    def reallocate_text_image(self, pretrained_pos_emb: torch.Tensor, max_len: int, img_size: int):  # not used
        """
        Re-allocate some of the image token slots to the language position embeddings

        Args:
        pretrained_pos_emb: Pretrained position embeddings which are extended for creating extra language token slots
        max_len: new maximum length of language inputs (original is 40 for ViLT)
        img_size: new size of images, which requires fewer slots so that extra image slots can be allocated to extending language slots
        """
        vilt_config = self.vilt.config
        assert max_len % vilt_config.max_position_embeddings == 0

        self.reset_processor(max_len, img_size)

        # copy the pretrained positional embeddings to support texts with longer max_len
        extended_pos_emb = torch.cat([pretrained_pos_emb \
                                      for _ in range(0, max_len, vilt_config.max_position_embeddings)], 0)
        # extend & re-init Embedding
        self.vilt.embeddings.text_embeddings.position_embeddings = \
            nn.Embedding(max_len, vilt_config.hidden_size).from_pretrained(extended_pos_emb, freeze=False)


        # extend self.position_ids
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vilt/modeling_vilt.py#L274
        self.vilt.embeddings.text_embeddings. \
            register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def process_inputs(self, images: List, texts: List[str]) -> Dict:
        """
        Returns encodings that can be inputted to the ViLT transformer

        Args:
        images - list of PIL Image objects
        texts - list of text strings

        Returns:
        encodings - dictionary, where each key corresponds to a different argument of the vilt's forward method
        """
        encodin