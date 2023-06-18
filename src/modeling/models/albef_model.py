from functools import partial
from src.modeling.models.vit import VisionTransformer
from src.modeling.models.xbert import BertConfig, BertModel, BertLMHeadModel

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 text_decoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config["distill"]

        self.visual_encoder = VisionTransformer(
            img_size=config["image_res"], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            adapter_config=config["adapter_config"] if 'adapter_config' in config.keys() else None
            )

        config_encoder = BertConfig(**config["bert_config"])
        config_decoder = BertConfig(**config["bert_config"])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6

        if 'adapter_config' in config.keys():
            config_encoder.adapter_config = config["adapter_config"]
            config_decoder.adapter_config = config["adapter_config"]

        # self.text_encoder = BertModel.from_pretrained(text_encoder,force_download=True, config=config_encoder, add_pooling_layer=False)
        BERT_LOCAL_PATH = "/home/stud/yyang/CARVEN/bert-base-uncased"
        