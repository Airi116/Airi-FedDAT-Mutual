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
        self.text_encoder = BertModel.from_pretrained(BERT_LOCAL_PATH, local_files_only=True, config=config_encoder, add_pooling_layer=False)
        self.text_decoder = BertLMHeadModel.from_pretrained(BERT_LOCAL_PATH, local_files_only=True, config=config_decoder)

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config["image_res"], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                adapter_config=config["adapter_config"]  if 'adapter_config' in config.keys() else None)
            ##online model changed to local model
            self.text_encoder_m = BertModel.from_pretrained(BERT_LOCAL_PATH, local_files_only=True, config=config_encoder, add_pooling_layer=False)
            self.text_decoder_m = BertLMHeadModel.from_pretrained(BERT_LOCAL_PATH, local_files_only=True, config=config_decoder)
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_decoder, self.text_decoder_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

    def set_active_gating(self):
        for i in range(len(self.text_encoder.encoder.layer)):
            self.text_encoder.encoder.layer[i].output.adapter.gating_activated = True

        for i in range(len(self.text_decoder.bert.encoder.layer)):
            self.text_decoder.bert.encoder.layer[i].output.adapter.gating_activated = True

        for i in range(len(self.visual_encoder.blocks)):
            self.visual_encoder.blocks[i].adapter.gating_activated = True

    def forward(self, image, question, answer=None, alpha=0, k=None, weights=None, train=True, prev_f=None):

        image_embeds = self.visual_encoder(image)
        if prev_f is not None:
            image_embeds += self.pnn_layer_visual(prev_f[0])

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            """
            k: number of answers for each question
            weights: weight for each answer
            """
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            if prev_f is not None:
                text_embeds = self.pnn_layer_text(prev_f[1])

            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [question.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            if self.distill:
                with torch.no_grad():
                    # to do
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)
                    question_output_m = self.text_encoder_m(question.input_ids,
                             