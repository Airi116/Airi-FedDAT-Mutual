from src.modeling.albef import *
from src.modeling.vilt import *
from src.modeling.vilt_clf import *
from src.modeling.viltbert import *

ALLOWED_CL_ENCODERS = ["vilt", "viltbert", "flava", "albef_distill", "albef_no_distill"]

#### for ViLT
vilt_config = {
    'encoder_dim': 768,
    'visual_input_type': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'batch2inputs_converter': convert_batch_to_vilt_input_dict,
    'encoder_name': 'ViLT'
}


viltbert_config = {
    "encoder_dim": 768,
    "visual_input_type": "pil-image",
    "encoder_class": ViltBertEncoderWrapper,
    "batch2inputs_converter": convert_batch_to_viltbert_input_dict,
    "encoder_name": "ViLT-BERT",
}
viltbert_lang_seq_config = {
    "encoder_dim": 768,
    "visual_input_type": "pil-image",
    "encoder_class": ViltBertEncoderWrapper,
    "classifier_class": ViltBertForSequenceClassification,
    "batch2inputs_converter": convert_seq_batch_to_vilt_input_dict,
}
viltbert_lang_mc_config = {
    "encoder_dim": 768,
    "visual_input_type": "pil-image",
    "encoder_class": ViltBertEncoderWrapper,
    "classifier_class": ViltBertForMultipleChoice,
    "batch2inputs_converter": convert_mc_batch_to_vilt_input_dict,
}

config_bert = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 1