import torch
import torch.nn as nn
import torch.nn.functional as F

def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class Adapter(nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """
    def __init__(self, names, device, model_dim=768, adapter_reduction_factor=16):
        super().__init__()
        self.actv = nn.ReLU(