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
        self.actv = nn.ReLU()
        self.scaling = 1.0
        self.gating = False
        print(names)

        if isinstance(names, str):
            names = [names]
        self.adapter_dict = {}
        for name in names:
            if 'adapter' in name:
                n = f'{name}_down'
                setattr(self, n, nn.Linear(model_dim, model_dim//adapter_reduction_factor).to(device))
                m = getattr(self, n)
                m.apply(init_bert_weights)
                for p in m.parameters():
                    p.requires_grad = True
                n = f'{name}_up'
                setattr(self, n, nn.Linear(model_dim//adapter_reduction_factor, model_dim).to(device))
                m = getattr(self, n)
                m.apply(init_bert_weights)
                for p in m.parameters():
                    p.requires_grad = True

            elif name in ['gating']:
                # for each client we init a spec gating
                setattr(self, f'{name}_module', nn.Linear(model_dim, 2).to(device))
                m = getattr(self, f'{name}_module')
                m.apply(init_bert_weights)
                for p in m.parameters():
                    p.requires_grad = True

        if hasattr(self, 'adapter_2_down'):
            for m in [self.adapter_2_down, self.adapter_2_up]:
                for p in m.parameters():
                    p.requires_grad = False

    def deactivate_gating(self):
        self.gating = False

    def activate_gating(self):
        self.gating = True

    def set_active_adapter(self, name):
        if isinstance(name, str):
            self.active_adapter_down = getattr(self, f'{name}_down')
            self.active_adapter_up = getattr(self, f'{name}_up')

        if name == 'adapter_0':
            for m in [s