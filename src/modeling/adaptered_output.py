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
from sr