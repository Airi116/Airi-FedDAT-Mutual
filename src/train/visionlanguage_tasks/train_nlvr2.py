import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import math
import shutil
import pickle as pkl
import copy
import pdb
from tqdm import tqdm
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from src.data.visionlanguage_datasets.nlvr2_dataset import build_nlvr2_dataloader
from src.train.visionlanguage_tasks.task_trainer import TaskTrainer
from src.utils.wandb import wandb_logger

sys.path.insert(0, '.')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class NLVR2Trainer(TaskTrainer):

    def __init__(self,
                 logger,
                 args: argparse.Namespace,
                 task_configs: Dict,
                 model_config: Dict,
                 device: torch.device,
                 task_key,
                 task_output_dir,
                 accelerator):

        '''
        Initializes a Trainer that handles training of a model on the VCR task

        args: Arguments pro