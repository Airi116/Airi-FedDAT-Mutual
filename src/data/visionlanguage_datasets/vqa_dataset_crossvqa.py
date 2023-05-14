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
            self.ans2label_file = file_root + "GQA/ans2label_fed.pkl"
        elif "vizwiz" in task_key:
            self.ans2label_file = file_root + "vizwiz/ans2label_fed.pkl"
        elif "clove_scene" in task_key:
            scene_key = task_key.replace("clove_", "")
            root = "/CLOVE/json/scene"
            for fname in os.listdir(root):
                if scene_key in fname and 'ans2label' in fname:
                    break
            self.ans2label_file = os.path.join(root, fname)
        elif "clove_function" in task_key:
            k = task_key.replace("clove_function_", "")
            function_key = {"a": "attribute",
                 "b": "knowledge",
                 "c": "logical",
                 "d": "object",
                 "e": "relation",
            }[k]

            root = file_root + "/CLOVE/json/function"
            for fname in os.listdir(root):
                if function_key in fname and 'ans2label' in fname:
                    break
            self.ans2label_file = os.path.join(root, fname)

        # Load mapping from answers to labels
        self.ans2label = pkl.load(open(self.ans2label_file, "rb"))
        self.label2ans = {v: k for k, v in self.ans2label.items()}
        self.num_labels = 100 # len(self.label2ans)

        self.cached_data_file = os.path.join(
            self.data_dir, "cached_vqa_data", "vqa_{}.pkl".format(split)
        )
        if task_key in ["gqa", "vizwiz"]:
            self.cached_data_file = os.path.join(
                self.data_dir, "{}_fed.pkl".format(split.split('_')[0])
            )
        elif "clove" in task_key:
            if "test" in split:
                self.cached_data_file = self.ans2label_file.replace("ans2label", "val")
            else:
                self.cached_data_file = self.ans2label_file.replace("ans2label", split.split('_')[0])

        if os.path.isfile(self.cached_data_file):
            # Load cached data
            # self.data = pkl.load(open(self.cached_data_file, "rb"))
            # if not os.path.exists(self.cached_data_file):
            if task_key not in ["gqa", "vizwiz"] and "clove" not in task_key:
                p = self.cached_data_file.replace('.', '_fed.')
            else:
                p = self.cached_data_file
            self.data = pkl.load(open(p, "rb"))
            for d in self.data:
                if "question_input_ids" not in d.keys():
                    d["question_input_ids"] = []
            random.shuffle(self.data)
            # ct = 0
            # temp = []
            # for d in self.data:
            #     f = True
            #     for l in d["labels"]:
            #         if l>=100:
            #             f = False
            #             break
            #     if f and len(d["labels"])>0:
            #         temp.append(d)
            # self.data = []
            # for d in temp:
            #     if 'train' in self.split:
            #         if random.random() <= 2500.0/len(temp):
            #             self.data.append(d)
            #     else:
            #         if random.random() <= 500.0/len(temp):
            #             self.data.append(d)
            # pkl.dump(self.data, open(p, "wb"))

        else:
            # Create map from question id to question
            # vqav2 & abstractqq
            # questions = json.load(open(self.questions_file))['questions']
            questions = json.load(open(self.questions_file))
            qid2qdata = {x["question_id"]: x for x in questions}

            # Create data for each annotation
            # vqav2 & abstract
            # annotations = json.load(open(self.annotations_file))['annotations']
            annotations = json.load(open(self.annotations_file))
            self.data = []
            # annotations_dict = {x['question_id']: x for x in annotations}
            # for ques in questions:
            #   qid = ques['question_id']
            #   image_id = int(ques['image'].split('/')[-1].split('.')[0].split('_')[-1])
            #   anno = annotations_dict[qid]
            #   assert image_id == anno['image_id']
            for anno in annotations:
                qid = anno["question_id"]
                # vqav2 & abstract
                # image_id = int(anno['image'].split('/')[-1].split('.')[0].split('_')[-1])
                # pvqa
                image_id = anno["image"].split("/")[-1].split(".")[0]
                # image_id = anno['image'].strip('.jpg').split('/')[-1]
                # image_id = int(anno['image'].strip('.jpg').split('-')[0])

                # Retrieve the question for this annotation
                qdata = qid2qdata[qid]
                # assert qdata['image_id'] == image_id
                # qdata_img_id = int(qdata['image'].split('/')[-1].split('.')[0].split('_')[-1])
                # pvqa
                qdata_img_id = qdata["image"].split("/")[-1].split(".")[0]
                # qdata_img_id = qdata['image'].strip('.jpg').split('/')[-1]
                # qdata_img_id = int(qdata['image'].strip('.jpg').split('-')[0])
                assert qdata_img_id == image_id
                question = qdata["question"]
                if self.tokenizer is not None:
                    tokens = self.tokenizer.tokenize(question)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    tokens = []
                    input_ids = []

                # Map from each crowdsourced answer to occurrences in annotation
                # answers = [a['answer'] for a in anno['answers']]
                answers = anno["answer"]
                answer_count = defaultdict(int)
                for ans in answers:
                    answer_count[ans] += 1

                # Get label and score (0.3/0.6/1) corresponding to each crowdsourced answer
                labels = []
                scores = []
                answers = []
                for answer in answer_count:
                    if answer not in self.ans2label:
                        continue
                    labels.append(self.ans2label[answer])
                    if task_key in ["toronto", "pvqa", "med", "art", "gqa"] or "clova" in task_key:
                        score = 1 / answer_count[answer]
                    else:
                        score = get_score(answer_count[answer])
                    scores.append(score)
                    answers.append(answer)
                correct_answer = answers[0]

                # Store pre-processed example
                example = {
                    "question_id": qid,
                    "image_id": image_id,
                    "question": question,
                    "question_input_ids": input_ids,
                    "correct_answer": correct_answer,
                    "labels": labels,
                    "answers": answers,
                    "scores": scores,
                }
            # if not os.path.isdir(self.cached_data_file):
            #     os.makedirs(self.cached_data_file)
            pkl.dump(self.data, open(self.cached_data_file, "wb"))

        self.n_examples = len(self.data)
        # for data in self.data:
        #    data['correct_answer'] = data['correct_answer'][0]
        # pkl.d