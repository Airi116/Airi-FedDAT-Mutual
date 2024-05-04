
import json
import pprint
import os
import pdb
import numpy as np
from collections import defaultdict
import sys
import glob


def merge_all_results(all_scores, fns, backbone):
    for fn in fns:
        with open(fn, "r") as f:
            rdict = json.load(f)

        name = os.path.basename(fn).split('_')[:-1]
        if len(name) == 2:
            algo = backbone
            t_order = 'task0'
            t_name = 'NA'
        elif len(name) == 3:
            algo = 'single'
            t_order, t_name = name[1:]
        elif len(name) == 4:
            t_order, t_name, algo = name[1:]

        for k in rdict.keys():
            scores = np.array(list(rdict[k].values()))
            test_scores, dev_scores = scores[:, 0], scores[:, 1]

            n_shot = k.split('-')[-1]
            if 'vision' in fn:
                all_scores[algo][t_order][t_name][n_shot] = f'{test_scores[0]:.1f}'
            else:
                #               assert scores.shape == (3,3)
                all_scores[backbone][algo][t_order][t_name][n_shot] = f'{test_scores.mean():.1f} ±{test_scores.std():.1f}'

    return all_scores


def dump_outputs(all_scores, task_name):
    out_fn = f"{task_name}.json"
    with open(out_fn, "w") as outfile: