# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import string
import numpy as np
import torch

def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False, data_dir='data', full_train=False):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task.strip()+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data = []
    num_data = 0
    for dataset in datasets:
        if split=='train':
            if full_train:
                data_path = os.path.join(data_dir, dataset, "{}_full_train.jsonl".format(dataset))
            else:
                data_path = os.path.join(data_dir, dataset,
                        "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        elif split=='dev':
            data_path = os.path.join(data_dir, dataset,
                        "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        elif split=='test':
            data_path = os.path.join(data_dir, dataset,
                        "{}_test.jsonl".format(dataset))
        else:
            print("choose split from [train, dev, test]")
            exit(1)

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                num_data += 1
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)

    return data

