# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np
import time
from datetime import datetime

from collections import Counter, defaultdict

from transformers import GPT2Tokenizer, AutoTokenizer

from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel
from utils.data import load_data

def main(logger, args):
    max_length_per_example = 256
    max_length = 256
    # if args.use_demonstrations:
    #     max_length = min(max_length * args.test_k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.batch_size, max_length, max_length_per_example))

    if args.dataset is not None:
        train_data = load_data(args.dataset, "train", args.k, seed=args.seed, 
                datasets=None if args.dataset is None else args.dataset.split(","),
                data_dir=args.data_dir)
    elif args.task is not None:
        train_data = load_data(args.task, "train", args.k, seed=args.seed, 
                config_split=args.split, data_dir=args.data_dir)
    else:
        print("please specify a train dataset/task.")
        exit(1)

    train_counter = Counter()
    for dp in train_data:
        train_counter[dp["task"]] += 1
    if args.local_rank <= 0:
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))
        logger.info("%s on %s (%d train)" % (args.method, args.dataset, len(train_counter)))

    ######### load tensorize data
    metaicl_data = MetaICLData(logger, args.gpt2, args.method, args.use_demonstrations,
                               args.test_k, max_length, max_length_per_example,
                               do_tensorize=args.do_tensorize,
                               tensorize_dir=args.tensorize_dir,
                               n_process=args.n_process, n_gpu=args.n_gpu, 
                               local_rank=args.local_rank, 
                               n_prefix_tokens=args.n_prefix_tokens,
                               task_counts=train_counter)
    keyword = args.dataset if args.dataset is not None else args.task
    metaicl_data.tensorize_for_training(train_data, keyword=keyword, 
        seed=args.seed, use_random_english_words=args.use_random_english_words)

    if args.do_tensorize:
        return

    ######## actual training part

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.train_seed)

    num_training_steps = args.num_training_steps
    save_period = 1000
    log_period = 100

    if args.no_masking:
        metaicl_data.tensorized_inputs["token_type_ids"] = torch.ones_like(metaicl_data.tensorized_inputs["input_ids"])
    metaicl_data.print_tensorized_example()

    logger.info(args.out_dir)

    if args.local_rank<=0 and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    with open(os.path.join(args.out_dir, 'task2token.json'), 'w') as f:
        json.dump(metaicl_data.prefix_token_ids, f, ensure_ascii=False)

    metaicl_model = MetaICLModel(args.gpt2, logger, 
        args.out_dir, args.fp16, args.local_rank, True, args.n_prefix_tokens,
        prefix_embed_file=args.prefix_embed_file, task_counts=train_counter)
    metaicl_model.to_device()
    metaicl_model.setup_optimizer(args.optimization, num_training_steps, args.lr,
                                  args.weight_decay, args.warmup_steps)
    metaicl_model.parallel()
    metaicl_model.train()
    metaicl_model.do_train(metaicl_data, args.batch_size, num_training_steps, save_period, log_period)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_tensorize", default=False, action="store_true")
    parser.add_argument("--tensorize_dir", type=str, default="tensorized")
    parser.add_argument("--n_gpu", type=int, default=8)
    parser.add_argument("--n_process", type=int, default=40)
    parser.add_argument("--n_prefix_tokens", type=int, default=10)

    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--log_dir", default='logs', type=str)
    parser.add_argument("--prefix_embed_file", default=None, type=str)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--test_k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--no_masking", default=False, action="store_true")
    parser.add_argument("--use_random_english_words", default=False, action="store_true")

    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--method", type=str, default="direct", 
        choices=["direct", "channel", "causal", "anti-causal"])
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    parser.add_argument("--optimization", type=str, default="adamw")
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    log_file = os.path.join(args.log_dir, datetime.fromtimestamp(time.time()).isoformat())
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
