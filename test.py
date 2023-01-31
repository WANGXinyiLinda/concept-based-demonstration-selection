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
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from collections import Counter, defaultdict

from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel
from gpt3 import GPT3Model

from utils.data import load_data

def main(logger, args):
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)

    if args.gpt.startswith("gpt3"):
        metaicl_model = GPT3Model(args.gpt[5:], args.api, logger)
        add_newlines = True
    else:
        add_newlines = not args.gpt.startswith("gpt2")
        
        task_counts = None
        if args.prefix_embed_file is not None:
            model_dir = Path(args.prefix_embed_file).parent.absolute()
            if os.path.exists(os.path.join(model_dir, 'task2token.json')):
                with open(os.path.join(model_dir, 'task2token.json')) as f:
                    task_counts = json.load(f)

        metaicl_model = MetaICLModel(args.gpt, logger, args.out_dir, 
            soft_prefix=args.use_soft_prefix or args.use_soft_postfix, 
            n_tokens=args.n_prefix_tokens, prefix_embed_file=args.prefix_embed_file,
            task_counts=task_counts)
        metaicl_model.cuda()
        metaicl_model.eval()

    if "most_similar" in args.prior:
        embedding_model = sentence_transformers.SentenceTransformer(args.embedding_model)
        embedding_model.cuda()
        embedding_model.eval()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data
    max_length_per_example = 256
    if args.use_demonstrations:
        max_length = min(max_length_per_example * args.k, args.max_length)
    else:
        max_length = max_length_per_example

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    if args.use_soft_prefix or args.use_soft_postfix:
        metaicl_data = MetaICLData(logger, args.gpt, args.method,
            args.use_demonstrations, args.use_instruction, args.k, max_length, 
            max_length_per_example, add_newlines=add_newlines, 
            n_prefix_tokens=args.n_prefix_tokens, prefix=args.use_soft_prefix,
            task_counts=task_counts, prefix_token_ids=task_counts)
    else:
        metaicl_data = MetaICLData(logger, args.gpt, args.method,
            args.use_demonstrations, args.use_instruction, args.k,
            max_length, max_length_per_example, add_newlines=add_newlines)
    
    def load_configs():
        config_dict = {}
        for task in os.listdir("config/tasks"):
            if not task.startswith("unifiedqa:"):
                with open(os.path.join("config/tasks", task), "r") as f:
                    config = json.load(f)
                config_dict[task.split(".")[0]] = config
        return config_dict
    config_dict = load_configs()

    all_f1s = []
    all_accs = []
    errors = []
    all_scores = []
    all_dlls = []
    all_predictions = []
    seeds = args.seed.split(",")
    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"

    for seed in seeds:
        np.random.seed(int(seed))

        ### data ...
        train_data = load_data(args.task, "train", args.k, 
            seed=seed, config_split=config_split,
            datasets=None if args.dataset is None else args.dataset.split(","),
            data_dir=args.data_dir, full_train=True)

        dev_data = load_data(args.task, args.split, args.k, seed=seed, config_split=config_split,
                             datasets=None if args.dataset is None else args.dataset.split(","), 
                             data_dir=args.data_dir, full_train=True)

        if args.use_random_english_words:
            from english_words import get_english_words_set
            english_words_set = sorted(get_english_words_set(['web2']))

        train_counter = Counter()
        dev_counter = Counter()
        for dp in train_data:
            train_counter[dp["task"]] += 1
        for dp in dev_data:
            dev_counter[dp["task"]] += 1
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))
        for k, v in dev_counter.items():
            logger.info("[Dev] %s\t%d" % (k, v))

        logger.info("%s on %s (%d train, %d dev)" % (args.method, args.task, len(train_counter), len(dev_counter)))

        for test_task in dev_counter:
            curr_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
            assert len(curr_dev_data)>0
            if args.test_size < len(curr_dev_data) and args.split=="test":
                subsample_ids = np.random.choice(len(curr_dev_data), args.test_size, replace=False)
                curr_dev_data = np.array(curr_dev_data)[subsample_ids].tolist()

            config_file = "config/tasks/{}.json".format(test_task)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"] == "classification"
            is_multi_choice = config["task_type"] == "multi-choice"
            if is_classification:
                options = curr_dev_data[0]["options"]
                assert np.all([d["options"]==options for d in curr_dev_data])

            if args.use_demonstrations:
                _train_data = [dp for dp in train_data if dp["task"]==test_task]
                if args.train_size > 0:
                    subsample_ids = np.random.choice(len(_train_data), args.train_size, replace=False)
                    curr_train_data = np.array(_train_data)[subsample_ids].tolist()
                else:
                    curr_train_data = _train_data

                if "most_similar" in args.prior:
                    all_embedding_dir = os.path.join(args.embedding_dir, test_task, 
                        args.embedding_model)
                    all_embedding_file = os.path.join(all_embedding_dir, 'train.npy')
                    if os.path.isfile(all_embedding_file):
                        all_embeddings = np.load(all_embedding_file)[subsample_ids]
                    else:
                        os.makedirs(all_embedding_dir, exist_ok=True)
                        all_embeddings = embedding_model.encode(
                            [d["input"] for d in _train_data])
                        np.save(all_embedding_file, all_embeddings)
                        all_embeddings = all_embeddings[subsample_ids]

                    dev_embedding_dir = os.path.join(args.embedding_dir, test_task, 
                            args.embedding_model)
                    dev_embedding_file = os.path.join(dev_embedding_dir, 'dev.npy')
                    if os.path.isfile(dev_embedding_file):
                        dev_embeddings = np.load(dev_embedding_file)
                    else:
                        os.makedirs(dev_embedding_dir, exist_ok=True)
                        dev_embeddings = embedding_model.encode(
                            [d["input"] for d in curr_dev_data])
                        np.save(dev_embedding_file, dev_embeddings)

                    sims = sentence_transformers.util.cos_sim(dev_embeddings, all_embeddings)
                    sims = sims.cpu().detach().numpy()
                    dev_train_exp_sims = np.exp(sims/args.similarity_temperature)

                priors = set(args.prior)
                use_difficulty = len(set(["easiest", "hardest"]).intersection(priors))>0

                if len(args.prior) > 0:
                    sorted_priors = sorted(args.prior)
                    prior_text = '_'.join(sorted_priors)
                    if use_difficulty:
                        prior_text += f"_diff={args.difficulty}"
                else:
                    prior_text = 'uniform'

                if use_difficulty:

                    if args.difficulty == "concept_likelihood":
                        if args.train_size > 0:
                            concept_dir = os.path.join(args.concept_dir, 
                                f"{test_task}-train-{seed}")
                        else:
                            concept_dir = os.path.join(args.concept_dir, 
                                f"{test_task}-train")

                        if os.path.exists(concept_dir):
                            logger.info("loading saved concept likelihoods")
                            all_nll = np.load(os.path.join(concept_dir, f'{test_task}-nll.npy'))
                            gt_labels = np.load(os.path.join(concept_dir, f'{test_task}-gt.npy'))
                            
                        else:
                            assert args.prefix_embed_file is not None
                            model_dir = Path(args.prefix_embed_file).parent.absolute()
                            if os.path.exists(os.path.join(model_dir, 'task2token.json')):
                                with open(os.path.join(model_dir, 'task2token.json')) as f:
                                    task_counts = json.load(f)
                            else:
                                task_counts = None
                            
                            logger.info("start running soft prefix model")
                            start_time = time.time()
                            concept_model = MetaICLModel("gpt2-large", 
                                logger, args.out_dir, soft_prefix=True, 
                                n_tokens=args.n_prefix_tokens,
                                prefix_embed_file=args.prefix_embed_file, 
                                task_counts=task_counts)
                            concept_model.cuda()
                            concept_model.eval()        

                            concept_data = MetaICLData(logger, "gpt2-large", 
                                args.method, False, args.use_instruction, args.k,
                                max_length, max_length_per_example, 
                                add_newlines=add_newlines, 
                                n_prefix_tokens=args.n_prefix_tokens,
                                prefix=False, task_counts=task_counts, 
                                prefix_token_ids=task_counts)

                            _, _, _, _, all_nll, gt_labels = run(test_task, 
                                concept_data, concept_model, None, curr_train_data, 
                                is_classification, None, config_dict)

                            del concept_model
                            del concept_data
                            os.makedirs(concept_dir)
                            np.save(os.path.join(concept_dir, f'{test_task}-nll.npy'), all_nll)
                            np.save(os.path.join(concept_dir, f'{test_task}-gt.npy'), gt_labels)

                            logger.info(f"time use for computing {len(curr_train_data)} examples: {time.time()-start_time}")

                        opt_log_p = []
                        for _nll, l in zip(all_nll, gt_labels):
                            opt_log_p.append(-_nll[l]/args.concept_temperature)
                        opt_p = np.exp(opt_log_p)
                        difficulties = 1 - opt_p

                    elif args.difficulty == "concept_calibrated":
                        
                        if args.prefix_embed_file is not None:
                            model_dir = Path(args.prefix_embed_file).parent.absolute()
                            if os.path.exists(os.path.join(model_dir, 'task2token.json')):
                                with open(os.path.join(model_dir, 'task2token.json')) as f:
                                    task_counts = json.load(f)
                            else:
                                task_counts = None
                            
                        all_log_ps = []
                        difficulties = []
                        initial_mask = np.ones(len(curr_train_data), dtype=bool)

                        for task in task_counts:
                            if args.train_size > 0:
                                concept_dir = os.path.join(args.concept_dir, 
                                    f"{task}-train-{seed}")
                            else:
                                concept_dir = os.path.join(args.concept_dir, 
                                    f"{task}-train")

                            if os.path.exists(concept_dir):
                                logger.info("loading saved concept likelihoods")
                                all_nll = np.load(os.path.join(concept_dir, f'{task}-nll.npy'))
                                gt_labels = np.load(os.path.join(concept_dir, f'{task}-gt.npy'))
                                
                            else:
                                assert args.prefix_embed_file is not None
                                logger.info("start running soft prefix model")
                                start_time = time.time()
                                concept_model = MetaICLModel("gpt2-large", 
                                    logger, args.out_dir, soft_prefix=True, 
                                    n_tokens=args.n_prefix_tokens,
                                    prefix_embed_file=args.prefix_embed_file, 
                                    task_counts=task_counts)
                                concept_model.cuda()
                                concept_model.eval()        

                                concept_data = MetaICLData(logger, "gpt2-large", 
                                    args.method, False, args.use_instruction, args.k,
                                    max_length, max_length_per_example, 
                                    add_newlines=add_newlines, 
                                    n_prefix_tokens=args.n_prefix_tokens,
                                    prefix=False, task_counts=task_counts, 
                                    prefix_token_ids=task_counts, task=task)

                                _, _, _, _, all_nll, gt_labels = run(test_task, concept_data, 
                                    concept_model, None, curr_train_data, is_classification, 
                                    None, config_dict)

                                del concept_model
                                del concept_data
                                os.makedirs(concept_dir)
                                np.save(os.path.join(concept_dir, f'{task}-nll.npy'), all_nll)
                                np.save(os.path.join(concept_dir, f'{task}-gt.npy'), gt_labels)

                                logger.info(f"time use for computing {len(curr_train_data)} examples: {time.time()-start_time}")

                            log_p = []
                            for _nll, l in zip(all_nll, gt_labels):
                                log_p.append(-_nll[l]/args.concept_temperature)

                            if task == test_task:
                                opt_log_p = log_p
                            all_log_ps.append(log_p)
                        
                        z=0
                        for log_p in all_log_ps:
                            z += np.exp(log_p)
                        calibrated_p = np.exp(opt_log_p - np.log(z))
                        difficulties = 1-calibrated_p
                        
                    else:
                        print(f"{args.difficulty} is not defined.")
                        exit(1)

                    difficulties = np.array(difficulties)
                    assert len(difficulties) == len(curr_train_data)
                    print(difficulties)

                    sorted_diff = np.sort(difficulties)
                    min_diff = sorted_diff[0]
                    logger.info(f"min difficulty: {min_diff}")
                    max_diff = sorted_diff[-1]
                    logger.info(f"max difficulty: {max_diff}")
                    logger.info(f"average difficulty: {np.mean(difficulties)}")

                if "hardest" in args.prior or "easiest" in args.prior:
                    if "balanced" in args.prior and is_classification:
                        all_labels = np.array([d["output"] for d in curr_train_data])
                        all_ids = np.arange(len(curr_train_data))
                        _k = math.ceil(args.k/len(options))
                        top_ids = []
                        top_diff = []
                        for c in options:
                            curr_ids = all_ids[all_labels == c]
                            sorted_ids = curr_ids[np.argsort(difficulties[all_labels == c])]
                            if "hardest" in args.prior:
                                top_ids += list(sorted_ids[-_k:])
                                top_diff += list(sorted_diff[-_k:])
                            else:
                                top_ids += list(sorted_ids[:_k])
                                top_diff += list(sorted_diff[:_k])
                        top_ids = np.array(top_ids)
                        if "hardest" in args.prior:
                            demo_ids = top_ids[np.argsort(top_diff)[-args.k:]]
                        else:
                            demo_ids = top_ids[np.argsort(top_diff)[:args.k]]
                    else:
                        sorted_ids = np.argsort(difficulties)
                        if "hardest" in args.prior:
                            demo_ids = sorted_ids[-args.k:]
                        else:
                            demo_ids = sorted_ids[:args.k]
                    
                    if args.reorder:
                        demo_ids_perm = permutation(list(demo_ids))
                        demos_perm = [[curr_train_data[i] for i in _demo_ids] 
                            for _demo_ids in demo_ids_perm]
                        save_dir = os.path.join(args.concept_dir, "reorder")
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        if args.difficulty == "concept_likelihood":
                            curr_nll_path = os.path.join(save_dir, 
                                f"{test_task}-nll-demos_perm-{seed}.npy")
                            demo_ids_path = os.path.join(save_dir, 
                                f"{test_task}-reordered_demo_ids-{seed}.npy")

                            if os.path.exists(demo_ids_path):
                                demo_ids = np.load(demo_ids_path)

                            if os.path.exists(curr_nll_path):
                                logger.info("loading saved demo nlls")
                                all_nll = np.load(curr_nll_path)
                            else:
                                logger.info("start running soft prefix model")
                                start_time = time.time()
                                concept_model = MetaICLModel("gpt2-large", 
                                    logger, args.out_dir, soft_prefix=True, 
                                    n_tokens=args.n_prefix_tokens,
                                    prefix_embed_file=args.prefix_embed_file, 
                                    task_counts=task_counts)
                                concept_model.cuda()
                                concept_model.eval()

                                concept_data = MetaICLData(logger, "gpt2-large", 
                                    args.method, True, args.use_instruction, args.k,
                                    max_length, max_length_per_example, 
                                    add_newlines=add_newlines, 
                                    n_prefix_tokens=args.n_prefix_tokens,
                                    prefix=False, task_counts=task_counts, 
                                    prefix_token_ids=task_counts)

                                all_nll, gt_labels = run(test_task, 
                                    concept_data, concept_model, demos_perm, 
                                    None, is_classification, None, config_dict)

                                del concept_model
                                del concept_data

                                logger.info(f"time use for computing {len(demos_perm)} examples: {time.time()-start_time}")
                                np.save(curr_nll_path, all_nll)

                            opt_log_p = []
                            for _nll in all_nll:
                                opt_log_p.append(-_nll[0]/args.concept_temperature)
                            demo_ids = demo_ids_perm[np.argmax(opt_log_p)]
                            np.save(demo_ids_path, demo_ids)

                        elif args.difficulty == "concept_calibrated":
                            demo_ids_path = os.path.join(save_dir, 
                                f"{test_task}-reordered_demo_ids-cali-{seed}.npy")
                            if os.path.exists(demo_ids_path):
                                demo_ids = np.load(demo_ids_path)

                            if not os.path.exists(os.path.join(args.concept_dir, 
                                "reorder")):
                                os.makedirs(os.path.join(args.concept_dir, "reorder"))

                            if not os.path.exists(os.path.join(args.concept_dir, 
                                "prefix")):
                                os.makedirs(os.path.join(args.concept_dir, "prefix"))

                            all_log_ps = []
                            all_prefix_ps = []
                            for task in task_counts:
                                curr_nll_path = os.path.join(args.concept_dir, 
                                    "reorder", f"{task}-nll-demos_perm-{seed}.npy")

                                curr_prefix_p_path = os.path.join(args.concept_dir, 
                                    "prefix", f"{task}-p-{seed}.npy")

                                concept_model = MetaICLModel("gpt2-large", 
                                    logger, args.out_dir, soft_prefix=True, 
                                    n_tokens=args.n_prefix_tokens,
                                    prefix_embed_file=args.prefix_embed_file, 
                                    task_counts=task_counts)
                                concept_model.cuda()
                                concept_model.eval()

                                if os.path.exists(curr_nll_path):
                                    logger.info("loading saved demo nlls")
                                    all_nll = np.load(curr_nll_path)
                                else:
                                    start_time = time.time()
                                    concept_data = MetaICLData(logger, "gpt2-large", 
                                        args.method, True, args.use_instruction, args.k,
                                        max_length, max_length_per_example, 
                                        add_newlines=add_newlines, 
                                        n_prefix_tokens=args.n_prefix_tokens,
                                        prefix=False, task_counts=task_counts, 
                                        prefix_token_ids=task_counts, task=task)

                                    all_nll, _ = run(test_task, 
                                        concept_data, concept_model, demos_perm, 
                                        None, is_classification, None, config_dict)

                                    del concept_data

                                    logger.info(f"time use for computing {len(demos_perm)} examples: {time.time()-start_time}")
                                    np.save(curr_nll_path, all_nll)

                                del concept_model

                                log_p = []
                                for _nll in all_nll:
                                    log_p.append(-_nll[0]/args.concept_temperature)

                                if task == test_task:
                                    opt_log_p = log_p
                                all_log_ps.append(log_p)
                            
                            z=0
                            for log_p in all_log_ps:
                                z += np.exp(log_p)
                            calibrated_p = np.exp(opt_log_p - np.log(z))
                            demo_ids = demo_ids_perm[np.argmax(calibrated_p)]
                            np.save(demo_ids_path, demo_ids)

                elif "most_similar" in args.prior: 
                    sorted_sims = np.argsort(dev_train_exp_sims)
                    demo_ids = sorted_sims[:, -args.k:].reshape(-1)
                                
                else:
                    demo_ids = np.random.choice(len(curr_train_data), 
                        args.test_size*args.k)

                demonstrations = []
                for i in demo_ids:
                    demonstrations.append(curr_train_data[i])
                if len(demo_ids) != args.k:
                    demo_ids = np.reshape(demo_ids, (args.test_size, args.k))

                if args.use_random_english_words:
                    # create a mapping
                    prior_text += '_rand_words'
                    options = curr_dev_data[0]["options"]
                    mapping = {option: np.random.choice(english_words_set) for option in options}
                    new_options = list(mapping.values())
                    for dp_idx, dp in enumerate(demonstrations):
                        if dp["output"] in options:
                            demonstrations[dp_idx]["output"] = mapping[dp["output"]]
                            demonstrations[dp_idx]["options"] = new_options
                        else:
                            assert dp["output"] in new_options, (dp, new_options)
                    for dp_idx, dp in enumerate(curr_dev_data):
                        if dp["output"] in options:
                            curr_dev_data[dp_idx]["output"] = mapping[dp["output"]]
                            curr_dev_data[dp_idx]["options"] = new_options
                        else:
                            assert dp["output"] in new_options, (dp, new_options)
                
                if args.use_random_label:
                    prior_text += '_rand_label'
                    options = curr_dev_data[0]["options"]
                    for dp_idx, dp in enumerate(demonstrations):
                        assert dp["output"] in options, (dp, options)
                        demonstrations[dp_idx]["output"] = options[random.randint(0,len(options)-1)]
                    for dp_idx, dp in enumerate(curr_dev_data):
                        assert dp["output"] in options, (dp, options)
                        curr_dev_data[dp_idx]["output"] = options[random.randint(0,len(options)-1)]

                if len(demonstrations) == args.k:
                    save_path = None
                else:
                    demonstrations = np.reshape(demonstrations, 
                        (args.test_size, args.k))

                    if args.load_dir is None:
                        dir_name = f"{test_task}-{metaicl_data.method}-p={prior_text}"
                        if not args.use_fixed_val:
                            dir_name += '-nonfixed_val'
                        dir_name += f"-k={args.k}-s={seed}"
                        save_path = os.path.join(args.out_dir, dir_name)
                    else:
                        save_path = args.load_dir

            else:
                if args.load_dir is None:
                    if len(seeds) > 1:
                        save_path = os.path.join(args.out_dir, 
                            f"{test_task}-{metaicl_data.method}-{args.split}-s={seed}")
                    else:
                        save_path = os.path.join(args.out_dir, 
                            f"{test_task}-{metaicl_data.method}-{args.split}")
                else:
                    save_path = args.load_dir
                demonstrations = None
                

            f1, acc, pred, gt, nll, gt_label = run(test_task, metaicl_data, 
                        metaicl_model, demonstrations, curr_dev_data,
                        is_classification, save_path, config_dict)

            if save_path is not None and args.split=='train':
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, f'{args.split}-pred.npy'), pred)
                np.save(os.path.join(save_path, f'{args.split}-gt.npy'), gt_label)
                np.save(os.path.join(save_path, f'{args.split}-nll.npy'), nll)
                np.save(os.path.join(save_path, f'{args.split}-acc.npy'), acc)
train_size
            all_predictions.append(pred)
            logger.info("%s task (seed=%s): Macro-F1: %.1f, Accuracy: %.1f" % 
                (args.task, seed, 100*f1, 100*acc))
            all_f1s.append(f1)
            all_accs.append(acc)

    final_predictions = []
    for p in np.transpose(all_predictions):
        v, c = np.unique(p, return_counts=True)
        final_predictions.append(v[np.argmax(c)])
    final_f1, final_acc = metaicl_data.evaluate(final_predictions, gt, is_classification)
    logger.info("%s over %d target tasks with majority vote: Macro-F1: %.1f, Accuracy: %.1f" % 
        (args.task, len(all_f1s) // len(seeds), 100*final_f1, 100*final_acc))

    logger.info("%s over %d target tasks on average: Macro-F1: %.1f +- %.1f, Accuracy: %.1f +- %.1f" % 
        (args.task, len(all_f1s) // len(seeds), 100*np.mean(all_f1s), 100*np.std(all_f1s), 
        100*np.mean(all_accs), 100*np.std(all_accs)))

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")

def permutation(lst):
    if len(lst) == 0:
        return []
    if len(lst) == 1:
        return [lst]
 
    l = [] 
    for i in range(len(lst)):
       m = lst[i]
       remLst = lst[:i] + lst[i+1:]
       for p in permutation(remLst):
           l.append([m] + p)
    return l

def run(task, metaicl_data, metaicl_model, train_data, dev_data,
        is_classification, save_path, config_dict, return_all=False):

    if args.gpt.startswith("gpt3"):
        gpt3_dataloader, gpt3_metadata = metaicl_model.prepare_data(
            train_data if args.use_demonstrations else [],
            dev_data, args.method, batch_size=args.test_batch_size)
        losses, gpt3cache = metaicl_model.do_inference(gpt3_dataloader)	
        predictions, all_nlls, gt_labels, pred_labels = metaicl_model.do_predict(
            losses=losses, metadata=gpt3_metadata, return_nll=True)
    else:
        if args.use_instruction:
            instruction = config_dict[task]["instruction"]
            metaicl_data.tensorize(train_data, dev_data, instruction)
        else:
            metaicl_data.tensorize(train_data, dev_data)
        metaicl_data.print_tensorized_example()
        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
        assert len(losses)==len(metaicl_data)
        predictions, all_nlls, gt_labels, pred_labels = metaicl_model.do_predict(
            metaicl_data, losses=losses, return_nll=True)
    try:
        groundtruths = [dp["output"] for dp in dev_data]
        f1, acc = metaicl_data.evaluate(predictions, groundtruths, 
            is_classification, return_all)
        return f1, acc, predictions, groundtruths, all_nlls, gt_labels
    except:
        return all_nlls, gt_labels

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--use_soft_prefix", default=False, action="store_true")
    parser.add_argument("--use_soft_postfix", default=False, action="store_true")
    parser.add_argument("--n_prefix_tokens", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--prior", type=str, nargs='+', default=[], 
        choices=["most_similar", "easiest", "hardest"])
    parser.add_argument("--difficulty", type=str, default="length", 
        choices=["concept_likelihood", "concept_calibrated"])
    parser.add_argument("--reorder", default=False, action="store_true")

    parser.add_argument("--log_dir", default='logs', type=str)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--concept_dir", default=None, type=str)
    parser.add_argument("--prefix_embed_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")

    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")
    parser.add_argument("--use_random_label", default=False, action="store_true")

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--method", type=str, default="direct", 
        choices=["direct", "channel"])
    parser.add_argument("--gpt", type=str, default="gpt2-large", 
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                "gpt3-ada", "gpt3-babbage", "gpt3-curie", "gpt3-davinci", 
                "gpt3-text-ada-001", "gpt3-text-babbage-001", "gpt3-text-curie-001", 
                "gpt3-text-davinci-001", "gpt3-text-davinci-002", 
                "gpt3-code-davinci-002", "gpt3-text-davinci-003"])
    parser.add_argument("--api", type=str, default=None)

    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--embedding_dir", type=str, default='embedding')
    parser.add_argument("--embedding_model", type=str, default='all-mpnet-base-v2', 
        choices=['all-mpnet-base-v2'])
    parser.add_argument("--similarity_temperature", type=float, default=1.0)
    parser.add_argument("--concept_temperature", type=float, default=10.0)

    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, datetime.fromtimestamp(time.time()).isoformat())
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)