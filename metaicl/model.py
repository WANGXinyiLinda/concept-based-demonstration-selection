# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM

class MetaICLModel(object):

    def __init__(self, gpt2="gpt2-large", logger=None, 
        out_dir=None, fp16=False, local_rank=-1, soft_prefix=False, n_tokens=10, 
        prefix_embed_file=None, task_counts=None):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank

        if self.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            ws = 1
        else:  # distributed mode
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1

        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None
        self.load(gpt2)
        self.soft_prefix = soft_prefix
        if soft_prefix:
            if task_counts is None:
                self.n_tokens = n_tokens
            else:
                self.n_tokens = n_tokens * len(task_counts)
            self.orig_vocab_size = self.model.get_input_embeddings().weight.size(0)
            print("original vocab size: ", self.orig_vocab_size)
            self.model.resize_token_embeddings(self.orig_vocab_size + self.n_tokens)
            self.new_vocab_size = self.model.get_input_embeddings().weight.size(0)
            assert self.new_vocab_size == self.n_tokens + self.orig_vocab_size
            if prefix_embed_file is not None:
                self.model.set_input_embeddings(torch.load(prefix_embed_file))
            else:
                self.model.get_input_embeddings().weight.data[-self.n_tokens:] = \
                    self.model.get_input_embeddings().weight.data[:self.n_tokens]
            self.model.tie_weights()
                    

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self, gpt2="gpt2-large"):

        model = AutoModelForCausalLM.from_pretrained(gpt2)
        self.model_name = gpt2

        if torch.__version__ == '1.14.0.dev20221208+cu117':
            self.model = torch.compile(model)
        else:
            self.model = model 

    def save(self, step, save_all=False):
        if self.local_rank <= 0:
            if save_all:
                model_state_dict = {key[7:] if key.startswith("module.") else key: value.cpu()
                                    for key, value in self.model.state_dict().items()}
                torch.save(model_state_dict, os.path.join(self.out_dir, "model-{}.pt".format(step)))
                self.logger.info("Saving model parameters at step=%d" % step)
            else:
                torch.save(self.model.get_input_embeddings(), 
                    os.path.join(self.out_dir, "soft_embeddings-{}.pt".format(step)))

    def setup_optimizer(self, optimization, num_training_steps, lr, weight_decay, warmup_steps):
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #         {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        #         {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

        # freeze all parameters but soft prefix
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.get_input_embeddings().weight.requires_grad = True

        optimizer_grouped_parameters = [
                {'params': self.model.get_input_embeddings().weight, 'weight_decay': weight_decay}
        ]
        print("fine tune parameters: ", optimizer_grouped_parameters)

        if optimization=="adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters,
                                  lr=lr,
                                  relative_step=False,
                                  warmup_init=False,
                                  weight_decay=weight_decay)
            scheduler = None
        elif optimization.startswith("adamw"):
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=lr,
                              eps=1e-08,
                              weight_decay=weight_decay)
            if self.fp16:
                self.model, optimizer = setup_fp16(self.model, optimizer)
            if optimization=="adamw":
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_training_steps)
            else:
                raise NotImplementedError()
        elif optimization=="8bit-adam":
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters,
                                           lr=lr, betas=(0.9, 0.995))
            if self.fp16:
                self.model, optimizer = setup_fp16(self.model, optimizer)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_training_steps)
        else:
            raise NotImplementedError()

        self.optimizer = optimizer
        self.scheduler = scheduler

    def parallel(self):
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)


    def do_train(self, data, batch_size, num_training_steps, save_period, log_period,
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        dataloader = data.get_dataloader(batch_size, is_training=True)
        n_trainable_params = len([param for param in self.model.parameters() if param.requires_grad])
        n_gpus = torch.cuda.device_count()
        self.logger.info("Training {} parameters on {} examples for {} steps using {} GPUs".format(
            n_trainable_params, len(data), num_training_steps, self.n_gpu))

        global_step = 0
        train_losses = []
        best_accuracy = -1
        stop_training=False

        for epoch in range(num_training_steps):
            print("epoch: ", epoch)
            for batch in dataloader:
                global_step += 1

                input_ids=batch[0].to(self.device)
                attention_mask=batch[1].to(self.device)
                token_type_ids=batch[2].to(self.device)
                if len(batch)==3:
                    labels=None
                else:
                    labels=batch[3].to(self.device)

                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
                loss = loss.mean()

                if torch.isnan(loss).data:
                    print ("Stop training because loss=%s" % (loss.data))
                    stop_training=True
                    break
                train_losses.append(loss.detach().cpu())

                if self.fp16:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.model.get_input_embeddings().weight.grad[:self.orig_vocab_size] = 0

                if global_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()    # We have accumulated enought gradients
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.model.zero_grad()

                if global_step % log_period == 0:
                    self.logger.info("local rank %d\tglobal step %d\ttrain loss %.2f" % (self.local_rank, global_step, np.mean(train_losses)))
                    train_losses = []

                if global_step % save_period == 0:
                    self.save(global_step)

                if global_step==num_training_steps:
                    break

            if global_step==num_training_steps:
                break

        self.logger.info("Finish training")

    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        if verbose:
            dataloader = tqdm(dataloader)
        losses = []
        for batch in dataloader:
            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist() 
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False, return_nll=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        all_nlls = []
        gt_labels = []
        pred_labels = []
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            all_nlls.append(curr_label_losses)
            gt_labels.append(dp["label"])
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            pred_labels.append(prediction_idx)
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        if return_nll:
            return predictions, all_nlls, np.array(gt_labels), np.array(pred_labels)
        else:
            return predictions

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)

def setup_fp16(model, optimizer):
    try:
        import apex
        from apex import amp
        apex.amp.register_half_function(torch, "einsum")
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    fp16_opt_level = "O1"
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer



