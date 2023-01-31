import time
import sys
import os
import numpy as np
import torch
import json
import openai
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import GPT2Tokenizer

class GPT3Model(object):

    def __init__(self, model_name, api_key, logger=None):
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception:
            pass
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger


    def prepare_data(self, train_data, test_data, method, batch_size=10, dp_sep="\n", max_length=1024):

        demos = []
        if type(train_data[0])==dict:
            for _ in range(len(test_data)):
                demo = []
                for dp in train_data:
                    assert type(dp)==dict, ("Each example should be a dictionary", dp)
                    assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                    demo.append(dp.copy())
                demos.append(demo)
        elif type(train_data[0])==list:
            assert len(train_data) == len(test_data)
            for _demo in train_data:
                demo = []
                for dp in _demo:
                    assert type(dp)==dict, ("Each example should be a dictionary", dp)
                    assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                    demo.append(dp.copy())
                demos.append(demo)
        else:
            print(train_data)
            exit(1)

        # append demonstrations and separate options
        inputs = []
        outputs = []
        metadata = []
        for dp, demo in zip(test_data, demos):
            prompt = dp["input"]
            options = dp["options"]

            indices = [i for i in range(len(inputs), len(inputs) + len(options))]

            assert dp["output"] in dp["options"]
            for i, op in enumerate(dp["options"]):
                if dp["output"] == op:
                    label = i

            metadata.append({"indices": indices, "options": options, "label": label})

            with open(os.path.join('config', 'causal_direction.json')) as f:
                causal_direction_ = json.load(f)

            with open(os.path.join('config', 'task_type.json')) as f:
                task_type = json.load(f)

            causal_direction = {}
            for k in causal_direction_:
                causal_direction[k] = []
                for t in causal_direction_[k]:
                    causal_direction[k] += task_type[t]

            if method=="direct":
                method = "direct"
            elif method=="channel":
                method = "channel"
            elif method == "causal":
                if dp["task"] in causal_direction["x->y"]:
                    method = "direct"
                elif dp["task"] in causal_direction["y->x"]:
                    method = "channel"
                else:
                    print("No such task in config.")
                    raise NotImplementedError()
            elif method == "anti-causal":
                if dp["task"] in causal_direction["x->y"]:
                    method = "channel"
                elif dp["task"] in causal_direction["y->x"]:
                    method = "direct"
                else:
                    print("No such task in config.")
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            if method=="direct":
                demonstrations = ''.join(["{}{}{}\n\n\n".format(d["input"], 
                    dp_sep, d["output"]) for d in demo])
                inputs += [demonstrations + prompt + dp_sep for option in options]
                outputs += [option for option in options]
            elif method=="channel":
                demonstrations = ''.join(["{}{}{}\n\n\n".format(d["output"], 
                    dp_sep, d["input"]) for d in demo])
                inputs += [demonstrations + option + dp_sep for option in options]
                outputs += [prompt for option in options]
            else:
                raise NotImplementedError()

        # truncate inputs
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            input_ids = self.tokenizer.encode(inp)
            output_ids = self.tokenizer.encode(out)
            if (len(input_ids) + len(output_ids) > max_length):
                if len(output_ids) > len(input_ids):
                    output_ids = output_ids[:max_length-len(input_ids)]
                else:
                    input_ids = input_ids[len(input_ids)+len(output_ids) - max_length:]
                assert len(input_ids)+len(output_ids) == max_length, (len(input_ids), len(output_ids))
            inputs[i] = self.tokenizer.decode(input_ids)

        if self.logger is not None:
            self.logger.info("Checking the first example...")
            self.logger.info(inputs[0] + "" + outputs[0])

        # construct a dataloader
        dataset = zip(inputs, outputs)
        input_chunks = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        output_chunks = [outputs[i : i + batch_size] for i in range(0, len(outputs), batch_size)]
        dataloader = [(input_chunks[i], output_chunks[i]) for i in range(0, len(input_chunks))]

        return dataloader, metadata


    def do_inference(self, dataloader):
        losses = []
        cache = []
        cost = 0
        for inputs, outputs in dataloader:
            data = [inp + out for inp, out in zip(inputs, outputs)]
            response = self.gpt3(data)
            for choice in response["choices"]:
                cost += len(choice["logprobs"]["tokens"]) * 0.00006
            print("current cost = " + str(cost))
            cache.append((data, response))
            # get the beginning of the target from the response (based on tokenization)
            for inp, outp, out in zip(inputs, outputs, response["choices"]):
                assert inp+outp==out["text"]
                i = 0
                while out['logprobs']['text_offset'][i] < len(inp):
                    i += 1
                loss = -sum(out['logprobs']["token_logprobs"][i:])
                losses.append(loss / (len(out['logprobs']['text_offset']) - i))
        return losses, cache

    def do_predict(self, losses, metadata, return_nll=False):
        predictions = []
        all_nlls = []
        gt_labels = []
        pred_labels = []
        for idx, dp in enumerate(metadata):
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


    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        while not received:
            try:
                response = openai.Completion.create(model=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response
