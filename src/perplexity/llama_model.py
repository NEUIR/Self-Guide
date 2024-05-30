import numpy as np
import os
from typing import Optional

import torch
import torch.nn as nn
import transformers

import utils


class LM:
    def get_perplexity_data(self, text) -> Optional[dict]:
        raise NotImplementedError

    @classmethod
    def create_from_config(cls, path):
        raise NotImplementedError



class GPT2LM(LM):

    def __init__(self, model_name, device="cuda:0", context_len=512, max_seq_len=1024, verbose=False):
        self.model_name = model_name
        self.device = torch.device(device)
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        torch.set_grad_enabled(False)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name).eval().to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

    # noinspection DuplicatedCode
    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        # noinspection PyListCreation
        all_logprobs = []
        all_positions = []

        # Remaining windows
        for input_tokens, pred_tokens in rolling_token_windows:
            block_output = self.get_token_logprobs(
                input_tokens=input_tokens,
                pred_tokens=pred_tokens,
            )
            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])

        if not all_logprobs:
            return None

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        assert len(all_logprobs) == len(input_ids)
        return {
            "logprobs": all_logprobs,
            "positions": all_positions,
            "length": len(all_logprobs),
            "utf8_length": len(text.encode('utf-8')),
        }

    def get_token_logprobs(self, input_tokens, pred_tokens):
        input_tokens = torch.tensor([input_tokens]).long().to(self.device)
        pred_tokens = torch.tensor(pred_tokens).long().to(self.device)
        output = self.model(input_tokens, return_dict=True)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        neg_logprobs = loss_fct(
            output.logits[0][-len(pred_tokens):],
            pred_tokens,
        ).detach().cpu().numpy()
        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(input_tokens))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(pred_tokens))
            print("Perplexity:", np.exp(neg_logprobs.mean()))

        positions = np.arange(len(input_tokens[0]) - len(pred_tokens), len(input_tokens[0]))

        return {
            "logprobs": - neg_logprobs,
            "positions": positions,
        }


def create_model():
    model = GPT2LM(model_name="/data/LLaMA-2/hf/Llama-2-13b-hf")
    return model