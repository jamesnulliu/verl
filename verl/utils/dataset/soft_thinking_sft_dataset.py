# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import torch
from datasets import load_from_disk as hf_load_from_disk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.model import compute_position_id_with_mask


class SoftThinkingSFTDataset(Dataset):
    """
    This is an in-memory SoftThinkingSFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, sft_dataset_path: str, tokenizer, config, max_samples: int = -1):
        self.prompt_key = config.get("prompt_key", "question")
        self.all_branch_response_tkids_key = config.get("all_branch_response_tkids_key", "all_branch_response_tkids")
        self.n_thinking_branches = config.get("n_thinking_branches", 10)
        self.thinking_embed_key = config.get("thinking_embed_key", "thinking_input_embed")
        self.thinking_tkids_key = config.get("thinking_tkids_key", "thinking_tkids")
        self.thinking_probs_key = config.get("thinking_probs_key", "thinking_token_probs")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {"enable_thinking": True})
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.hf_dataset = hf_load_from_disk(sft_dataset_path)
        print(f"Loaded hf_dataset from {sft_dataset_path}, len={len(self.hf_dataset)}")
        self.max_length: int = config.get("max_length", 10000)
        self.truncation = config.get("truncation", "error")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        tokenizer = self.tokenizer

        sample = self.hf_dataset[idx]
        prompt: str = sample[self.prompt_key]

        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        assert isinstance(prompt_chat_str, str)
        prompt_tkids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_tkids = prompt_tkids_output["input_ids"][0]
        prompt_attention_mask = prompt_tkids_output["attention_mask"][0]

        # ↓ Shape: (n_branches, resp_len*)
        all_branch_response_tkids: list[list[int]] = sample[self.all_branch_response_tkids_key]
        selected_branch_idx = 0  # for simplicity, we select the first branch
        response_tkids = torch.tensor(all_branch_response_tkids[selected_branch_idx], dtype=torch.long)
        response_attention_mask = torch.ones_like(response_tkids, dtype=torch.long)

        prompt_len = prompt_tkids.size(0)
        response_len = response_tkids.size(0)

        input_ids = torch.cat((prompt_tkids, response_tkids), dim=-1)
        # ↓ Shape: (seq_len,)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # ↓ Shape: (think_len, embed_dim)
        thinking_embed_tensor = torch.tensor(sample[self.thinking_embed_key], dtype=torch.float)
        # ↓ Shape: (think_len, n_branches)
        thinking_tkids_tensor = torch.tensor(sample[self.thinking_tkids_key], dtype=torch.long)
        thinking_tkids_tensor = thinking_tkids_tensor[:, : self.n_thinking_branches]
        # ↓ Shape: (think_len, n_branches)
        thinking_probs_tensor = torch.tensor(sample[self.thinking_probs_key], dtype=torch.float)
        thinking_probs_tensor = thinking_probs_tensor[:, : self.n_thinking_branches]
        think_len = thinking_embed_tensor.size(0)
        # `thinking_embed` only contains the thinking part [t0, t1, ..., tm],
        # so we need to create a mask for inference:
        # [p0, p1, ..., pn, <think>, t0, t1, ..., tm, </think>, r0, r1, ..., rk]
        think_start = prompt_len + 1  # After the <think> token
        think_end = think_start + think_len  # Before the </think> token
        thinking_mask = torch.zeros_like(attention_mask)
        thinking_mask[think_start:think_end] = 1

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            thinking_mask = torch.cat((thinking_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                thinking_mask = thinking_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                thinking_mask = thinking_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_len > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_len, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_len + response_len, loss_mask.size(0)) - 1] = 0

        return {
            # ↓ Shape: (seq_len,)
            "input_tkids": input_ids,
            # ↓ Shape: (think_len, embed_dim)
            "thinking_embeds": thinking_embed_tensor,
            # ↓ Shape: (think_len, n_branches)
            "thinking_tkids": thinking_tkids_tensor,
            # ↓ Shape: (think_len, n_branches)
            "thinking_probs": thinking_probs_tensor,
            # ↓ Shape: (seq_len,)
            "thinking_mask": thinking_mask,
            # ↓ Shape: (seq_len,)
            "attention_mask": attention_mask,
            # ↓ Shape: (seq_len,)
            "position_ids": position_ids,
            # ↓ Shape: (seq_len,)
            "loss_mask": loss_mask,
        }
