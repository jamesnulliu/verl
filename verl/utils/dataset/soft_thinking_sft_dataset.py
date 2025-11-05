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
        self.thinking_probs_key = config.get("thinking_probs_key", "thinking_probs")
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

        # [TODO] The padding logic should be moved to STPO.tree_search.input_embed_generator

        # ↓ Shape: (think_len, embed_dim)
        original_thinking_embed_tensor = torch.tensor(sample[self.thinking_embed_key], dtype=torch.float)
        # ↓ Shape: (think_len, n_branches)
        original_thinking_tkids_tensor = torch.tensor(sample[self.thinking_tkids_key], dtype=torch.long)
        original_thinking_tkids_tensor = original_thinking_tkids_tensor[:, : self.n_thinking_branches]
        # ↓ Shape: (think_len, n_branches)
        original_thinking_probs_tensor = torch.tensor(sample[self.thinking_probs_key], dtype=torch.float)
        original_thinking_probs_tensor = original_thinking_probs_tensor[:, : self.n_thinking_branches]
        
        think_len = original_thinking_embed_tensor.size(0)
        
        think_start = prompt_len + 1  # After the <think> token
        think_end = think_start + think_len  # Before the </think> token
        thinking_mask = torch.zeros_like(attention_mask)
        # Handle edge case where think_end might be > attention_mask length if response is empty
        # This shouldn't happen if data is well-formed, but good to be safe.
        if think_start < attention_mask.size(0):
            thinking_mask[think_start:min(think_end, attention_mask.size(0))] = 1
        
        # This mask must be applied *before* truncation to select the right data
        pre_trunc_ones_indices = (thinking_mask == 1).nonzero(as_tuple=True)[0]

        sequence_length = input_ids.shape[0]
        
        # Logic to pad/truncate all sequence-length tensors
        # and simultaneously prepare the (potentially truncated) thinking data

        if sequence_length < self.max_length:
            pad_len = self.max_length - sequence_length
            padded_input_ids = (
                torch.ones(size=(pad_len,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(pad_len,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            thinking_mask = torch.cat((thinking_mask, padded_attention_mask)) # This is the final mask
            
            # No truncation, so we use the original thinking data
            data_thinking_embeds = original_thinking_embed_tensor
            data_thinking_tkids = original_thinking_tkids_tensor
            data_thinking_probs = original_thinking_probs_tensor

        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # Find which original thinking tokens survive left truncation
                trunc_start_idx = sequence_length - self.max_length
                surviving_indices_mask = pre_trunc_ones_indices >= trunc_start_idx
                
                # Truncate the sequence-length tensors
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                thinking_mask = thinking_mask[-self.max_length :] # This is the final mask

            elif self.truncation == "right":
                # Find which original thinking tokens survive right truncation
                trunc_end_idx = self.max_length
                surviving_indices_mask = pre_trunc_ones_indices < trunc_end_idx
                
                # Truncate the sequence-length tensors
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                thinking_mask = thinking_mask[: self.max_length] # This is the final mask

            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

            # Select the surviving thinking data
            data_thinking_embeds = original_thinking_embed_tensor[surviving_indices_mask]
            data_thinking_tkids = original_thinking_tkids_tensor[surviving_indices_mask]
            data_thinking_probs = original_thinking_probs_tensor[surviving_indices_mask]
        
        else: # sequence_length == self.max_length
            # No padding or truncation needed
            data_thinking_embeds = original_thinking_embed_tensor
            data_thinking_tkids = original_thinking_tkids_tensor
            data_thinking_probs = original_thinking_probs_tensor
        
        # Initialize with zeros
        padded_thinking_embeds = torch.zeros(
            self.max_length, original_thinking_embed_tensor.size(1),
            dtype=original_thinking_embed_tensor.dtype
        )
        padded_thinking_tkids = torch.zeros(
            self.max_length, original_thinking_tkids_tensor.size(1),
            dtype=original_thinking_tkids_tensor.dtype
        )
        padded_thinking_probs = torch.zeros(
            self.max_length, original_thinking_probs_tensor.size(1),
            dtype=original_thinking_probs_tensor.dtype
        )
        
        # Use the final thinking_mask to scatter the (potentially truncated) data
        if data_thinking_embeds.size(0) > 0:
            # Check that the number of 1s in the final mask matches the data length
            assert torch.sum(thinking_mask) == data_thinking_embeds.size(0), \
                f"Mask sum ({torch.sum(thinking_mask)}) != data len ({data_thinking_embeds.size(0)})"

            padded_thinking_embeds[thinking_mask == 1] = data_thinking_embeds
            padded_thinking_tkids[thinking_mask == 1] = data_thinking_tkids
            padded_thinking_probs[thinking_mask == 1] = data_thinking_probs
            
        # --- MODIFICATION END ---

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
            # ↓ Shape: (seq_len, embed_dim)
            "thinking_embeds": padded_thinking_embeds,
            # ↓ Shape: (seq_len, n_branches)
            "thinking_tkids": padded_thinking_tkids,
            # ↓ Shape: (seq_len, n_branches)
            "thinking_probs": padded_thinking_probs,
            # ↓ Shape: (seq_len,)
            "thinking_mask": thinking_mask,
            # ↓ Shape: (seq_len,)
            "attention_mask": attention_mask,
            # ↓ Shape: (seq_len,)
            "position_ids": position_ids,
            # ↓ Shape: (seq_len,)
            "loss_mask": loss_mask,
        }