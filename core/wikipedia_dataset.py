from dataclasses import dataclass
import os
from typing import Dict, Sequence
from torch.utils.data import Dataset
import datasets
import logging
import torch.distributed as dist
import torch
import transformers
import copy
import math


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess_wikipedia(
    samples: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Preprocess the Wikipedia data by tokenizing."""
    examples = [text + DEFAULT_EOS_TOKEN for text in samples["text"]]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    input_ids = examples_tokenized["input_ids"]

    # Using input_ids as labels for a language modeling task
    labels = copy.deepcopy(input_ids)

    return dict(input_ids=input_ids, labels=labels)



def filter_long_samples_wikipedia(
    samples: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Sequence[bool]:
    """Filter out long samples."""
    examples = samples["text"]
    tokenized = tokenizer(examples, return_length=True, truncation=True)
    return [length < tokenizer.model_max_length for length in tokenized["length"]]


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, tokenizer: transformers.PreTrainedTokenizer, limit=None
    ):
        super(SupervisedDataset, self).__init__()
        workers = math.ceil(os.cpu_count() / dist.get_world_size())
        logging.warning(f"TOKENIZING WITH NUM_WORKERS: {workers}")

        # Loading the Wikipedia dataset
        # dataset = datasets.load_dataset("aghilrs/fawiki20231001", split="train")
        dataset = datasets.load_dataset("aghilrs/journals-translation-text", split="train")

        # Filtering and preprocessing
        dataset = (
            dataset.filter(
                lambda samples: filter_long_samples_wikipedia(samples, tokenizer),
                batched=True,
                batch_size=3000,
                num_proc=workers,
            )
            .map(
                lambda samples: preprocess_wikipedia(samples, tokenizer),
                batched=True,
                batch_size=3000,
                num_proc=workers,
            )
        )

        if limit:
            dataset = dataset.select(range(limit))

        self.input_ids = dataset["input_ids"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=torch.tensor(self.input_ids[i]),
            labels=torch.tensor(self.labels[i]),
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
