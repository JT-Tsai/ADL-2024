"""
1. import dependent library
2. args_parse setting
3. load testing file including test.json and context (using load_dataset in huggingface library)
4. setting config, AutoTokenizer, AutoMultiChoiceModel
    4.1 need to resize embedding size when tokenizer shape doesn't match model embedding layer shape
    4.2 setting padding mode by args gained from args_parse() 
5. setting preprocess_function
6. transform data to dataset format
7. setting data_collator for dataloader and transfor dataset to dataloader
8. setting device and put into needed model and data to device
9. inference for each epoch
10. load test.json as f and add f[idx]['relenvant'] val by model output result
11. save ouput_json
"""
import json
import math
import os
import random

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_for_qa import *
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)

from parse_args import parse_args

def main():
    args = parse_args()

    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    if args.context_file is not None:
        with open(args.context_file, 'r', encoding='utf-8') as file:
            context = json.load(file)

    raw_datasets = load_dataset(extension, data_files=data_files)


    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    
    tokenizer = AutoTokenizer(
        args.tokenizer_name or args.model_name_or_path,
        use_fast = True,
        trust_remote_code = args.trust_remote_code
    )

    model = AutoModelForQuestionAnswering(
        args.model_name_or_path,
        from_tf = bool('.ckpt' in args.model_name_or_path),
        config = config,
        trust_remote_code = args.trust_remote_code
    )

    column_names = raw_datasets["test"].column_names
        
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def prepare_validation_feature(examples):
        questions = [q.lstrip() for q in examples["question"]]
        contexts = [context[id] for id in examples["relevant"]]

        tokenized_examples = tokenizer(
            questions,
            contexts,
            truncation = "only_second",
            max_length = max_seq_length,
            stride = args.doc_stride,
            return_overflowing_tokens = True,
            return_offset_mapping=True,
            padding = "max_length" if args.pad_to_max_length else False
        )

        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

