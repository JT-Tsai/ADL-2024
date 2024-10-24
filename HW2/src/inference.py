import json
import math
import os
import random
import pandas as pd
from pathlib import Path

import datasets
from datasets import Dataset
import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from parse_args import parse_args
from tw_rouge import get_rouge
from utils import *
import ipdb


def Prepare_work(args, debug = False):
    if args.jsonl_data_file is not None:
        df = pd.read_json(args.jsonl_data_file, lines=True, encoding='utf-8')
        if args.debug:
            df = df.sample(n=300)
        
        eval_dataset = Dataset.from_pandas(df)

        if args.debug:
            print(df.head(1))
    
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        use_fast = True,
        trust_remote_code = args.trust_remote_code,
    )

    if debug:
        print("tokenizer prepared")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf = False,
        config = config,
        trust_remote_code = args.trust_remote_code,
    )

    if debug:
        print("model prepared")
        ipdb.set_trace()


    prefix = "summarize: "
    column_names = eval_dataset.column_names

    dataset_columns = ["maintext", "title"]
    text_column = args.text_column if args.text_column is not None else dataset_columns[0]
    summary_column = args.summary_column if args.summary_column is not None else dataset_columns[1]

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_eval_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]

        targets = examples[summary_column]
        
        index = examples["id"]
        index = [int(id) for id in index]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length = max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["id"] = index
        return model_inputs
    
    eval_dataset = eval_dataset.map(
        preprocess_eval_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset"
    )

    if debug:
        print("inference dataset prepared")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    for index in random.sample(range(len(eval_dataset)), 1):
        print(eval_dataset[index])
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    if debug:
        print("eval_dataloader prepared")

    return model, tokenizer, eval_dataloader

def inference(args, model, tokenizer, eval_dataloader, flag = False):
    ids_set = []
    pred_set = []
    ref_set = []

    DEVICE = "cuda"

    model.to(DEVICE)
    model.eval()

    gen_kwargs = {
        "max_length"    : args.val_max_target_length, # this option will pad to length we want to generate
                                                        # the rest of will pad to tokenizer.pad_token_id
        "num_beams"     : args.num_beams,
        "do_sample"     : args.do_sample,
        "top_k"         : args.top_k,
        "top_p"         : args.top_p,
        "temperature"   : args.temperature,
    }

    for batch in tqdm(eval_dataloader):
        for key in batch:
            batch[key] = batch[key].to(DEVICE)

        # ipdb.set_trace()
        
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask = batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = generated_tokens.cpu().numpy()

            labels = batch["labels"].cpu().numpy()

            # ipdb.set_trace()

            if args.ignore_pad_token_for_loss:
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            
            index = batch["id"].cpu().numpy()
            for id, pred, ref in zip(index, decoded_preds, decoded_labels):
                ids_set.append(str(id))
                pred_set.append(pred)
                ref_set.append(ref)
    if flag:
        # record each inference rouge metrics during training
        # print(ref_set, pred_set)
        try:
            rouge_score = get_rouge(ref_set, pred_set)
        except:
            rouge_score = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        # ipdb.set_trace()
        print(f"succeed inference. rouge value will be recorded.")
        return rouge_score
    else:
        write_jsonl_file(args.output_file, ids_set, pred_set)
        print(f"succeed inference. jsonl file write down in {args.output_file}")

        
if __name__ == "__main__":
    # do all thing
    args = parse_args()
    element = Prepare_work(args, debug = args.debug)
    inference(args, *element)
