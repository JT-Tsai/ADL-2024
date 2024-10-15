#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import json
import logging
import math
import os
import random
import pandas as pd
from pathlib import Path

import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
# from transformers.utils.versions import require_version

from parse_args import parse_args
from utils import *
from inference import inference
from tw_rouge import get_rouge
# from eval import eval
# import ipdb

logger = get_logger(__name__)

def main():
    args = parse_args()
    # ipdb.set_trace()
    # Sending telemetry. Tracking the example usage helps us better allocate resourses to maintain them. The
    # information sent is the one passed as arguments along with your Python/Pytorch versions.
    send_example_telemetry("summarization", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps = args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # run t5 model need to give a prefix to specify task to do (summarize: )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only = False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    """This section load dataset from pandas DataFrames and split training and testing set"""
    if args.jsonl_data_file is not None:
        df = pd.read_json(args.jsonl_data_file, lines = True, encoding = 'utf-8')
        if args.debug:
            df = df.sample(n=args.n_test_data)
        raw_datasets = Dataset.from_pandas(df)
        if args.debug:
            print(df.head(1))
        raw_datasets = raw_datasets.train_test_split(args.split_rate)
        raw_datasets["validation"] = raw_datasets.pop("test")
        # ipdb.set_trace()

    # Load pretrained model and tokenizer
    # download model and vocab
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code = args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code = args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast = not args.use_slow_tokenizer, trust_remote_code = args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast = not args.use_slow_tokenizer, trust_remote_code = args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf = bool(".ckpt" in args.model_name_or_path),
            config = config,
            trust_remote_code = args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code = args.trust_remote_code)

    # resize the embeddings only when necessary to avoid index errors.
    # on a small vocab and want a smaller embedding size, remove this test
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # The decoder_start_token_id is a special token ID that marks the beginning of the decoder input.
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correcctly defined")
    
    # for t5
    prefix = "summarize: "

    # Preprocessing the datasets
    # First we tokenizer all the texts

    column_names = raw_datasets["train"].column_names
    # Get the column names for input/target.

    dataset_columns = ["maintext", "title"]
    text_column = args.text_column if args.text_column is not None else dataset_columns[0]
    summary_column = args.summary_column if args.summary_column is not None else dataset_columns[1]

    # WTF
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # Temporarily set max_target_length for training
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_train_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]

        targets = examples[summary_column]

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
        return model_inputs
    
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
    
    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_train_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )

        # Temprorarily set max_target_length for validation
        max_target_length = args.val_max_target_length
        eval_dataset = raw_datasets["validation"].map(
            preprocess_eval_function,
            batched = True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        # ipdb.set_trace()
        logger.info(train_dataset)
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(eval_dataset)
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of
    )
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn = data_collator, batch_size = args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn = data_collator, batch_size = args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, onw with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math aroud the number of training steps.
    overrode_max_train_step = False
    num_update_steps_per_epoch =  math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_step = True
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer = optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps = args.max_train_steps
        if overrode_max_train_step
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_step:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we reclaculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    
    # We need to intialize the tracker we use, and also store oure configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard can't log Enums, need to raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization", experiment_config)
    
    # ipdb.set_trace()

    # """---------------------------------------modify_line------------------------------------------------"""

    # Metric
    # metric = evaluate.load("rouge")

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f" Num examples = {len(train_dataset)}")
    logger.info(f" Num Epochs = {args.num_train_epochs}")
    logger.info(f" Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f" Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f" Total optimization steps = {args.max_train_steps}")

    # only shoe the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
            
        # Extract  `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)


    T_LOSS = []
    ROUGE = [[]] * 3 # rouge-1 rouge-2 rouge-L
    result = {}

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # we skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Check if the accelerator has performed an optimization step behind the 
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                if output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        T_LOSS.append(total_loss.item()/ len(active_dataloader))

        model.eval()

        # this section using tw_rouge validate model performance
        score = inference(args, model, tokenizer, eval_dataloader, flag=True)
        # ipdb.set_trace()
        ROUGE[0].append(score['rouge-1']['f'])
        ROUGE[1].append(score['rouge-2']['f'])
        ROUGE[2].append(score['rouge-L']['f'])

        print(ROUGE[0], ROUGE[1], ROUGE[2])
        """--------------------------modify_line---------------------------------"""
        # 1. record loss and rouge metrics
        # 2. plot visualize loss and rouge
        
        logger.info(result)

        if args.with_tracking:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step = completed_steps)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            # save model
            for param in model.parameters():
                param.data = param.data.contiguous()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process = accelerator.is_main_process, save_function = accelerator.save
            )
            # save tokenizer   
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

                """modify mapping result dict to rouge metrics"""
                all_results = {f"eval_{k}": v for k, v in result.items()}
                with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)

if __name__ == "__main__":
    main()