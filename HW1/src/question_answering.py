#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model for question answering using 🤗 Accelerate.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

# import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils_for_qa import *

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from parse_args import parse_args
# import ipdb


# check_min_version("4.45.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = get_logger(__name__)

def main():
    args = parse_args()
    # ipdb.set_trace()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as argument alsong with your Python/Pytorch versions.
    send_example_telemetry("question_answering", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    # WTF
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute.name
            # Create repo and retrieve repo_id
            api = HfApi()
            # WTF
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), 'w+') as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub:
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
        )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split('.')[-1]
        if args.validation_file is not None:
            data_files['validation'] = args.validation_file
            extension = args.validation_file.split('.')[-1]
        # if args.test_file is not None:
        #     data_files['test'] = args.test_file
        #     extension = args.test_file.split('.')[-1]
        if args.context_file is not None:
            with open(args.context_file, 'r', encoding="utf-8") as file:
                context = json.load(file)
                # ipdb.set_trace()

        raw_datasets = load_dataset(extension, data_files=data_files)
        # ipdb.set_trace()
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # config setting
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code = args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # tokenizer setting
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=True, trust_remote_code = args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True, trust_remote_code = args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # model setting
    if args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf = bool('.ckpt' in args.model_name_or_path),
            config=config,
            trust_remote_code = args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config, trust_remote_code = args.trust_remote_code)


    # Preprocessing the datasets
    # Preprocessing is slightly different for training and evaluation

    column_names = raw_datasets["train"].column_names

    # question_column_name = "question" if column_names else column_names[0]
    # context_column_name = "context" if column_names else column_names[1]
    # answer_column_name = "answer" if column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question)
    # pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing

    def prepare_train_feature(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        questions = [q.lstrip() for q in examples["question"]]
        contexts = [context[id] for id in examples["relevant"]]
        answers = []
        for i, ans in enumerate(examples["answer"]):
            answers.append({
                "text": [ans["text"]],
                "ans_start": [ans["start"]]
            })

        """
        [{"text": [xxx], "ans_start": [xxx]}, {"text": [xxx], "ans_start": [xxx]} ... ]
        """
        # Tokenizer our examples with truncation and maybe padding, but keep the overflow using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        tokenized_examples = tokenizer(
            questions, # questions
            contexts, # contexts
            # P(question | contexts)
            truncation="only_second",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True, # return overflowing id
            return_offsets_mapping=True, 
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give use several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position iin the origin context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token
            input_ids = tokenized_examples["input_ids"][i]
            if tokenizer.cls_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.cls_token_id)
            elif tokenizer.bos_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.bos_token_id)
            else:
                cls_index = 0
            
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text
            sample_index = sample_mapping[i]
            answer = answers[sample_index]
            # If no answers are given, set the cls_index as answer.
            # ipdb.set_trace()
            if len(answer["ans_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answer["ans_start"][0]
                end_char = start_char + len(answer["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                
                # End token index of the current span in the text
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer if out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case)
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if args.debug:
        # We will select sample from whole data if argument is specified
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # Create train feature from dataset
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            prepare_train_feature,
            batched = True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file = not args.overwrite_cache,
            desc = "Running tokenizer on train dataset",
        )
        if args.debug:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(args.max_train_samples))
        
    # Validation preprocessing
    def prepare_validation_features(examples):
        """just need to token embedding, id of original data and mapping table to compare with label"""
        # Some of the question have lots of whitespace on the left, which is not useful and will make the
        # truncation of teh context fail (the tokenizer question will take a lots of space). So we remove that
        # left whitespace
        questions = [q.lstrip() for q in examples["question"]]
        contexts = [context[id] for id in examples["relevant"]]

        # Tokenizer our examples with truncation any maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several feartures if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')

        # For evaluation, we will need to covert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question)
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        # tokenizer_examples have "input_ids", "attention_mask", "example_id", "offset_mapping" keys
        return tokenized_examples
    
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_examples = raw_datasets["validation"]
    if args.debug:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(args.max_eval_samples))

    # Validation Feature Creation
    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched = True,
            num_proc = args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file = not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        # ipdb.set_trace()
    if args.debug:
        # During Feature creation dataset might increase, we will select required samples again
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples =  raw_datasets["test"]
        if args.max_predict_samples is not None:
            # We will select from whole data
            predict_examples = predict_examples.select(range(args.max_predict_samples))
        # Predict Feature Creation
        with accelerator.main_process_first():
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched = True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file = not args.overwrite_cache,
                desc = "Running tokenizer on prediction dataset",
            )
            if args.max_predict_samples is not None:
                # During feature creation dataset samples might increase (seq_len and slide window), we will select required samples again
                predict_examples = predict_dataset.select(range(args.max_predict_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    
    # DataLoader creation:
    if args.pad_to_max_length:
        # If padding was already done at max length, we use the default data collator that will jsut convert everything
        # to tensor
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        # For fp8, we pad to multiple of 16.
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size = args.per_device_train_batch_size
    )

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    # ipdb.set_trace()
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    if args.do_predict:
        predict_datasets_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        # ipdb.set_trace()
        predict_dataloader = DataLoader(
            predict_datasets_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )


    # Post-precessing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Psot-processing: we match the start logits and end logits to answers in the original context
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size = args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
            context=context
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id":k, "prediction_text": v} for k, v in predictions.items()]
        """
            format like:
            predictions = [{'prediction_text': '1999', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
            references = [{'answers': {'answer_start': 97, 'text': '1976'}, 'id': '56e10a3be3433e1400422b22'}]
        """
        answers = [{"answer_start": ex["answer"]["start"], "text": ex["answer"]["text"]} for ex in examples]
        references = [{"id": ex["id"], "answers": answers} for ex in examples]
        
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    
    metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    # Optimizer
    # split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
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

    # scheduler and math around the number of training steps.
    overrode_max_train_step = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_step = True
    
    lr_scheduler = get_scheduler(
        name = args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_step
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the train steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_step:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator status
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The tracker initializes automatically on the main process
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("question_answering", experiment_config)
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path=dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract  `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", ""))
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

    train_loss = []
    valid_loss = []
    exact_match = []

    last_EM = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
            eval_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from checkpoint
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

            # Check if the accelerator have performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_dict(output_dir)
                
            if completed_steps >= args.max_train_steps:
                break

            if args.checkpointing_steps == "epochs":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

            if args.push_to_hub and epoch < args.num_train_epoch - 1:
                accelerator.wait_for_everyone()
                unwarpped_model = accelerator.upwrap_model(model)
                unwarpped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    api.upload_folder(
                        commit_message=f"Training in progress epoch {epoch}",
                        folder_path=args.output_dir,
                        repo_id = repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )

        all_start_logits = []
        all_end_logits = []

        model.eval()

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                # ipdb.set_trace()
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if args.with_tracking:
                    eval_loss += loss.float()

                if not args.pad_to_max_length: # necessary to pad predictions and labels foor begin gathered
                    start_logits = accelerator.pad_across_precesses(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])

        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
        # ipdb.set_trace()
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f"Evaluation metrics: {eval_metric}")

        train_loss.append(total_loss.item() / len(train_dataloader))
        valid_loss.append(eval_loss.item() / len(eval_dataloader))
        exact_match.append(eval_metric['exact_match'])

        if args.dynamic_save:
            if args.output_dir is not None and eval_metric['exact_match'] > last_EM:
                last_EM = eval_metric['exact_match']
                accelerator.wait_for_everyone()
                # ipdb.set_trace()
                # avoid model state_dict not memory contiguous
                for param in model.parameters():
                    param.data = param.data.contiguous()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                print("saved model")
                
            metrics = {"train_loss": train_loss, "valid_loss": valid_loss, "EM": exact_match}
            with open(os.path.join(args.output_dir, "metrics.json"), 'w') as file:
                json.dump(metrics, file)

    # Prediction
    if args.do_predict:
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(predict_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        all_start_logits = []
        all_end_logits = []

        model.eval()

        for step, batch in enumerate(predict_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if not args.pad_to_max_length: # necessary to pad prediction and labels for being gathered
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
                
                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())
        
        max_len = max([x.shape[1] for x in all_start_logits]) # Get the max_length of the tensor
        # concatenate the numpy array
        all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
        all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
        predict_metric = metric.compute(predictions = prediction.predictions, references = prediction.label_ids)
        logger.info(f"Predict metrics: {predict_metric}")
        

    if args.with_tracking:
        log = {
            "squad_v2" if args.version_2_with_negative else "squad": eval_metric,
            "train_loss": total_loss.item() / len(train_dataloader),
            "epoch": epoch,
            "step": completed_steps,
        }
    if args.do_predict:
        log["squad_v2_predict" if args.version_2_with_negative else "squad_predict"] = predict_metric

        accelerator.log(log, step=completed_steps)

    if args.output_dir is not None:
        if not args.dynamic_save:
            accelerator.wait_for_everyone()
            # ipdb.set_trace()
            # avoid model state_dict not memory contiguous
            for param in model.parameters():
                param.data = param.data.contiguous()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            logger.info(json.dumps(eval_metric, indent=4))
            save_prefixed_metrics(eval_metric, args.output_dir)
        metrics = {"train_loss": train_loss, "valid_loss": valid_loss, "EM": exact_match}
        with open(os.path.join(args.output_dir, "metrics.json"), 'w') as file:
            json.dump(metrics, file)
    
    if args.dynamic_save:
        print(f"train_done, best_ex = {last_EM}")
    else:
        print("train_done")

if __name__ == "__main__":
    main()