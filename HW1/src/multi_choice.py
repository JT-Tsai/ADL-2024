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
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

# import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
# from typing import Optional, Union

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from transformers.utils import PaddingStrategy, check_min_version, send_example_telemetry
from parse_args import parse_args
from utils_for_mc import DataCollatorForMultipleChoice
# import ipdb

logger = get_logger(__name__)

def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("multi_choice", args)

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
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
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
            # Downloading and loading a dataset from the hub
            raw_dataset = load_dataset(
                args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
            )
        else:
            data_files = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
                extension = args.train_file.split(".")[-1]
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
                extension = args.validation_file.split(".")[-1]
            if args.context_file is not None:
                with open(args.context_file, 'r', encoding="utf-8") as file:
                    # ipdb.set_trace()
                    context = json.load(file)

            raw_dataset = load_dataset(extension, data_files=data_files)
            
        # Trim a number of training example
        if args.debug:
            for split in raw_dataset.keys():
                raw_dataset[split] = raw_dataset[split].select(range(100))
            # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
            # https://huggingface.co/docs/datasets/loading_datasets.

        if raw_dataset["train"] is not None:
            column_names = raw_dataset["train"].column_names
        else:
            column_names = raw_dataset["validation"].column_names

        # When using your own dataset or a different dataset from swag, you will pprobably neeed to change this

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
            )
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        
        if args.model_name_or_path:
            model = AutoModelForMultipleChoice.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config = config,
                trust_remote_code = args.trust_remote_code,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForMultipleChoice.from_config(config, trust_remote_code=args.trust_remote_code)
            # magic word    
            for param in model.parameters(): param.data = param.data.contiguous()
        
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        padding = "max_length" if args.pad_to_max_length else False

        """data format modify"""

        def preprocess_function(examples):
            # Questions
            first_sentences = [[question]* 4 for question in examples["question"]]
            # Answers_id
            paragraphs_ids = [[ids for ids in para] for para in examples["paragraphs"]]
            # context
            second_sentences = [
                [context[id] for id in ids] for ids in paragraphs_ids
            ]
            # the index of correct answer
            labels = [paragraphs_ids[i].index(examples['relevant'][i]) for i in range(len(examples["relevant"]))]
            # ipdb.set_trace()

            # Flatten out
            first_sentences = list(chain(*first_sentences))
            second_sentences = list(chain(*second_sentences))
            # ipdb.set_trace()

            # Tokenize
            tokenized_example = tokenizer(
                first_sentences, # [[question, question, question, question], ...]
                second_sentences, # [[answer_1, answer_2, answer_3, answer_4], ...]
                max_length=args.max_seq_length,
                padding=padding,
                truncation=True,
            )
            # ipdb.set_trace()

            # Un-flatten
            # k -> input_ids, attention_mask
            tokenized_input = {k: [v[i: i+4] for i in range(0, len(v), 4)] for k, v in tokenized_example.items()}
            tokenized_input["label"] = labels
            # ipdb.set_trace()
            
            return tokenized_input
        
        with accelerator.main_process_first():
            processed_dataset = raw_dataset.map(
                preprocess_function, batched=True, remove_columns=raw_dataset["train"].column_names
            )

        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["validation"]

        # Log a few random samples from the training set
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        
        # ipdb.set_trace()

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
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
            data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

        # from list data to pt data and package data to batch data
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)

        # Optimizer
        # Split weights in two group, onw with weight decay and ther other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any (nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any (nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Use the device given by the `accelerator` object
        device = accelerator.device
        model.to(device)
        
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps* accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # We need to recalculate our total trainning steps as the size of the trainning dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many step we shold save the Accelerator states
        checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        # We need to initialize the trackers we use, and also store our configuration
        # The trackers intitializes automatically on the main process
        if args.with_tracking:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("multi_choice", experiment_config)

        # Metrics
        metric = evaluate.load("accuracy")

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weight and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)
            
            accelerator.print(f"Resume from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            # Extract `epoch{i} or step{i}`
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


        train_loss = []
        valid_loss = []
        acc = []

        # model training
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
                eval_loss = 0
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming fron a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            for _ , batch in enumerate(active_dataloader):
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

                # Check if the accelerator has performed an optimization-step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            for _, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                # ipdb.set_trace()
                eval_loss += outputs.loss.float()
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                # ipdb.set_trace()
                metric.add_batch(
                    predictions = predictions,
                    references = references
                )

            eval_metric = metric.compute()
            accelerator.print(f"epoch {epoch}: {eval_metric}")

            t_loss = total_loss.item() / len(train_dataloader)
            e_loss = eval_loss.item() / len(eval_dataloader)

            if args.with_tracking:
                accelerator.log(
                    {
                        "accuracy": eval_metric,
                        "train_loss": t_loss,
                        "eval_loss": e_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step = completed_steps,
                )
            
            print("train_loss: ", t_loss)
            print("eval_loss: ", e_loss)
            train_loss.append(t_loss)
            valid_loss.append(e_loss)
            acc.append(eval_metric['accuracy'])

            if args.push_to_hub and epoch < args.num_train_epoch - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process = accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    api.upload_folder(
                        commit_message=f"Training in progres epoch {epoch}",
                        folder_path=args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )
            
            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        if args.with_tracking:
            accelerator.end_training()

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            for param in model.parameters():
                param.data = param.data.contiguous()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_fucntion=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    api.upload_folder(
                        commit_message="End of training",
                        folder_path=args.output_dir,
                        repo_id = repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )
                all_result = {f"eval_{k}": v for k, v in eval_metric.items()}
                with open(os.path.join(args.output_dir, "all_result.json"), "w") as f:
                    json.dump(all_result, f)

                metrics = {"train_loss": train_loss, "valid_loss": valid_loss, "acc": acc}
                with open(os.path.join(args.output_dir, "metrics.json"), "w") as file:
                    json.dump(metrics, file)

if __name__ == "__main__":
    main()