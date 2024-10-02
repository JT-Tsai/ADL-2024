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
from itertools import chain
from tqdm import tqdm
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
)
from parse_args import parse_args
from utils_for_mc import DataCollatorForMultipleChoice

import ipdb

"""
essential args
--test_file
--context_file
--model_name_or_path
--tokenizer_name
--pad_to_max_length
--max_seq_length
--per_device_eval_batch_size
"""


def main():
    args = parse_args()
    
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    if args.context_file is not None:
        with open(args.context_file, 'r', encoding='utf-8') as file:
            context = json.load(file)

    raw_dataset = load_dataset(extension, data_files=data_files)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast = not args.use_slow_tokenizer, trust_remote_code = args.trust_remote_code
    )
    model = AutoModelForMultipleChoice(
        args.model_name_or_path,
        from_tf = bool(".ckpt" in args.model_name_or_path),
        config = config,
        trust_remote_code = args.trust_remote_code,
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

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
            labels = [0]* len(examples['id'])
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

    processed_dataset = raw_dataset.map(
        preprocess_function, batched = True, remove_columns=raw_dataset["test"].column_names
    )

    test_dataset = processed_dataset["test"]

    # testing
    print(len(test_dataset))
    for index in random.sample(range(len(test_dataset)), 3):
        print(f"Sample {index} of the testing set: {test_dataset[index]}")
    
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=None)
    
    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    # ipdb.set_trace()
    device = torch.device("cuda")
    model.to(device)

    result = None
    model.eval()
    for batch in enumerate(test_dataloader): # batch type is dict
        for k in batch:
            batch[k].to(device) # input_ids, attention_mask, 
        with torch.no_grad():
            outputs = model(**batch)
        # (batch, num_classes)
        predictions = outputs.logits.argmax(dim=-1) # compared in num_classes dimension
        predictions = predictions.cpu()
        result = predictions if result is None else result = torch.cat((result, predictions))

    result_numpy = result.numpy()
    
    with open(args.test_file, 'r', encoding='utf-8') as file:
        result_json = json.load(file)

    assert len(result_json) == result.shape[0]

    for i in range(len(result)):
        print(result_numpy[i])
        result_json[i]["relevant"] = result[i]['paragraphs'][int(result_numpy[i])]

    inference_path = os.path.join(args.output_dir, "test.json")
    with open(inference_path, 'w') as file:
        json.dump(result_json, file, ensure_ascii=False)

    print(f"model ouputs is Saved at {inference_path}, process is done")

if __name__ == "__main__":
    main()