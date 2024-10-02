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
import pandas as pd

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

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        # tokenizer_examples have "input_ids", "attention_mask", "example_id", "offset_mapping" keys
        return tokenized_examples
    
    predict_examples = raw_datasets["test"]
    predict_dataset = predict_examples.map(
        prepare_validation_feature,
        batched = True,
        num_proc = args.preprocessing_num_workers,
        remove_columns = column_names,
        load_from_cache_file = not args.overwrite_cache,
        desc = "Running tokenizer on prediction dataset",
    )

    for index in random.sample(range(len(predict_dataset)), 3):
        print(f"Sample {index} of the predict dataset set: {predict_dataset[index]}.")

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    
    predict_dataloader = DataLoader(
        predict_dataset_for_model, collate_fn=data_collator, batch_size = args.per_device_eval_batch_size
    )
    
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
        
        return formatted_predictions
    
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    all_start_logits = []
    all_end_logits = []

    for _, batch in enumerate(predict_dataloader):
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        
        all_start_logits.append(start_logits.cpu().numpy())
        all_end_logits.append(end_logits.cpu().numpy())
    
    max_len = max([x.shape[1] for x in all_start_logits])
    
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    del all_start_logits
    del all_end_logits

    output_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(predict_examples, predict_dataset, output_numpy)
    
    # {"id":k, "prediction_text": v}
    prediction_id = [prediction[i]["id"] for i in range(len(prediction))]
    prediction_text = [prediction[i]["prediction_text"] for i in range(len(prediction))]

    df = pd.DataFrame({
        "id": prediction_id,
        "answer": prediction_text,
    })

    inference_path = os.path.join(args.output_dir, "result.csv")
    df.to_csv(inference_path, index=False)

    print("Inference Done")

if __name__ == "__main__":
    main()



