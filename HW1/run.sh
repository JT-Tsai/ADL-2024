#!/bin/bash
python3 src/inference_mc.py \
    --test_file "${2}" \
    --context_file "${1}" \
    --model_name_or_path output/chinese-lert-base/mc \
    --tokenizer_name output/chinese-lert-base/mc \
    --pad_to_max_length \
    --max_seq_length 512 \
    --per_device_eval_batch_size 8 \
    --output_dir output/chinese-lert-base/mc                            

python3 src/inference_qa.py  \
    --test_file output/chinese-lert-base/mc/test.json \
    --context_file data/context.json \
    --model_name_or_path output/chinese-lert-large/qa \
    --tokenizer_name output/chinese-lert-large/qa \
    --pad_to_max_length --max_seq_length 512 \
    --per_device_eval_batch_size 8 \
    --result_path "${3}"