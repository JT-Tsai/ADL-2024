#!/bin/bash
python multi_choice.py \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --max_seq_length 512 \
    --padding_to_max_length \
    --model_name_or_type bert-base-chinese \
    --tokenizer_name bert-base-chinese \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 4 \
    --output_dir output/bert_base_chinese \
    --context_file data/context.json \
    --with_tracking \
    --debug \