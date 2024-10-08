#!/bin/bash
# for mc using lert-base
python src/multi_choice.py \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --max_seq_length 512 \
    --pad_to_max_length \
    --model_name_or_path hfl/chinese-lert-base \
    --tokenizer_name hfl/chinese-lert-base \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 2 \
    --output_dir output/chinese-lert-base/mc \
    --context_file data/context.json \
    --with_tracking \
    # --debug \

# for qa using lert-large
python src/question_answering.py \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --max_seq_length 512 \
    --pad_to_max_length \
    --model_name_or_path hfl/chinese-lert-large \
    --tokenizer_name hfl/chinese-lert-large \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 25 \
    --gradient_accumulation_steps 4 \
    --output_dir output/chinese-lert-large/qa \
    --context_file data/context.json \
    --with_tracking \
    # --dynamic_save \
    # --debug \
    # --max_train_samples 100 \
    # --max_eval_samples 100 \


