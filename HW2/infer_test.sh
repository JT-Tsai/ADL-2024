#!bin/bash

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 128 \
    --num_beams 4 \
    --do_sample \
    --top_p 0.95 \
    --output_file result/tmp.jsonl \
    # --debug \
 