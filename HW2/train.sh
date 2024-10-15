#!bin/bash
# for sumarization training

python src/summarization.py \
 --jsonl_data_file data/data/train.jsonl \
 --model_name_or_path google/mt5-small \
 --pad_to_max_length \
 --per_device_train_batch_size 8 \
 --per_device_eval_batch_size 24 \
 --gradient_accumulation_steps 3 \
 --learning_rate 8e-5 \
 --num_train_epochs 20 \
 --num_warmup_steps 500 \
 --output_dir output/first_train \
 --output_file output/first_train/infer.jsonl \
 --checkpoint_steps 500 \
 --with_tracking \
 --debug  \
 --n_test_data 100 \
 --num_beams 5 \
 --do_sample \
 --top_p 0.95
