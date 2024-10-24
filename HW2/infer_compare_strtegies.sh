#!bin/bash

# 1
python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 3 \
    --do_sample \
    --top_k 50 \
    --top_p 0.7 \
    --temperature 1 \
    --output_file result/1.jsonl \
    # --debug \

#2
python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 3 \
    --do_sample \
    --top_k 50 \
    --top_p 0.7 \
    --temperature 1.2 \
    --output_file result/2.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 3 \
    --do_sample \
    --top_k 50 \
    --top_p 0.95 \
    --temperature 1 \
    --output_file result/3.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 3 \
    --do_sample \
    --top_k 50 \
    --top_p 0.95 \
    --temperature 1.2 \
    --output_file result/4.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 3 \
    --do_sample \
    --top_k 100 \
    --top_p 0.7 \
    --temperature 1 \
    --output_file result/5.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 3 \
    --do_sample \
    --top_k 100 \
    --top_p 0.7 \
    --temperature 1.2 \
    --output_file result/6.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 3 \
    --do_sample \
    --top_k 100 \
    --top_p 0.95 \
    --temperature 1 \
    --output_file result/7.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 3 \
    --do_sample \
    --top_k 100 \
    --top_p 0.95 \
    --temperature 1.2 \
    --output_file result/8.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 5 \
    --do_sample \
    --top_k 50 \
    --top_p 0.7 \
    --temperature 1 \
    --output_file result/9.jsonl \
    # --debug \

#2
python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 5 \
    --do_sample \
    --top_k 50 \
    --top_p 0.7 \
    --temperature 1.2 \
    --output_file result/10.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 5 \
    --do_sample \
    --top_k 50 \
    --top_p 0.95 \
    --temperature 1 \
    --output_file result/11.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 5 \
    --do_sample \
    --top_k 50 \
    --top_p 0.95 \
    --temperature 1.2 \
    --output_file result/12.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 5 \
    --do_sample \
    --top_k 100 \
    --top_p 0.7 \
    --temperature 1 \
    --output_file result/13.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 5 \
    --do_sample \
    --top_k 100 \
    --top_p 0.7 \
    --temperature 1.2 \
    --output_file result/14.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 5 \
    --do_sample \
    --top_k 100 \
    --top_p 0.95 \
    --temperature 1 \
    --output_file result/15.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 5 \
    --do_sample \
    --top_k 100 \
    --top_p 0.95 \
    --temperature 1.2 \
    --output_file result/16.jsonl \
    # --debug \

python src/inference.py \
    --jsonl_data_file data/data/public.jsonl \
    --model_name_or_path output/first_train \
    --pad_to_max_length \
    --per_device_eval_batch_size 32 \
    --val_max_target_length 128 \
    --num_beams 1 \
    --output_file result/17.jsonl \
    # --debug \
