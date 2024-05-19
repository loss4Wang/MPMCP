#!/bin/bash

python_exec="/local/xiaowang/anaconda3/envs/LJP/bin/python"
script_path='/local/xiaowang/LJP Task/train/bert_roberta_lawformer/run_LJP_baseline.py'

export CUDA_VISIBLE_DEVICES="4"
export WANDB_PROJECT="LJP_baselines_task2"
export WANDB_USERNAME="loss4wang"
export WANDB_API_KEY="81a3c9c3fa59f92cbf80af38f03905c8ea01a3fc"
export WANDB_NAME="bert_data4_t2_bs32_lr7e-5" #! need change every run

# hfl/chinese-roberta-wwm-ext, bert-base-chinese, thunlp/Lawformer
# label: t1: relevant_article; t2: defendant_accusation; t3: imprisonment
# text: t1: fact; t2: defendant,fact ; t3: defendant,fact
# is_single_classification: T: t2 data1 & t2 data3
# max_position_embeddings:4098 default for Lawformer
$python_exec "$script_path" \
    --model_name_or_path "bert-base-chinese" \
    --trust_remote_code "True" \
    --train_file "/local/xiaowang/LJP Task/Data/Datasets/train_data4.csv" \
    --validation_file "/local/xiaowang/LJP Task/Data/Datasets/val_data4.csv" \
    --test_file "/local/xiaowang/LJP Task/Data/Datasets/test_data4.csv" \
    --is_single_classification "False"\
    --label_column_name "defendant_accusation" \
    --text_column_names "defendant,fact" \
    --text_column_delimiter "[SEP]" \
    --max_seq_length "512" \
    --pad_to_max_length "False" \
    --shuffle_train_dataset "False" \
    --seed "42" \
    --per_device_train_batch_size "32" \
    --optim "adamw_torch" \
    --evaluation_strategy "epoch" \
    --early_stopping_patience "3" \
    --save_strategy "epoch" \
    --logging_strategy "epoch" \
    --num_train_epochs "200.0" \
    --learning_rate "7e-5" \
    --load_best_model_at_end "True" \
    --do_train "True" \
    --do_eval "True" \
    --do_predict "True" \
    --output_dir "/local/xiaowang/LJP Task/baseline_output/Task2/Data4/bert_lr7" \
    --overwrite_output_dir "True" \
    --report_to "wandb"

