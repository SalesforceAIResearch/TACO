#!/bin/bash
set -e
source /export/agentstudio-family/zixian/.bashrc
model_name=$1
data_file=$2
llm_lr=1e-5
epoch=1
max_seq_len=8192

## activate llava environment
source /export/share/zixianma/miniconda/bin/activate /export/share/zixianma/miniconda/envs/llava-next
bash llava/scripts/train/finetune_onevision_from_mid_stage_tune_vision.sh $model_name $data_file $llm_lr $epoch $max_seq_len
# bash llava/scripts/train/finetune_onevision_from_mid_stage.sh $model_name $data_file $llm_lr $epoch $max_seq_len
# bash llava/scripts/train/finetune_onevision_from_pretrained.sh $model_name $data_file $llm_lr $epoch $max_seq_len