#!/bin/bash
set -e
source /export/share/zixianma/miniconda/bin/activate /export/share/zixianma/miniconda/envs/mantis
source /export/agentstudio-family/zixian/.bashrc

model_name=$1
data_file=$2
cd mantis/train/
bash scripts/train_mllava_clip.sh $model_name $data_file
# bash scripts/train_mllava_siglip.sh $model_name $data_file
# bash scripts/train_mllava_siglip_from_instruction_tuned.sh $model_name $data_file