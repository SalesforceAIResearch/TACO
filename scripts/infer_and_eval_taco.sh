#!/bin/bash
## change for each dataset
model=$1
echo $model

format=$2
echo $format

device_id=$3
echo $device_id

max_reply=10
exp_id="1130-release-test"

## activate llava environment
source /export/share/zixianma/miniconda/bin/activate
conda activate /export/agentstudio-family/miniconda3/envs/mmall
source /export/agentstudio-family/zixian/.bashrc

## Iterate over datasets
for ((i=4; i<=$#; i++))
do
    eval dataset=\$$i
    lowercase_dataset=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    echo "Evaluating: $lowercase_dataset"
    python -m taco.run_multimodal_agent --execute --max-reply $max_reply --exp-id $exp_id --model $model  --dataset $dataset --infer-device-id "cuda:${device_id}" --prompt-format $format
    model_dir=$(echo "$model" | tr '/' '-')
    python VLMEvalKit/run_eval_on_preds.py --data $dataset --result-file "prediction/${model_dir}/${lowercase_dataset}/${exp_id}-${format}-max-reply-${max_reply}-seed-42.jsonl" 
done

