# 🌮 TACO: Learning Multi-modal Action Models with Synthetic Chains-of-Thought-and-Action

<h3 align="left"> <a href="https://taco-project.github.io/">🌐 Website</a> | <a href="https://arxiv.org/pdf/2412.05479">📑 Arxiv</a> | <a href="https://huggingface.co/collections/Salesforce/taco-models-6764b2ad9ed8cf7fc0946581">🤗 Model Weights</a> | <a href="">💻 Demo</a></h3>
    
<h5 align="left"> If you like our project or are interested in its updates, please star us on GitHub :) Thank you! ⭐ </h2>

## News
 
 🔥 [2024-12-09]: Code released!

## What is TACO?
We introduce TACO as a family of multi-modal large action models designed to improve performance on such complex, multi-step and multi-modal tasks. During inference, TACO produces chains-of-thought-and–action (CoTA), executes intermediate steps by invoking external tools such as OCR, depth estimation and calculator, then integrates both the thoughts and action outputs to produce coherent responses. Our TACO models outperform the instruction-tuned baseline across 8 benchmarks, achieving a 3.6% improvement on average, with gains up to 15% in MMVet tasks involving OCR, mathematical reasoning and spatial reasoning. 

<p align="center">
  <img src="image/teaser.png" width="1000" style="margin-bottom: 0.2;"/>
  <p align="center">Figure 1. TACO vs. other multi-modal models</p>
<p>

## Code usage
### Installation
You can easily download the repo and set up the environment via:
```
git clone https://github.com/airesearch-emu/taco.git
cd taco

pip install -r requirements.txt
```
Note that this ```requirements.txt``` is mainly for running inference and eval with taco. For training taco, see the [Training](#training) section for additional requirements.
### Inference and Eval
#### Inference only
Run the python command below
```
python -m taco.run_multimodal_agent --execute --max-reply $max_reply --exp-id $exp_id --model $model  --dataset $dataset --infer-device-id "cuda:${device_id}" --prompt-format $format
```
For example,
```
python -m taco.run_multimodal_agent --execute --max-reply 10 --exp-id test --model gpt-4o-2024-08-06 --dataset MMVet --infer-device-id cuda:0 --prompt-format cota
```

#### Evalation only
Run the python command below:
```
python VLMEvalKit/run_eval_on_preds.py --data $dataset --result-file $prediction_file 
```
For example,
```
python VLMEvalKit/run_eval_on_preds.py --data MMVet --result-file prediction/gpt-4o-2024-08-06/mmvet/test-cota-max-reply-10-seed-42.jsonl
```
#### Infer and eval
Run the bash script below:
```
bash scripts/infer_and_eval_taco.sh $model $prompt_format $cuda_id $dataset1 $dataset2 ...
```
For example,
```
bash scripts/infer_and_eval_taco.sh gpt-4o-2024-08-06 cota 0 MMVP MMVet RealWorldQA MMStar 
```
### Training
Note that we recommend setting up separate environments for training Mantis and LLaVA-OneVision models, as they share some common packages but require different versions. 

#### Mantis
1. Create a new environment and install required packages
```
pip install -r requirements_mantis.txt
```
2. Prepare training data 
   - a. Download Mantis training data to ```train_data/```
   - b. Download images to to ```image_dir```
   - c. Update ```path``` and ```image_dir``` in data config file
3. Make necessary changes (e.g. conda environment and .bashrc) in the script ```bash scripts/train_taco_mantis_llava.sh```
4. Run the bash script below:
```
bash scripts/train_taco_mantis_llava.sh $model_name $data_config_file
```
#### LLaVA-OneVision
1. Create a new environment and install required packages
```
pip install -r requirements_llava.txt
```
2. Prepare training data
   - a. Download the training data to ```trian_data```
3. Make necessary changes (e.g. conda environment and .bashrc) in the script ```scripts/train_taco_llava_onevision.sh```
4. Run the bash script below:
```
bash scripts/train_taco_llava_onevision.sh $model_name $data_json_file
```

### Notice
This release is for research purposes only in support of an academic paper. This repository is licensed under the noncommercial license [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 
Some of our TACO models were built with Meta Llama 3, which is licensed under the [Meta Llama 3 Community License](https://www.llama.com/llama3/license/), Copyright © Meta Platforms, Inc. All Rights Reserved.

### Citation
Please cite us if you find our repository helpful. Thank you!
```
@misc{ma2024tacolearningmultimodalaction,
      title={TACO: Learning Multi-modal Action Models with Synthetic Chains-of-Thought-and-Action}, 
      author={Zixian Ma and Jianguo Zhang and Zhiwei Liu and Jieyu Zhang and Juntao Tan and Manli Shu and Juan Carlos Niebles and Shelby Heinecke and Huan Wang and Caiming Xiong and Ranjay Krishna and Silvio Savarese},
      year={2024},
      eprint={2412.05479},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05479}, 
}
```
