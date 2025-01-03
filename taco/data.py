import json
import os
import string
import random
import shutil
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from taco.config import *
from copy import deepcopy

class AgentDataset(Dataset):
    def __init__(self, dataset_name, dataset_file=None, random_seed=42):
        self.dataset_name = dataset_name
        if dataset_file is None and dataset_name in DATASET_TO_PATH:
            self.dataset_file = DATASET_TO_PATH[dataset_name]
        else:
            self.dataset_file = dataset_file
        self.random_seed = random_seed
        self.load_data()

    def load_data(self):
        if self.dataset_file is not None:
            ext = os.path.splitext(self.dataset_file)[-1]
            if ext in [".tsv", ".csv"]:
                examples = pd.read_csv(self.dataset_file, sep= '\t' if ext == ".tsv" else ',') 
            elif ext in [".json", ".jsonl"]:
                examples = pd.read_json(self.dataset_file, lines=(ext == ".jsonl"))
            elif ext == ".xlsx":
                examples = pd.read_excel(self.dataset_file)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            examples = examples.to_dict(orient="records")
        elif self.dataset_name.startswith("mantis"):
            subset_name = self.dataset_name[len("mantis-"):]
            dataset = load_dataset("TIGER-Lab/Mantis-Instruct", subset_name, split="train", cache_dir=CACHE_DIR)
            image_dir = os.path.join(MANTIS_IMAGE_PATH, subset_name)
            examples = []
            for i, item in enumerate(dataset):
                example = dict(item)
                example['example_id'] = i
                image_paths = []
                for i, elem in enumerate(example['images']):
                    image_path = os.path.join(image_dir, elem['path'])
                    image_paths.append(image_path)
                example['images'] = image_paths
                texts = example['conversation']
                if len(texts) % 2 != 0: continue
                total_num_turns = len(texts) // 2
                # sample <= 3 random turns
                sample_size = min(3, total_num_turns)
                random.seed(self.random_seed)
                random_turn_indices = random.sample(range(total_num_turns), sample_size)
                for k in random_turn_indices:
                    convo_pos = k * 2
                    new_example = deepcopy(example)
                    new_example['question'] = texts[convo_pos]['content'].replace("<image>", "").strip()
                    new_example['answer'] = texts[convo_pos+1]['content']
                    new_example['turn_id'] = k
                    new_example['index'] = f"{new_example['example_id']}-{new_example['turn_id']}"
                    examples.append(new_example)
        else:
            dataset = load_dataset("HuggingFaceM4/the_cauldron", self.dataset_name, split="train", cache_dir=CACHE_DIR)
            examples = []
            for i, item in enumerate(dataset):
                example = dict(item)
                example['image_id'] = i
                image_base_path = os.path.join(INPUT_IMAGE_PATH, self.dataset_name.lower(), str(example['image_id']))
                os.makedirs(image_base_path, exist_ok=True)
                new_images = []
                for j, jpg_img in enumerate(example['images']):
                    # save cauldron images to disk
                    path = os.path.join(image_base_path, f"image-{j}.jpg")
                    if not os.path.exists(path):
                        jpg_img = jpg_img.convert("RGB")
                        jpg_img.save(path)
                    new_images.append(path)
                example['images'] = new_images
                texts = example['texts']
                # sample <= 3 random turns
                sample_size = min(3, len(texts))
                random.seed(self.random_seed)
                random_turn_indices = random.sample(range(len(texts)), sample_size)
                for k in random_turn_indices:
                    new_example = deepcopy(example)
                    new_example['question'] = texts[k]['user']
                    new_example['answer'] = texts[k]['assistant']
                    new_example['turn_id'] = k
                    new_example['index'] = f"{new_example['image_id']}-{new_example['turn_id']}"
                    examples.append(new_example)

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def build_mcq_prompt(self, example):
        question = example['question']
        options = {
            cand: example[cand]
            for cand in string.ascii_uppercase
            if cand in example and not pd.isna(example[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = example['hint'] if ('hint' in example and not pd.isna(example['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        return prompt

    def build_free_form_prompt(self, example):
        question = example['question']
        prompt = f'Question: {question}' + '\nAnswer the question using a single word or phrase.'
        return prompt

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        if self.dataset_name in ["MMVP", "MMMU_DEV_VAL", "MMStar", "A-OKVQA", "TaskMeAnything_v1_imageqa_random"]:  # "CV-Bench", "SEEDBench_IMG", "AI2D_TEST",
            # mcq questions with options answers
            example['question'] = self.build_mcq_prompt(example)
            base_path = os.path.join(EVAL_IMAGE_BASE_PATH, self.dataset_name)
            # single-image examples
            example['images'] = [os.path.join(base_path, f"{example['index']}.jpg")]
        elif self.dataset_name == "BLINK":
            example['question'] = self.build_mcq_prompt(example)
            base_path = os.path.join(EVAL_IMAGE_BASE_PATH, self.dataset_name)
            images = []
            # multi-image examples
            for path in eval(example['image_path']):
                full_path = os.path.join(base_path, path)
                if os.path.exists(full_path):
                    images.append(full_path)
                else:
                    full_path = full_path.replace("_1.jpg", ".jpg")
                    if os.path.exists(full_path):
                        images.append(full_path)
            example['images'] = images
            
            
        if 'prediction' in example:
            del example['prediction']
        if 'image' in example:
            del example['image']
        return example

def format_query(images, query):
    if isinstance(images, list):
        prompt = ""
        for i, img in enumerate(images):
            prompt += f"""image-{i}: <img "{img}">\n"""
        prompt += f"{query}\n"
        return prompt
    else:
        raise NotImplementedError(f"Image type {type(images)} is not supported.")
    