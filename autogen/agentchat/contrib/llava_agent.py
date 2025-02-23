import json
import logging
import os
from typing import List, Optional, Tuple
from PIL import Image

import replicate
import requests
import torch
import transformers
import re
import math

from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.img_utils import get_image_data, llava_formatter
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.code_utils import content_str
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List


from ...formatting_utils import colored

logger = logging.getLogger(__name__)

# we will override the following variables later.
SEP = "###"

DEFAULT_LLAVA_SYS_MSG = "You are an AI agent and you can view images."

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

class LLaVAAgent(MultimodalConversableAgent):
    def __init__(
        self,
        name: str,
        system_message: Optional[Tuple[str, List]] = DEFAULT_LLAVA_SYS_MSG,
        *args,
        **kwargs,
    ):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the ChatCompletion inference.
                Please override this attribute if you want to reprogram the agent.
            **kwargs (dict): Please refer to other kwargs in
                [ConversableAgent](../conversable_agent#__init__).
        """
        super().__init__(
            name,
            system_message=system_message,
            *args,
            **kwargs,
        )

        assert self.llm_config is not None, "llm_config must be provided."
        self.register_reply([Agent, None], reply_func=LLaVAAgent._image_reply, position=2)
        self.device = self.llm_config['device_id'] if 'device_id' in self.llm_config else "cuda"
        config = self.llm_config['config_list'][0]
        
        attn_implementation = None # or "flash_attention_2"

        if config["base_url"].find("0.0.0.0") >= 0 or config["base_url"].find("localhost") >= 0:
            self.llava_mode = "local"
        else:
            self.llava_mode = config["llava_mode"]

        if self.llava_mode == "mantis-hf":
            model_path = config['model_path']
            if os.path.exists(model_path):
                self.processor = MLlavaProcessor.from_pretrained(model_path)
                self.model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map=self.device, torch_dtype=torch.float16, attn_implementation=attn_implementation)
            else:
                self.processor = MLlavaProcessor.from_pretrained(config['model'])
                self.model = LlavaForConditionalGeneration.from_pretrained(config['model'], torch_dtype=torch.float16, attn_implementation=attn_implementation)
            
                self.processor.save_pretrained(model_path)
                self.model.save_pretrained(model_path)
            
            self.model.to(self.device)
            self.processor(self.device)
        elif self.llava_mode == "llava-ov-hf":
            model_name = config['model']
            model_path = config['model_path']
            if os.path.exists(model_path):
                # local model path
                self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map=self.device)
            else:
                # Huggingface model path
                self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_name.replace("_qwen", ""), None, model_name, device_map=self.device)
                self.tokenizer.save_pretrained(model_path)
                self.model.save_pretrained(model_path)
            self.model.to(self.device)
        
    def _image_reply(self, messages=None, sender=None, config=None):
        # Note: we did not use "llm_config" yet.

        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        # The formats for LLaVA and GPT are different. So, we manually handle them here.
        images = []
        prompt = content_str(self.system_message) + "\n"
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            images += [d["image_url"]["url"] for d in msg["content"] if d["type"] == "image_url"]
            content_prompt = content_str(msg["content"])
            prompt += f"{SEP}{role}: {content_prompt}\n"
        prompt += "\n" + SEP + "Assistant: "

        # PIL to base64
        if self.llava_mode == "local":
            images = [get_image_data(im) for im in images]

        out = ""
        retry = 10
        while len(out) == 0 and retry > 0:
            # image names will be inferred automatically from llava_call
            out = self.llava_call_binary(
                prompt=prompt,
                images=images,
                config_list=self.llm_config["config_list"],
                temperature=self.llm_config.get("temperature", 0.5),
                max_new_tokens=self.llm_config.get("max_new_tokens", 2000),
                do_sample= self.llm_config.get("do_sample", True)
            )
            retry -= 1

        assert out != "", "Empty response from LLaVA."

        return True, out


    def _llava_call_binary_with_config(
        self, prompt: str, images: list, config: dict, max_new_tokens: int = 1000, temperature: float = 0.5, seed: int = 1, do_sample: bool = False
    ):
        if self.llava_mode == "local":
            headers = {"User-Agent": "LLaVA Client"}
            pload = {
                "model": config["model"],
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "stop": SEP,
                "images": images,
            }

            response = requests.post(
                config["base_url"].rstrip("/") + "/worker_generate_stream", headers=headers, json=pload, stream=False
            )

            for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))
                    output = data["text"].split(SEP)[-1]
        elif self.llava_mode == "remote":
            # The Replicate version of the model only support 1 image for now.
            img = "data:image/jpeg;base64," + images[0]
            response = replicate.run(
                config["base_url"], input={"image": img, "prompt": prompt.replace("<image>", " "), "seed": seed}
            )
            # The yorickvp/llava-13b model can stream output as it's running.
            # The predict method returns an iterator, and you can iterate over that output.
            output = ""
            for item in response:
                # https://replicate.com/yorickvp/llava-13b/versions/2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591/api#output-schema
                output += item
        elif self.llava_mode == "mantis-hf":
            generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    # "num_beams": 1,
                    "do_sample": False
                }
            response, history = chat_mllava(prompt, images, self.model, self.processor, **generation_kwargs)
            output = response
        elif self.llava_mode == "llava-ov-hf":
            conv_mode = "qwen_1_5"

            conv = conv_templates[conv_mode].copy()
            input_ids = preprocess_qwen([{'from': 'human','value': prompt},{'from': 'gpt','value': None}], self.tokenizer, has_image=True).cuda()
        
            image_tensors = []
            for image in images:
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor.half().to(torch.device(self.device)))

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True if temperature > 0 else False,
                    temperature=None if temperature == 0 else temperature,
                    top_p=None, #top_p,
                    num_beams=1,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=max_new_tokens,
                    use_cache=True)

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
        # Remove the prompt and the space.
        output = output.replace(prompt, "").strip().rstrip()
        
        start_idx = output.rfind("Assistant:")
        if start_idx != -1:
            output = output[start_idx + len("Assistant:"):].strip().rstrip()
        return output


    def llava_call_binary(
        self, prompt: str, images: list, config_list: list, max_new_tokens: int = 1000, temperature: float = 0.5, seed: int = 1, do_sample: bool = False
    ):
        # TODO 1: add caching around the LLaVA call to save compute and cost
        # TODO 2: add `seed` to ensure reproducibility. The seed is not working now.
        for config in config_list:
            try:
                return self._llava_call_binary_with_config(prompt, images, config, max_new_tokens, temperature, seed, do_sample)
            except Exception as e:
                print(f"Error: {e}")
                continue


    def llava_call(self, prompt: str, llm_config: dict) -> str:
        """
        Makes a call to the LLaVA service to generate text based on a given prompt
        """

        prompt, images = llava_formatter(prompt, order_image_tokens=False)

        for im in images:
            if len(im) == 0:
                raise RuntimeError("An image is empty!")

        return self.llava_call_binary(
            prompt,
            images,
            config_list=llm_config["config_list"],
            max_new_tokens=llm_config.get("max_new_tokens", 2000),
            temperature=llm_config.get("temperature", 0.5),
            seed=llm_config.get("seed", None),
            do_sample= llm_config.get("do_sample", True)
        )
