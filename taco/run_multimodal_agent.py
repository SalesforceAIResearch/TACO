import json
import os
import re
import pickle
import traceback
import argparse
import pandas as pd
from typing import Dict, Optional, Union
from autogen import Cache
from autogen.agentchat import AssistantAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.llava_agent import LLaVAAgent  
from taco.agent import UserAgent
from taco.prompt import DirectAnswerPrompt, CoTAPrompt, FeedbackPrompt
from taco.parser import Parser
from taco.executor import Executor
from taco.data import AgentDataset, format_query
from taco.action import *
from taco.config import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-format', default='cota', type=str, help="the inference prompt format", choices=['cota', 'direct'])
    parser.add_argument('--model', default="gpt-4o-2024-08-06", type=str, help="a string representing a unique model.")
    parser.add_argument('--dataset', default="MMVet", type=str, help="a string representing a unique dataset to run the agent on.")
    parser.add_argument('--execute', action='store_true', help="whether to use feedback from execution or not.")
    parser.add_argument('--language-only', action='store_true', help="whether to use language only model or not.")
    parser.add_argument('--simple-prompt', action='store_true', help="whether to use simplified prompt (without action descriptions and examples) or not.")
    parser.add_argument('--max-reply', default=10, type=int, help="the maximum number of replies after the first attempt.")
    parser.add_argument('--seed', default=42, type=int, help="the random seed used in autogen.")
    parser.add_argument('--exp-id', default=None, type=str, help="a unique string for identifying the current experiment.")
    parser.add_argument('--output-dir', default='prediction', type=str, help="the directory to save outputs to.")
    parser.add_argument('--infer-device-id', default="cuda:0", type=str, help="the str id of the device to use.")
    args = parser.parse_args()
    args = parser.parse_args()
    return args

def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("Terminate") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("Terminate") > -1
    else:
        raise NotImplementedError
    
def main():
    args = get_args()

    # Set up the prediction output directory
    run_id = f"{args.exp_id if args.exp_id else ''}-{args.prompt_format}-max-reply-{str(args.max_reply)}-seed-{str(args.seed)}"
    run_id = run_id.replace('/', '-')
    model = args.model.replace('/', '-')
    output_dir = os.path.join(args.output_dir, model, args.dataset.lower())
    wf_name = os.path.join(output_dir, f"{run_id}.jsonl")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Record ids of examples which we have run inference on (to skip these)
    has_inferenced = {}
    if os.path.exists(wf_name):
        rf = open(wf_name, "r")
        for line in rf:
            saved_data = json.loads(line)
            has_inferenced[saved_data["index"]] = saved_data
        rf.close()
    wf = open(wf_name, "a")
    
    # Set up the input and result directories for action execution
    full_input_path = os.path.join(INPUT_IMAGE_PATH, args.dataset.lower())
    full_result_path = os.path.join(RESULT_PATH, model, args.dataset.lower(), run_id)
    if not os.path.exists(full_result_path):
        os.makedirs(full_result_path, exist_ok=True)
    
    # Set up executor, prompt generator, parser, and user agent
    executor = Executor(input_folder=full_input_path, result_folder=full_result_path) if args.execute else None 
    actions = [
                OCR(),
                LocalizeObjects(), 
                GetObjects(),
                EstimateRegionDepth(),
                EstimateObjectDepth(),
                Crop(), 
                ZoomIn(),
                QueryLanguageModel(),
                GetImageToImagesSimilarity(),
                GetImageToTextsSimilarity(),
                GetTextToImagesSimilarity(),
                DetectFaces(),
                QueryKnowledgeBase(),
                Calculate(),
                SolveMathEquation(),
                Terminate(),
            ]
    if args.prompt_format == 'cota':
        prompt_generator = CoTAPrompt(actions=actions)
    elif args.prompt_format == 'direct':
        prompt_generator = DirectAnswerPrompt(actions=[])
    else:
        raise NotImplementedError(f"Prompt format {args.prompt_format} not supported")
    parser = Parser(prompt_generator=prompt_generator) 
    feedback_generator = FeedbackPrompt()
    user = UserAgent(
        name="user_agent",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=args.max_reply,
        is_termination_msg=checks_terminate_message,
        prompt_generator=prompt_generator,
        feedback_generator=feedback_generator,
        parser=parser,
        executor=executor,
        code_execution_config={'use_docker': False}
    )
    if args.simple_prompt:
        all_action_names = [action.name for action in actions]
        all_actions_str = ", ".join(all_action_names)
        simple_task_goal = f"""[BEGIN OF GOAL] You are a helpful assistant, and your goal is to solve the # USER REQUEST #. You can either rely on your own capabilities or perform actions with external tools to help you. You can use these actions: {all_actions_str} [END OF GOAL]"""
        system_prompt = simple_task_goal
    else:
        system_prompt = prompt_generator.get_task_prompt_only()
    print(system_prompt)
    
    data = AgentDataset(args.dataset)
    print(len(data.examples))
    
    # Initialize agents 
    all_messages = {}
    if args.model.find('gpt') > -1:
        config_list = [{
            "model": args.model,
            "api_key": OPENAI_API_KEY
        }]
        llm_config = {
            "seed": args.seed,
            "config_list": config_list,
            "cache_seed": None,
        }
        if args.language_only:
            mm_agent = AssistantAgent(
                    name="planner",
                    llm_config=llm_config,
                    human_input_mode='NEVER',
                    system_message=system_prompt, # The default system message of the AssistantAgent is overwritten here
                )
        else:
            mm_agent = MultimodalConversableAgent(
            name="mm_agent",
            system_message=system_prompt,
            human_input_mode='NEVER',
            max_consecutive_auto_reply=args.max_reply,
            llm_config=llm_config,
            )
    else:
        if (args.model.find("mllava") > -1 or args.model.lower().find("mantis") > -1):
            llava_mode = "mantis-hf"
            if args.model.find("clip") > 0:
                base_path = os.path.join(MANTIS_CKPT_PATH, "Mantis-8B-clip-llama3-pretraind")
            else:
                base_path = os.path.join(MANTIS_CKPT_PATH, "Mantis-8B-siglip-llama3-pretraind")
            model_name = args.model
            model_path = os.path.join(base_path, model_name.replace("/", "_"), "checkpoint-final")
        else:
            llava_mode =  "llava-ov-hf"
            # append "_qwen" to the model name to make sure the qwen-based llava models are used in the llava codebase
            model_name = f"{args.model}_qwen"
            model_path =  os.path.join(LLAVA_OV_CKPT_PATH, args.model)

        config_list = [
            {
                "model": model_name,
                "model_path": model_path,
                "api_key": "None",
                "base_url": "",
                "llava_mode": llava_mode
            }
        ]
        default_config = {
            "seed": args.seed, 
            "config_list": config_list, 
            "cache_seed": None,
            "temperature": 0, 
            "do_sample": False,
            "device_id": args.infer_device_id,
            "max_new_tokens": 512 if args.prompt_format == "direct" else 2000,
            }

        mm_agent = LLaVAAgent(
            name="mm_agnet",
            system_message=system_prompt,
            human_input_mode='NEVER',
            max_consecutive_auto_reply=args.max_reply,
            llm_config=default_config
        )

    # Run experiment on all examples
    for idx in range(len(data)):
        failed = False
        example = data[idx]
        if not example: 
            continue
        if example['index'] in has_inferenced:
            continue
        else:
            query = example['question']

            formatted_query = format_query(example['images'], query)
            try:
                user.initiate_chat(
                    mm_agent,
                    message=formatted_query,
                    task=example
                )
                all_messages = mm_agent.chat_messages
            except Exception as e:
                print(e)
                print(f"skipping {example['index']}..")
                failed = True
                print("Traceback details:")
                traceback.print_exc()  # Prints the full traceback to the console

            if args.output_dir and not failed:
                messages = {agent.name: msg for agent, msg in all_messages.items()}
                example['all_messages'] = messages
                example['prediction'] = user.final_answer
                example['called_tools'] = user.called_tools
                example['images'] += user.new_image_paths
                if example['prediction'] is None:
                    # try to extract the final answer again
                    msg = messages['user_agent'][-1]
                    if 'content' in msg and msg['content']:
                        for content in msg['content']:
                            if 'text' in content:
                                final_text = content['text']
                                if final_text.find('answer') > -1:
                                    example['prediction'] = final_text
                                    break
                        text = example['prediction']
                        if text and isinstance(text, str):
                            match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
                            if match:
                                extracted_answer = match.group(1)
                                example['prediction'] = extracted_answer
                
                save_keys = ['index', 'question', 'prediction', 'all_messages']
                output_dict =  {}
                for k in example:
                    if k == "all_messages":
                        all_msgs = example[k]
                        save_path = os.path.join(full_result_path, str(example['index']))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path, exist_ok=True)
                        # save all messages into a pickle file instead of json because some content might be un-serializable
                        all_msg_path = os.path.join(save_path, "all_messages.pkl")
                        with open(all_msg_path, "wb") as file:
                            pickle.dump(all_msgs, file)
                        output_dict[k] = all_msg_path
                    else:
                        output_dict[k] = example[k]
                # print(output_dict)
                wf.write(json.dumps(output_dict) + "\n")
                wf.flush()

            user.reset()
            mm_agent.reset()
                    
if __name__ == '__main__':
    main()