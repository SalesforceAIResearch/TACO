import re
import os
import json
from typing import Dict, List, Optional, Union
from taco.observation import BaseObservation
from PIL import Image

demo_examples = [
    {
        "user_request": """In image-0, Which of the two objects on the plate is the biggest?\nA. The pile of scrambled eggs is the biggest.\nB. The strawberries are the biggest object.\nPlease answer directly with only the letter of the correct option and nothing else.""", 
        "steps": [
            {
                "id": 1, 
                "thought": "To determine which of the two objects on the plate is larger, I need to analyze the size of the scrambled eggs, and the strawberries", 
                "actions": [{"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": ["scrambled eggs", "strawberries"]}}],
                "observation": {
                "image": "image-1",
                "regions": [{"label": "eggs", "bbox": [0.5, 0.6, 0.6, 0.8], "score": 0.85}, {"label": "strawberries", "bbox": [0.4, 0.5, 0.45, 0.7], "score": 0.54}],
                }
            }, {
                "id": 2,
                "thought": "To calculate the area of a bounding box, we can use the formula: area = (x_max - x_min) * (y_max - y_min). We first get the area of the scrambled eggs.",
                "actions": [{"name": "Calculate", "arguments": {"expression": "(0.6-0.5) * (0.8-0.6)"}}],
                "observation": {"result": "0.02"}, 
            },
            {
                "id": 3,
                "thought": "Then, we also calculate the area of the strawberries.",
                "actions": [{"name": "Calculate", "arguments": {"expression": "(0.45-0.4) * (0.7-0.5)"}}],
                "observation": {"result": "0.01"},
            },
            {
                "id": 4,
                "thought": "Since 0.02 > 0.01, it is apparent that the eggs cover a larger area within their bounding box.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "A"}}],
                "observation": {},
            },
            ]
    },
    {
        "user_request": """Given the input image image-0, How many pedestrians are there in the image? Please answer directly with a single word or number.""", 
        "steps": [
            {
                "id": 1, 
                "thought": "To determine the number of pedestrians, I need to first localize them on the image.", 
                "actions": [{"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": ["person"]}}], 
                "observation": {
                "image": "image-1",
                "regions": [{"label": "person", "bbox": [0.77, 0.47, 0.79, 0.54], "score": 0.83}, {"label": "person-2", "bbox": [0.69, 0.49, 0.7, 0.52], "score": 0.43}]
                }
            }, {
                "id": 2,
                "thought": "The LocalizeObjects action returns two regions for \"person\", but one of the regions has a lower confidence score. Upon a closer look at the output image image-1, we can see that there is actually only one pedestrian in the image.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "1"}}],
                "observation": {},
            }]
        
    },
    {
        "user_request": """Based on image-0, is the object on top bigger than the object below?\nA. The object on the bottom is bigger.\nB. The object on top is bigger.\nC. Both objects are the same size.\nPlease answer directly with only the letter of the correct option and nothing else.""",
        "steps": [
            {
                "id": 1,
                "thought": "By looking at the image, we can see that both objects are game consoles of the same brand and size.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "C"}}],
                "observation": {},
            }]
    },
    {
        "user_request": """What is x in the image?""",
        "steps": [
            {
                "id": 1, 
                "thought": "To get the result of the equation, I need to first extract the equation from the image.",
                "actions": [{"name": "OCR", "arguments": {"image": "image-0"}}],
                "observation": {"text": "x-2^3=0"},
            },
            {
                "id": 2, 
                "thought": "The math equation is 'x-2^3=0', and I need to find x. I can solve it with a math-related tool.",
                "actions": [{"name": "SolveMathEquation", "arguments": {"query": "x-2^3=0, what is x?"}}],
                "observation": {"result": "x = 8"},
            },
            {
                "id": 3, 
                "thought": "As suggested in the last observation, the answer is 8.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "8"}}],
                "observation": {},
            }
        ]
    }
]

class PlanPrompt:
    def __init__(self, goal, instruction, tool_metadata, demos, requirements):
        """Generate a planning prompt that consists of instruction, tool metadat, demos and requirements.

        Args:
            query: the query of the user
        Returns:
            str: the generated prompt.
        """
        self.goal = goal
        self.instruction = instruction
        self.tool_metadata = tool_metadata
        self.demos = demos
        self.requirements = requirements

    def get_prompt_for_curr_query(self, query):
        """(Abstract method) Generate a prompt based on the received query.

        Args:
            query: the query of the user
        Returns:
            str: the generated prompt.
        """
        pass

    def get_task_prompt_only(self):
        """Get the task prompt only.

        Returns:
            str: the task prompt.
        """
        pass


class DirectAnswerPrompt(PlanPrompt):
    def __init__(self, actions):
        goal = """You are a helpful assistant, and your goal is to answer the # USER REQUEST # based on the image(s).\n"""

        super().__init__(goal, "", "", "", "")

    def get_prompt_for_curr_query(self, query):
        request = f"""\n# USER REQUEST #:\n{query}\n"""
        return request

    def get_task_prompt_only(self):
        return self.goal


class CoTAPrompt(PlanPrompt):
    def __init__(self, actions):
        goal = """[BEGIN OF GOAL]\n"""
        goal += """You are a helpful assistant, and your goal is to solve the # USER REQUEST #. You can either rely on your own capabilities or perform actions with external tools to help you. A list of all available actions are provided to you in the below.\n"""
        goal += """[END OF GOAL]\n\n"""
        
        action_metadata = "[BEGIN OF ACTIONS]\n"
        key2word = {"name": "Name", "description": "Description", "args_spec": "Arguments", "rets_spec": "Returns", "examples": "Examples"}
        for action in actions:
            for key, value in action.__dict__.items():
                if key not in key2word: continue
                word = key2word[key]
                if key == "examples":
                    action_metadata += f"{word}:\n"
                    for i, example in enumerate(value):
                        action_metadata += f"{json.dumps(example)}\n"
                elif key == "arguments" or key == "returns":
                    action_metadata += f"{word}: {json.dumps(value)}\n"
                else:
                    action_metadata += f"{word}: {value}\n"
            action_metadata += "\n"
        action_metadata += "[END OF ACTIONS]\n\n"
        
        instruction = """[BEGIN OF TASK INSTRUCTIONS]\n"""
        instruction += """1. You must only select actions from # ACTIONS #.\n"""
        instruction += """2. You can only call one action at a time.\n"""
        instruction += """3. If no action is needed, please make actions an empty list (i.e. “actions”: []).\n"""
        instruction += """4. You must always call Terminate with your final answer at the end.\n"""
        instruction += """[END OF TASK INSTRUCTIONS]\n\n""" 
        
        instruction += """[BEGIN OF FORMAT INSTRUCTIONS]\n"""
        instruction += """Your output should be in a strict JSON format as follows:\n"""
        instruction += """{"thought": "the thought process, or an empty string", "actions": [{"name": "action1", "arguments": {"argument1": "value1", "argument2": "value2"}}]}\n"""
        instruction += """[END OF FORMAT INSTRUCTIONS]\n\n"""
        
        demos = "[BEGIN OF EXAMPLES]:\n"
        for demo in demo_examples:
            demos += f"# USER REQUEST #:\n {demo['user_request']}\n"
            demos += f"# RESPONSE #:\n"
            for i, step in enumerate(demo["steps"]):
                thought_action_dict = {"thought": step["thought"], "actions": step["actions"]}
                demos += f"{json.dumps(thought_action_dict)}\n"
                if step["observation"]:
                    demos += f"OBSERVATION:\n"
                    demos += f"{json.dumps(step['observation'])}\n"
            demos += "\n"
        demos += "[END OF EXAMPLES]\n"

        super().__init__(goal, instruction, action_metadata, demos, "")

    def get_prompt_for_curr_query(self, query):
        request = f"""\n# USER REQUEST #:\n{query}\nNow please generate your response:\n""" # \n# STEP 1 #:
        return request

    def get_task_prompt_only(self):
        return self.goal +  self.tool_metadata + self.instruction + self.demos


class FeedbackPrompt:
    def __init__(self):
        self.default_feedback_msg = f"\nPlease try again to fix the error. Or, reply with Terminate only if you believe this error is not fixable."
        self.msg_prefix = "OBSERVATION:\n"
        self.msg_suffix = f"\nThe OBSERVATION can be incomplete or incorrect, so please be critical and decide how to make use of it. If you've gathered sufficient information to answer the question, call Terminate with the final answer. Now, please generate the response for the next step." 

    def get_prompt(self, stage, results):
        """Generate a feedback prompt based on the received observation.

        Args:
            observation: the observation of the user
        Returns:
            str: the generated prompt.
        """
        if stage == "parsing":
            error_code, error_msg = results["error_code"], results["message"]
            if error_code == "json":
                feedback_msg = f"\nPlease format the output to a strict JSON format can be converted by json.loads()."
                feedback_msg += """\nRequirements:\n1. Do not change the content;\n2. Consider changing single quotes to double quotes or vice versa where applicable;\n3. Consider adding or removing curly bracket or removing the period at the end if any.""" #;\n4. Don't tolerate any possible irregular formatting
            else: # unknown, or any other error codes
                feedback_msg = self.default_feedback_msg
            return self.msg_prefix + error_msg + feedback_msg
        elif stage == "execution":
            observation = results["content"]
            if results["status"]:
                obs_str = self.msg_prefix
                obs_dict = {}
                for attribute in dir(observation):
                    if attribute == "id": continue
                    # Skip private attributes and methods (those starting with '__')
                    if not attribute.startswith('__'):
                        value = getattr(observation, attribute)
                        if attribute == "image": 
                            # get the image filename only without the file extension (e.g. '.jpg') at the end
                            image_filename = os.path.basename(value).split('.')[0] 
                            # format image filename with this format so the image can be detected by autogen code
                            obs_dict[attribute] = f"{image_filename}: <img {value}>"
                        else:
                            obs_dict[attribute] = value
                obs_str += str(obs_dict)
            else:
                obs_str = results["message"]

            return obs_str + self.msg_suffix
