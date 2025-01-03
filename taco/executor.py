import os
import pickle
import json
from taco.config import *
from taco.action import *
from taco.observation import BaseObservation


class Executor:
    def __init__(
        self, 
        input_folder: str = INPUT_IMAGE_PATH, 
        result_folder: str = RESULT_PATH,
    ) -> None:
        self.result_folder = result_folder
        self.input_folder = input_folder

    def get_full_image_path(self, task, image_filename):
        input_image_id = int(image_filename.split("-")[1])
        num_input_images = len(task['images'])
        if input_image_id < num_input_images: # this image argument is one of the input images
            full_path = task['images'][input_image_id]
        else:
            full_path = os.path.join(self.full_result_path, f"{image_filename}.jpg")
        return full_path
        
    def execute(self, step_id, image_id, content_dict, task):
        task_id = task['index']
        # update input and result paths
        self.full_input_path = os.path.join(self.input_folder, str(task_id))
        self.full_result_path = os.path.join(self.result_folder, str(task_id))
        if not os.path.exists(self.full_result_path):
            os.makedirs(self.full_result_path)

        try:
            if len(content_dict) == 0:
                next_obs = BaseObservation(result_dict={'message': f"No action has been provided. Proceed to the next step."})
                return {'status': True, 'content': next_obs, 'message': f"No action has been provided. Proceed to the next step."}

            function = content_dict
            function_args = function['arguments']
            action = globals()[function['name']]()
            
            # preprocess the image arguments by getting their full paths
            if 'image' in function_args:
                function_args['image'] = self.get_full_image_path(task, function_args['image'])
            
            if 'other_images' in function_args:
                new_images = []
                function_args['other_images_raw'] = function_args['other_images']
                for image in function_args['other_images']:
                    new_image = self.get_full_image_path(task, image)
                    new_images.append(new_image)
                function_args['other_images'] = new_images
            result = action(**function_args)

            new_images = []
            for key, output in result.items():
                # save the image output to the result folder
                if isinstance(output, Image.Image):
                    file_path = os.path.join(self.full_result_path, f"image-{image_id}.jpg")
                    output.save(file_path)
                    new_images.append(file_path)
                    result[key] = file_path

            next_obs = BaseObservation(result_dict=result)
            return {'status': True, 'content': next_obs, 'message': f"Execution succeeded.", 'images': new_images}
        except Exception as err:
            next_obs = BaseObservation(result_dict={'message': f"Execution failed with {type(err)}: {err}."})
            return {'status': False, 'content': next_obs, 'message': f"Execution failed with {type(err)}: {err}."}
