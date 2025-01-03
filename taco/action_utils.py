from PIL import Image, ImageDraw
import os
from taco.config import *

def get_full_path_data(full_filename):
    """
    Given a filename, returns the full path of the image file.
    full_filename: A string representing the filename
    """
    extensions = [".png", ".webp", ".jpg"]
    filename, curr_extension = os.path.splitext(full_filename)
    if full_filename.find("/") == -1: # try adding the image base path
        base_path = INPUT_IMAGE_PATH
        img_path = os.path.join(base_path, filename) 
        if os.path.exists(img_path):
            return img_path
    else:
        for ext in extensions: # try other image file extensions in the same directory
            if ext == curr_extension: continue
            new_filename = full_filename.replace(curr_extension, ext)
            if os.path.exists(new_filename): return new_filename
    return None

def image_processing(img, return_path=False):
    """
    Given an image file path or an image object, returns the image object in RGB format or only the path if return_path is True.
    img: A string representing the image file path or an image object
    return_path: A boolean indicating whether to return the path of the image file
    """
    if isinstance(img, Image.Image):
        assert return_path == False, "Cannot return path for an image object input"
        return img.convert("RGB")
    elif isinstance(img, str):
        final_path = img
        if not os.path.exists(img):
            final_path = get_full_path_data(img)
        if final_path:
            return final_path if return_path else Image.open(final_path).convert("RGB")
        else:
            raise FileNotFoundError(f"Image file not found: {img}")

def expand_bbox(bbox, original_image_size, margin=0.5):
    """
    Expands the bounding box by half of its width and height.
    bbox: A tuple (left, top, right, bottom)
    original_image_size: A tuple (width, height) of the original image size
    return: A tuple (new_left, new_top, new_right, new_bottom)
    """
    left, upper, right, lower = bbox
    width = right - left
    height = lower - upper

    # Calculate the new width and height
    new_width = width * (1 + margin) if margin <= 1.0 else width + margin
    new_height = height * (1 + margin) if margin <= 1.0 else height + margin

    # Calculate the center of the original bounding box
    center_x = left + width / 2
    center_y = upper + height / 2

    # Determine the new bounding box coordinates
    new_left = max(0, center_x - new_width / 2)
    new_upper = max(0, center_y - new_height / 2)
    new_right = min(original_image_size[0], center_x + new_width / 2)
    new_lower = min(original_image_size[1], center_y + new_height / 2)

    return (int(new_left), int(new_upper), int(new_right), int(new_lower))

