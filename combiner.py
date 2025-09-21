import os 
import cv2 
import numpy as np
import math  

INPUT_CONFIG = {
    "operand":  "+",
    "colorspace": "bgr",
    "images": [
        {"path":"C:/Users/malte/Code/astro_project/m74out_long/jw02107-o039_t018_miri_f770w_i2d/b40_w99_nan0_bb0_aw255.png",
         "color": [255, 40, 130],
         "factor": 0.9
         },
        {"path": "C:/Users/malte/Code/astro_project/m74out_long/jw02107-o039_t018_miri_f1000w_i2d/b20_w99_nan0_bb0_aw255.png",
         "color": [100, 240, 0],
         "factor": 0.9
         },
          {"path":"C:/Users/malte/Code/astro_project/m74out_long/jw02107-o039_t018_miri_f1130w_i2d/b40_w99_nan0_bb0_aw255.png",
         "color": [0, 20, 255], 
         "factor": 1
         },
        {"path":"C:/Users/malte/Code/astro_project/m74out_long/jw02107-o039_t018_miri_f2100w_i2d/b40_w100_nan0_bb0_aw0.png",
         "color": [150, 150, 150],
         "factor": 0.01
         }
    ]
}

LUMINANCE_CONFIG = {
    "operand":  "+",
    "colorspace": "bgr",
    "images": [
        {"path":"C:/Users/malte/Code/astro_project/m74out_long/jw02107-o039_t018_miri_f770w_i2d/b40_w99_nan0_bb0_aw255.png",
         "color": [255, 0, 0],
         "factor": 0.8
         },
        {"path": "C:/Users/malte/Code/astro_project/m74out_long/jw02107-o039_t018_miri_f1000w_i2d/b20_w99_nan0_bb0_aw255.png",
        "color": [0, 255, 0],
         "factor": 1.0
         },
          {"path":"C:/Users/malte/Code/astro_project/m74out_long/jw02107-o039_t018_miri_f1130w_i2d/b40_w99_nan0_bb0_aw255.png",
         "color": [0, 0, 255],
         "factor": 0.9
         },
        {"path":"C:/Users/malte/Code/astro_project/m74out_long/jw02107-o039_t018_miri_f2100w_i2d/b40_w100_nan0_bb0_aw0.png",
         "color": [255, 255, 255],
         "factor": 0.3
         }
    ]
}


def combine_config(config):
    combined_image = None 
    images = []
    for image_config in config["images"]:
        if "operand" in image_config:
            loaded_image = (combine_config(image_config))
        else:
            loaded_image = get_color_image(image_config["path"], image_config["color"], image_config["factor"])
        if config["colorspace"] != "bgr":
            pass 
        images.append(loaded_image)
    
    for i in range(1, len(images)):
        assert images[i-1].shape == images[i].shape
    images = np.array(images)
    combined_image = images.sum(axis=0)
    if config["colorspace"] != "bgr":
        pass 
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    return combined_image


def get_color_image(path, color, factor):
    """
    Applies a color to a grayscale image, returning a float32 BGR image.
    """
    gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f"Could not load image at path: {path}")
        
    normalized_gray = gray_image.astype(np.float32) / 255.0
    color_array = np.array(color, dtype=np.float32)
    colored_image = normalized_gray[:, :, np.newaxis] * color_array * factor
    
    return colored_image

image =  combine_config(INPUT_CONFIG)
cv2.imwrite("lower_green3.png", image)