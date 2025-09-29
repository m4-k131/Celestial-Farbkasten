import os 
import cv2 
import numpy as np
import math  
import argparse
import json


def combine_config(config, final_image=False):
    combined_image = None 
    images = []
    for image_config in config["images"]:
        if "combination" in image_config:
            loaded_image = (combine_config(image_config))
        elif image_config["path"].endswith(".json"):
            loaded_image = combine_from_json(image_config["path"], image_config["factor"])
        else:
            loaded_image = get_color_image(image_config["path"], image_config["color"], image_config["factor"])
        if config["colorspace"] != "bgr":
            loaded_image = cv2.cvtColor(loaded_image, getattr(cv2, f"COLOR_BGR2{config['colorspace'].upper()}"))  
        images.append(loaded_image)
    
    for i in range(1, len(images)):
        assert images[i-1].shape == images[i].shape
    images = np.array(images)
    combined_image = images.sum(axis=0)
    if config["colorspace"] != "bgr":
        combined_image = cv2.cvtColor(combined_image, getattr(cv2, f"COLOR_{config['colorspace'].upper()}2BGR")) 
    if final_image:
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    return combined_image

def combine_from_json(json_path, factor=1):
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    out_image= combine_config(config)
    if factor != 1:
        return out_image * factor
        #out_image=np.clip((out_image.astype(np.float32) * factor), 0, 255).astype(np.uint8)
    return out_image


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


def main(input_json, imagename=None, suffix= None, outdir="outs"):
    print(imagename)
    with open(input_json, "r", encoding="utf-8") as f:
        config = json.load(f)
    out_image= combine_config(config, final_image=True)
    os.makedirs(outdir, exist_ok=True)
    if imagename is None:
        imagename = f'{os.path.basename(input_json).split(".")[-2]}'
        print(imagename)
    imagename =f"{imagename}.png" if suffix is None else f"{imagename}_{suffix}.png"
    cv2.imwrite(os.path.join(outdir, imagename), out_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--imagename", required=False, help="Uses name of input_json if not given")
    parser.add_argument("--suffix", required=False)
    parser.add_argument("--outdir", required=False, default="outs")
    args = parser.parse_args()
    main(args.input_json, args.imagename, args.suffix, args.outdir)