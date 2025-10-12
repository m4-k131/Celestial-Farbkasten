import os 
import cv2 
import numpy as np
import math  
import argparse
import json

from paths import COLOR_IMAGE

COLORS = {
    "$CosmicGold": (20, 150, 255),
    "$DeepSpaceBlue" : (100, 30, 20),
    "$NebulaMagenta": (200, 40, 180),
    "$CyanGas": (220, 200, 0),
    "$Starlight": (150, 223, 255),
    "$RoyalVoid": (130, 0, 75),
    "$OxidizedRust": (20, 90, 200),
    "$OxygenTeal": (160, 180, 40),
    "$PaleHotYellow": (205, 250, 255),
    "$DeepCrimson": (30, 10, 150),
    "$SunsetOrange": (0, 120, 255), 
    "$ElectricViolet": (211, 0, 148),
    "$LuminousMint": (175, 255, 100),
    "$CharcoalVoid":(30, 25, 25),
    "$StellarCrimson": (0, 50, 255),
    "$DeepRuby": (60, 0, 240)
}

def combine_config(config, clip_image=False):
    combined_image = None 
    images = []
    for image_config in config["images"]:
        if "combination" in image_config:
            loaded_image = (combine_config(image_config))
        elif image_config["path"].endswith(".json"):
            loaded_image = combine_from_json(image_config["path"], image_config["factor"])
            if "clip" in image_config:
                loaded_image = np.clip(loaded_image, 0,255)
        else:
            loaded_image = get_color_image(image_config["path"], image_config["color"], image_config["factor"])
        images.append(loaded_image)

    for i in range(1, len(images)):
        assert images[i-1].shape == images[i].shape
    images = np.array(images)
    combined_image = images.sum(axis=0)
    if clip_image:
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


def get_color_image(path, color, factor=1):
    """
    Applies a color to a grayscale image, returning a float32 BGR image.
    """
    gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f"Could not load image at path: {path}")
    normalized_gray = gray_image.astype(np.float32) / 255.0
    if isinstance(color, str):
        if color in COLORS:
            color_array = np.array(COLORS[color], dtype=np.float32)
        else:
            print(f"{color} is not a valid color. Available pre-defined colors are {list(COLORS.keys())}. Using pure black and thus skipping this color")
            color_array = np.array([0,0,0], dtype=np.float32)
    else: #TODO check if valid color
        color_array = np.array(color, dtype=np.float32)
    colored_image = normalized_gray[:, :, np.newaxis] * color_array * factor
    return colored_image


def main(input_json, imagename=None, suffix= None, outdir=None):
    with open(input_json, "r", encoding="utf-8") as f:
        config = json.load(f)
    out_image= combine_config(config, clip_image=True)
    if outdir is None:
        outdir = COLOR_IMAGE
    os.makedirs(outdir, exist_ok=True)
    if imagename is None:
        imagename = f'{os.path.basename(input_json).split(".")[0]}'
        print(imagename)
    imagename =f"{imagename}.png" if suffix is None else f"{imagename}_{suffix}.png"
    cv2.imwrite(os.path.join(outdir, imagename), out_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--imagename", required=False, help="Uses name of input_json if not given")
    parser.add_argument("--suffix", required=False)
    parser.add_argument("--outdir", required=False)
    args = parser.parse_args()
    main(args.input_json, args.imagename, args.suffix, args.outdir)