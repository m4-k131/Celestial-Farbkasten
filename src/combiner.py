import argparse
import json
import os

import cv2
import numpy as np
from lib.colors import COLORS
from paths import COLOR_IMAGE


def adjust_saturation_contrast(image: np.ndarray, saturation_scale: float = 1.5, contrast_scale: float = 1.2) -> np.ndarray:
    """Adjusts the saturation and contrast of an image.
    Saturation is adjusted first, then contrast, to prevent color data loss.

    Args:
        image: The input image in BGR format (standard for OpenCV).
        saturation_scale: Factor to scale saturation. >1 increases, <1 decreases.
        contrast_scale: Factor to scale value/contrast. >1 increases, <1 decreases.

    Returns:
        The adjusted image in BGR format.

    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    if saturation_scale:
        s = s.astype(np.float32)
        s = np.clip(s * saturation_scale, 0, 255)
        s = s.astype(np.uint8)
    if contrast_scale:
        v = v.astype(np.float32)
        v = np.clip(v * contrast_scale, 0, 255)
        v = v.astype(np.uint8)
    final_hsv = cv2.merge([h, s, v])
    adjusted_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return adjusted_image


def combine_config(config: dict, clip_image: bool = False) -> np.ndarray:
    combined_image = None
    images = []
    for image_config in config["images"]:
        if "combination" in image_config:
            loaded_image = combine_config(image_config)
        elif image_config["path"].endswith(".json"):
            loaded_image = combine_from_json(image_config["path"], image_config["factor"])
            if "clip" in image_config:
                loaded_image = np.clip(loaded_image, 0, 255)
        else:
            gray_image = cv2.imread(image_config["path"], cv2.IMREAD_GRAYSCALE)
            if gray_image is None:
                raise FileNotFoundError(f"Could not load image at path: {image_config['path']}")
            loaded_image = get_color_image(
                gray_image,
                image_config["color"],
                image_config["factor"]
            )
        images.append(loaded_image)

    for i in range(1, len(images)):
        print(images[i].shape)
        assert images[i - 1].shape == images[i].shape
    images = np.array(images)
    combined_image = images.sum(axis=0)
    if clip_image:
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    post_process = config.get("post_process")
    if post_process:
        combined_image = adjust_saturation_contrast(combined_image, post_process.get("saturation"), post_process.get("contrast"))
    return combined_image


def combine_from_json(json_path: str, factor: float | int = 1) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    out_image = combine_config(config)
    if factor != 1:
        return out_image * factor
        # out_image=np.clip((out_image.astype(np.float32) * factor), 0, 255).astype(np.uint8)
    return out_image


def get_color_image(gray_image: np.ndarray, color: str | tuple, factor: float | int = 1) -> np.ndarray:
    """Applies a color to a grayscale image, returning a float32 BGR image."""
    normalized_gray = gray_image.astype(np.float32) / 255.0
    if isinstance(color, str):
        if color in COLORS:
            color_array = np.array(COLORS[color], dtype=np.float32)
        else:
            print(f"{color} is not a valid color. Available pre-defined colors are {list(COLORS.keys())}. Using pure black and thus skipping this color")
            color_array = np.array([0, 0, 0], dtype=np.float32)
    else:  # TODO check if valid color
        color_array = np.array(color, dtype=np.float32)
    colored_image = normalized_gray[:, :, np.newaxis] * color_array * factor
    return colored_image


def main(input_json: str, imagename: str | None = None, suffix: str | None = None, outdir: str | None = None) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        config = json.load(f)
    out_image = combine_config(config, clip_image=True)
    if outdir is None:
        outdir = COLOR_IMAGE
    os.makedirs(outdir, exist_ok=True)
    if imagename is None:
        imagename = f"{os.path.basename(input_json).split('.')[0]}"
        print(imagename)
    imagename = f"{imagename}.png" if suffix is None else f"{imagename}_{suffix}.png"
    cv2.imwrite(os.path.join(outdir, imagename), out_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=str)
    parser.add_argument("--imagename", required=False, help="Uses name of input_json if not given", type=str)
    parser.add_argument("--suffix", required=False, type=str)
    parser.add_argument("--outdir", required=False, type=str)
    args = parser.parse_args()
    main(args.input_json, args.imagename, args.suffix, args.outdir)
