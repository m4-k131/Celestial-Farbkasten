import os
import cv2
import numpy as np
import math
import argparse
import json

from paths import COLOR_IMAGE

COLORS = {
    "$CosmicGold": (20, 150, 255),
    "$DeepSpaceBlue": (100, 30, 20),
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
    "$CharcoalVoid": (30, 25, 25),
    "$StellarCrimson": (0, 50, 255),
    "$DeepRuby": (60, 0, 240),
    # Pure, overwhelming red. Subtracts blue and green.
    "$AggressiveHydrogenAlpha": (-150, -150, 255),
    # Intense orange-red for sulphur emissions (SII).
    "$SulphurBurn": (-100, 50, 255),
    # A piercing cyan for oxygen (OIII), subtracting red.
    "$OxygenGlow": (255, 200, -100),
    # Darkens everything it touches, enhancing shadows.
    "$VoidCrusher": (-50, -50, -50),
    # A brighter, more intense gold that suppresses blue.
    "$StarfireGold": (-50, 200, 255),
    # A vibrant teal that removes green, useful for specific nebula gases.
    "$PlasmaTeal": (255, -100, 0),
}


def adjust_saturation_contrast(image: np.ndarray, saturation_scale: float = 1.5, contrast_scale: float = 1.2) -> np.ndarray:
    """
    Adjusts the saturation and contrast of an image.
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


def combine_config(config, clip_image=False):
    combined_image = None
    images = []
    for image_config in config["images"]:
        if "combination" in image_config:
            loaded_image = (combine_config(image_config))
        elif image_config["path"].endswith(".json"):
            loaded_image = combine_from_json(
                image_config["path"], image_config["factor"])
            if "clip" in image_config:
                loaded_image = np.clip(loaded_image, 0, 255)
        else:
            loaded_image = get_color_image(
                image_config["path"], image_config["color"], image_config["factor"])
        images.append(loaded_image)

    for i in range(1, len(images)):
        print(images[i].shape)
        assert images[i-1].shape == images[i].shape
    images = np.array(images)
    combined_image = images.sum(axis=0)
    if clip_image:
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    post_process = config.get("post_process")
    if post_process:
        combined_image = adjust_saturation_contrast(
            combined_image, post_process.get("saturation"), post_process.get("contrast"))

    return combined_image


def combine_from_json(json_path, factor=1):
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    out_image = combine_config(config)
    if factor != 1:
        return out_image * factor
        # out_image=np.clip((out_image.astype(np.float32) * factor), 0, 255).astype(np.uint8)
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
            color_array = np.array([0, 0, 0], dtype=np.float32)
    else:  # TODO check if valid color
        color_array = np.array(color, dtype=np.float32)
    colored_image = normalized_gray[:, :, np.newaxis] * color_array * factor
    return colored_image


def main(input_json, imagename=None, suffix=None, outdir=None):
    with open(input_json, "r", encoding="utf-8") as f:
        config = json.load(f)
    out_image = combine_config(config, clip_image=True)
    if outdir is None:
        outdir = COLOR_IMAGE
    os.makedirs(outdir, exist_ok=True)
    if imagename is None:
        imagename = f'{os.path.basename(input_json).split(".")[0]}'
        print(imagename)
    imagename = f"{imagename}.png" if suffix is None else f"{imagename}_{suffix}.png"
    cv2.imwrite(os.path.join(outdir, imagename), out_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--imagename", required=False,
                        help="Uses name of input_json if not given")
    parser.add_argument("--suffix", required=False)
    parser.add_argument("--outdir", required=False)
    args = parser.parse_args()
    main(args.input_json, args.imagename, args.suffix, args.outdir)
