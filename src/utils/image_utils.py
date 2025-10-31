import argparse
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CropConfig:
    """Holds all parameters for a crop and resize operation."""

    bottom_y: float | int = 1.0
    top_left: tuple[int, int] = (0, 0)
    target_resolution: tuple[int, int] = (3440, 1440)
    crop_by_target_size: bool = False


def crop_and_resize(image: np.ndarray, config: CropConfig) -> np.ndarray | None:
    """Crops and resizes an image, maintaining the target aspect ratio.

    Args:
        image (np.ndarray): The input image (H, W, C).
        bottom_y (float): Determines the bottom of the crop.
                          - If > 1.0, it's treated as an ABSOLUTE y-pixel coordinate.
                          - If <= 1.0, it's treated as a RELATIVE position (percentage of image height).
                          Only used when crop_by_target_size is False.
        top_left (tuple[int, int]): The (y, x) coordinates for the top-left of the crop.
        target_resolution (tuple[int, int]): The final (width, height) of the output image.
        crop_by_target_size (bool): If True, crops a window with the exact dimensions of
                                    target_resolution. If False, calculates the crop window
                                    based on bottom_y and the target aspect ratio.

    Returns:
        np.ndarray | None: The processed image, or None if the crop is invalid.

    """
    img_h, img_w = image.shape[:2]
    # Unpack values from the config object
    target_w, target_h = config.target_resolution
    top_y, top_x = config.top_left
    if not (0 <= top_y < img_h and 0 <= top_x < img_w):
        print("Error: top_left coordinate is outside the image boundaries.")
        return None
    if config.crop_by_target_size:
        abs_bottom_y = top_y + target_h
        right_x = top_x + target_w
    else:
        if config.bottom_y > 1.0:
            abs_bottom_y = int(config.bottom_y)
        else:  # Treat as relative
            if not 0 < config.bottom_y <= 1.0:
                print("Error: Relative bottom_y must be in the range (0, 1].")
                return None
            abs_bottom_y = int(config.bottom_y * img_h)
        crop_h = abs_bottom_y - top_y
        if crop_h <= 0:
            print(f"Error: Calculated crop height ({crop_h}px) is zero or negative.")
            return None
        aspect_ratio = target_w / target_h
        crop_w = int(crop_h * aspect_ratio)
        right_x = top_x + crop_w
    if not (abs_bottom_y <= img_h and right_x <= img_w):
        print("Error: Calculated crop window [...] exceeds image dimensions.")
        return None
    cropped_image = image[top_y:abs_bottom_y, top_x:right_x]
    resized_image = cv2.resize(cropped_image, config.target_resolution, interpolation=cv2.INTER_AREA)
    return resized_image


def main(image_path: str, out_path: str, config: CropConfig) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    cropped_image = crop_and_resize(image, config)
    if cropped_image is not None:
        cv2.imwrite(out_path, cropped_image)
        print(f"Saved cropped image to {out_path}")
    else:
        print("Cropping failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("out_path")
    parser.add_argument("--target_resolution", required=False, default=(3440, 1440))  # UltraWide master race
    parser.add_argument("--top_left_x", required=False, default=0, type=int)
    parser.add_argument("--top_left_y", required=False, default=0, type=int)
    parser.add_argument("--bottom_y", required=False, default=1)  # float, int
    parser.add_argument("--crop_by_target_size", action="store_true")
    args = parser.parse_args()
    config = CropConfig(bottom_y=args.bottom_y, top_left=(args.top_left_y, args.top_left_x), target_resolution=(args.target_w, args.target_h), crop_by_target_size=args.crop_by_target_size)
    main(args.image_path, args.out_path, config)
