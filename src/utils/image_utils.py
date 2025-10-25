import argparse

import cv2
import numpy as np


def crop_and_resize(image: np.ndarray, bottom_y: float = 1.0, top_left: tuple[int, int] = (0, 0), target_resolution: tuple[int, int] = (3440, 1440), crop_by_target_size: bool = False) -> np.ndarray | None:
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
    target_w, target_h = target_resolution
    top_y, top_x = top_left
    if not (0 <= top_y < img_h and 0 <= top_x < img_w):
        print("Error: top_left coordinate is outside the image boundaries.")
        return None
    if crop_by_target_size:
        abs_bottom_y = top_y + target_h
        right_x = top_x + target_w
    else:
        if bottom_y > 1.0:
            abs_bottom_y = int(bottom_y)
        else:  # Treat as relative
            if not 0 < bottom_y <= 1.0:
                print("Error: Relative bottom_y must be in the range (0, 1].")
                return None
            abs_bottom_y = int(bottom_y * img_h)
        crop_h = abs_bottom_y - top_y
        if crop_h <= 0:
            print(f"Error: Calculated crop height ({crop_h}px) is zero or negative.")
            return None
        aspect_ratio = target_w / target_h
        crop_w = int(crop_h * aspect_ratio)
        right_x = top_x + crop_w
    if not (abs_bottom_y <= img_h and right_x <= img_w):
        print(f"Error: Calculated crop window [(y1:{top_y}, x1:{top_x}), (y2:{abs_bottom_y}, x2:{right_x})] exceeds image dimensions (H:{img_h}, W:{img_w}).")
        return None
    cropped_image = image[top_y:abs_bottom_y, top_x:right_x]
    resized_image = cv2.resize(cropped_image, target_resolution, interpolation=cv2.INTER_AREA)
    return resized_image


def main(image_path: str, out_path: str, target_resolution: tuple[int, int], bottom_y: float | int = 1.0, top_left: tuple[int, int] = (0, 0), crop_by_target_size: bool = False) -> None:
    image = cv2.imread(image_path)
    cropped_image = crop_and_resize(image, bottom_y, top_left, target_resolution, crop_by_target_size)
    cv2.imwrite(out_path, cropped_image)


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
    main(args.image_path, args.out_path, args.target_resolution, float(args.bottom_y), (int(args.top_left_y), int(args.top_left_x)), args.crop_by_target_size)
