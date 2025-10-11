import cv2
import numpy as np

def crop_and_resize(
    image: np.ndarray,
    relative_bottom_y: float = 1.0,
    top_left: tuple[int, int] = (0, 0),
    target_resolution: tuple[int, int] = (3440, 1440),
    crop_by_target_size: bool = False
) -> np.ndarray | None:
    """
    Crops and resizes an image, ensuring the target aspect ratio is maintained.

    Args:
        image (np.ndarray): The input image (H, W, C).
        relative_bottom_y (float): The relative position of the bottom of the crop box
                                   (e.g., 0.8 means the bottom is at 80% of image height).
                                   Only used when crop_by_target_size is False.
        top_left (tuple[int, int]): The (y, x) coordinates for the top-left of the crop.
        target_resolution (tuple[int, int]): The final (width, height) of the output image.
        crop_by_target_size (bool): If True, crops a window with the exact dimensions of
                                    target_resolution. If False, calculates the crop window
                                    based on relative_bottom_y and the target aspect ratio.

    Returns:
        np.ndarray | None: The processed image, or None if the crop is invalid.
    """
    img_h, img_w = image.shape[:2]
    target_w, target_h = target_resolution
    top_y, top_x = top_left

    # Ensure top_left coordinates are valid
    if not (0 <= top_y < img_h and 0 <= top_x < img_w):
        print("Error: top_left coordinate is outside the image boundaries.")
        return None

    if crop_by_target_size:
        # Crop a fixed-size window directly
        bottom_y = top_y + target_h
        right_x = top_x + target_w
    else:
        # Calculate crop window to match the target aspect ratio
        if not (0.0 < relative_bottom_y <= 1.0):
            print("Error: relative_bottom_y must be between 0 and 1.")
            return None
        aspect_ratio = target_w / target_h
        # Calculate the bottom of the crop box in absolute pixels
        abs_bottom_y = int(relative_bottom_y * img_h)
        # Calculate crop height
        crop_h = abs_bottom_y - top_y
        if crop_h <= 0:
            print("Error: Calculated crop height is zero or negative.")
            return None
            
        crop_w = int(crop_h * aspect_ratio)
        bottom_y = top_y + crop_h
        right_x = top_x + crop_w
    if not (bottom_y <= img_h and right_x <= img_w):
        print("Error: Calculated crop window exceeds image boundaries.")
        return None
    cropped_image = image[top_y:bottom_y, top_x:right_x]
    resized_image = cv2.resize(cropped_image, target_resolution, interpolation=cv2.INTER_AREA)
    return resized_image