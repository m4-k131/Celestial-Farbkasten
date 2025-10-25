import numpy as np
import pytest
import cv2
from unittest.mock import patch

# Assumes pytest.ini is set up with 'pythonpath = src'
from src.combiner import get_color_image, adjust_saturation_contrast

# --- Constants for Tests ---
TEST_COLOR_TUPLE = (100, 50, 20)  # BGR
TEST_COLOR_NAME = "$TestColor"
TEST_COLOR_VALUE = (10, 20, 30)  # BGR


@pytest.fixture
def sample_gray_image():
    """A simple 2x2 grayscale image for testing."""
    # Values: 0 (black), 255 (white), 127.5 (mid-gray)
    return np.array([[0, 255], [127.5, 50]], dtype=np.float32)


def test_get_color_image_with_tuple(sample_gray_image):
    """
    Tests get_color_image with a simple array and a (B, G, R) tuple.
    """
    factor = 1.0
    colored_image = get_color_image(sample_gray_image, TEST_COLOR_TUPLE, factor)
    # --- Assertions ---
    assert colored_image.dtype == np.float32
    assert colored_image.shape == (2, 2, 3)  # Should now have 3 channels
    # 1. Black (0) should remain black (0, 0, 0)
    expected_black = [0.0, 0.0, 0.0]
    assert np.allclose(colored_image[0, 0], expected_black)
    # 2. White (255) should become the full color
    expected_white = [100.0, 50.0, 20.0]  # TEST_COLOR_TUPLE
    assert np.allclose(colored_image[0, 1], expected_white)
    # 3. Mid-gray (127.5) should be 50% of the color
    # 127.5 / 255.0 = 0.5
    expected_mid = [50.0, 25.0, 10.0]  # 0.5 * TEST_COLOR_TUPLE
    assert np.allclose(colored_image[1, 0], expected_mid)


def test_get_color_image_with_factor(sample_gray_image):
    """
    Tests that the 'factor' correctly scales the output.
    """
    factor = 0.5
    colored_image = get_color_image(sample_gray_image, TEST_COLOR_TUPLE, factor)
    # --- Assertions ---
    # White (255) should now be 50% of the full color
    # 1.0 * (100, 50, 20) * 0.5
    expected_white_factored = [50.0, 25.0, 10.0]
    assert np.allclose(colored_image[0, 1], expected_white_factored)


@patch("src.combiner.COLORS", {TEST_COLOR_NAME: TEST_COLOR_VALUE})
def test_get_color_image_with_string(sample_gray_image):
    """
    Tests that a color string is correctly looked up from the COLORS dict.
    We patch the dict to avoid depending on the real values.
    """
    colored_image = get_color_image(sample_gray_image, TEST_COLOR_NAME, factor=1.0)
    # --- Assertions ---
    # White (255) should become the full TEST_COLOR_VALUE
    expected_color = [10.0, 20.0, 30.0]
    assert np.allclose(colored_image[0, 1], expected_color)


def test_adjust_saturation_contrast():
    """
    Tests that saturation and contrast are increased.
    """
    # A simple BGR image (a dull red)
    original_bgr = np.array([[[100, 100, 200]]], dtype=np.uint8)
    # Convert to HSV to get base values
    original_hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    original_s = original_hsv[0, 0, 1]
    original_v = original_hsv[0, 0, 2]
    # --- Run the function ---
    adjusted_image = adjust_saturation_contrast(original_bgr, saturation_scale=1.5, contrast_scale=1.2)
    # --- Assertions ---
    adjusted_hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
    adjusted_s = adjusted_hsv[0, 0, 1]
    adjusted_v = adjusted_hsv[0, 0, 2]

    assert adjusted_s > original_s
    assert adjusted_v > original_v
