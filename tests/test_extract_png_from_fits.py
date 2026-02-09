from unittest.mock import patch

import numpy as np
import pytest

from src.extract_png_from_fits import (
    process_and_save_pngs,
    rescale_image_to_uint,
    RescaleConfig,
    ProcessingConfig,  # Assuming you created this dataclass as we discussed
)

VAL_0_PERCENT = 0
VAL_99_PERCENT = 255
VAL_NAN_BG = 42
VAL_BELOW_BLACK = 1
VAL_ABOVE_WHITE = 2
VAL_SCALED_MIN = 0
VAL_SCALED_MIDDLE = 254
VAL_PERCENTILE_BLACK = 20
VAL_PERCENTILE_WHITE = 99


@pytest.fixture
def sample_float_image():
    """A simple 10x10 float image with values from 0 to 99."""
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    data[5, 5] = np.nan
    return data


def test_rescale_basic(sample_float_image):
    """Tests a basic 0-100% percentile rescale."""
    # Note: np.nanpercentile ignores NaNs, so 0% is 0.0 and 100% is 99.0
    config = RescaleConfig(
        percentile_black=0.0,
        percentile_white=100.0,
        background_color=VAL_NAN_BG,
        replace_below_black=None,  # Explicitly set defaults
        replace_above_white=None,  # Explicitly set defaults
    )
    rescaled = rescale_image_to_uint(sample_float_image, config)
    # Check that 0.0 maps to 0
    assert rescaled[0, 0] == VAL_0_PERCENT
    # Check that 99.0 maps to 255
    # (rescaled[9, 9] corresponds to 99.0)
    assert rescaled[9, 9] == VAL_99_PERCENT
    # Check that the NaN value became the background_color
    assert rescaled[5, 5] == VAL_NAN_BG


def test_rescale_clipping_and_replace(sample_float_image):
    """
    Tests clipping and replacing values outside the percentiles.
    black_level = np.nanpercentile(..., 10.0) = 9.9
    white_level = np.nanpercentile(..., 90.0) = 89.1
    """
    config = RescaleConfig(
        percentile_black=10.0,
        percentile_white=90.0,
        background_color=0,  # Default
        replace_below_black=VAL_BELOW_BLACK,
        replace_above_white=VAL_ABOVE_WHITE,
    )
    rescaled = rescale_image_to_uint(sample_float_image, config)
    # Value 8.0 (at [0, 8]) is < 9.9, should be replaced with 1
    assert rescaled[0, 8] == VAL_BELOW_BLACK
    # Value 9.0 (at [0, 9]) is < 9.9, should be replaced with 1
    # This is the line that was failing!
    assert rescaled[0, 9] == VAL_BELOW_BLACK
    # Value 10.0 (at [1, 0]) is >= 9.9, should be scaled
    # (10.0 - 9.9) / (89.1 - 9.9) * 255 = 0.32... -> 0
    assert rescaled[1, 0] == VAL_SCALED_MIN
    # Value 89.0 (at [8, 9]) is <= 89.1, should be scaled
    # (89.0 - 9.9) / (89.1 - 9.9) * 255 = 251.7... -> 252
    assert rescaled[8, 9] == VAL_SCALED_MIDDLE
    # Value 90.0 (at [9, 0]) is > 89.1, should be replaced with 2
    assert rescaled[9, 0] == VAL_ABOVE_WHITE


def test_process_and_save_pngs_task_generation():
    """
    Tests that process_and_save_pngs generates the correct task list
    and skips existing files.
    """
    dummy_data = np.array([1, 2], dtype=np.float32)
    mock_params = {"percentile_black": [10, 20], "percentile_white": [99], "background_color": [0], "replace_below_black": [0], "replace_above_white": [255], "stretch_function": ["AsinhStretch"], "interval_function": ["ZScaleInterval"]}
    test_outdir = "fake/output/dir"

    with (
        patch("src.extract_png_from_fits.shared_memory.SharedMemory"),
        patch("src.extract_png_from_fits.np.ndarray"),
        patch("src.extract_png_from_fits.ProcessPoolExecutor") as mock_executor,
        patch("src.extract_png_from_fits.os.path.exists") as mock_exists,
        patch("builtins.open"),
        patch("src.extract_png_from_fits.json.load", return_value=mock_params),
    ):
        # Let's pretend the first file (b10_w99...) already exists
        # and the second one (b20_w99...) does not.
        def side_effect(path):
            if "b10_w99" in path:
                return True  # File exists
            return False  # File does not exist

        mock_exists.side_effect = side_effect
        # Get the 'map' method from the executor instance
        mock_map = mock_executor.return_value.__enter__.return_value.map
        process_and_save_pngs(
            data_to_process=dummy_data,
            processing_params=["dummy.json"],
            output_dir=test_outdir,
            overwrite=False,
        )
        # 6. --- Assertions ---
        tasks_passed_to_map = list(mock_map.call_args[0][1])
        # We expect only ONE task, for the b20_w99 file,
        # because the b10_w99 file was skipped.
        task_config_obj = tasks_passed_to_map[0][0]
        # Check that it is the correct object
        assert isinstance(task_config_obj, ProcessingConfig)
        # Access attributes by name, not index
        assert task_config_obj.percentile_black == VAL_PERCENTILE_BLACK
        assert task_config_obj.percentile_white == VAL_PERCENTILE_WHITE
        assert task_config_obj.background_color == 0
        assert task_config_obj.replace_below_black == 0
        assert task_config_obj.replace_above_white == VAL_99_PERCENT
        assert task_config_obj.stretch_function == "AsinhStretch"
        assert task_config_obj.interval_function == "ZScaleInterval"
