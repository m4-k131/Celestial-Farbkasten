"""PNG generation worker: imported alone by multiprocessing children to avoid loading reproject/skimage/shapely."""

import os
import warnings
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional, Tuple

import astropy
import cv2
import numpy as np
from astropy.visualization import ImageNormalize

STRETCH_FUNCTIONS = {
    "BaseStretch": astropy.visualization.stretch.BaseStretch,
    "LinearStretch": astropy.visualization.stretch.LinearStretch,
    "SqrtStretch": astropy.visualization.stretch.SqrtStretch,
    "PowerStretch": astropy.visualization.stretch.PowerStretch,
    "PowerDistStretch": astropy.visualization.stretch.PowerDistStretch,
    "SquaredStretch": astropy.visualization.stretch.SquaredStretch,
    "LogStretch": astropy.visualization.stretch.LogStretch,
    "AsinhStretch": astropy.visualization.stretch.AsinhStretch,
    "SinhStretch": astropy.visualization.stretch.SinhStretch,
    "HistEqStretch": astropy.visualization.stretch.HistEqStretch,
    "ContrastBiasStretch": astropy.visualization.stretch.ContrastBiasStretch,
}

INTERVAL_FUNCTIONS = {
    "ManualInterval": astropy.visualization.interval.ManualInterval,
    "MinMaxInterval": astropy.visualization.interval.MinMaxInterval,
    "AsymmetricPercentileInterval": astropy.visualization.interval.AsymmetricPercentileInterval,
    "ZScaleInterval": astropy.visualization.interval.ZScaleInterval,
}

# Mirror mode has no effect when white percentile is at full scale (100).
PERCENTILE_WHITE_FULL_FOR_MIRROR_SKIP = 100


@dataclass
class RescaleConfig:
    percentile_black: float = 0.1
    percentile_white: float = 0.9
    background_color: int = 0
    replace_below_black: Optional[int] = None
    replace_above_white: Optional[int] = None
    mirror_white_overflow: bool = False


@dataclass
class ProcessingConfig:
    """Holds all parameters for a single image processing variant."""

    percentile_black: float
    percentile_white: float
    background_color: int
    replace_below_black: Optional[int]
    replace_above_white: Optional[int]
    stretch_function: str
    interval_function: str
    mirror_white_overflow: bool


def unpack_and_run_worker(worker_args: Tuple[ProcessingConfig, str, str, tuple, np.dtype]) -> Optional[Exception]:
    """Helper function to unpack arguments and call the main worker."""
    config = worker_args[0]
    output_dir = worker_args[1]
    shm_name, shape, dtype = worker_args[-3:]
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    data_to_process = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    result = _worker(config, output_dir, data_to_process)
    existing_shm.close()
    return result


def get_normalized_images(data, stretch_function: str = "AsinhStretch", interval_function: str = "ZScaleInterval") -> np.ndarray:
    if stretch_function not in STRETCH_FUNCTIONS:
        print(f"{interval_function=} is not a valid stretch function. Available stretch functions are: {STRETCH_FUNCTIONS.keys()}. Using default AsinhStretch")
        stretch_function = "AsinhStretch"
    if interval_function not in INTERVAL_FUNCTIONS:
        print(f"{interval_function=} is not a valid interval function. Available interval functions are: {INTERVAL_FUNCTIONS.keys()}. Using default ZScaleInterval")
        interval_function = "ZScaleInterval"
    norm = ImageNormalize(data, interval=INTERVAL_FUNCTIONS[interval_function](), stretch=STRETCH_FUNCTIONS[stretch_function]())
    return norm(data)


def rescale_image_to_uint(source_data: np.ndarray, config: RescaleConfig) -> np.ndarray:
    """Rescales a float image to uint8, with options to replace out-of-band values."""
    nan_mask = np.isnan(source_data)
    with warnings.catch_warnings():  # Afaik ignoring the masked values is what we want
        warnings.filterwarnings("ignore", message=".*'partition' will ignore the 'mask' of the MaskedArray.*")
        black_level = np.nanpercentile(source_data, config.percentile_black)
        white_level = np.nanpercentile(source_data, config.percentile_white)
    if white_level <= black_level:
        uint8_image = np.full(source_data.shape, config.background_color, dtype=np.uint8)
        return uint8_image
    temp_float = np.empty(source_data.shape, dtype=np.float32)
    # Perform all arithmetic and clipping operations in-place on the temp array.
    denominator = white_level - black_level
    if denominator == 0:
        denominator = 1

    np.subtract(source_data, black_level, out=temp_float)
    np.divide(temp_float, denominator, out=temp_float)
    if getattr(config, "mirror_white_overflow", False):
        over_white_mask = temp_float > 1.0
        # In-place reflection: 2.0 - Value
        temp_float[over_white_mask] = 2.0 - temp_float[over_white_mask]
        # Clip values that reflected so far they went below 0 (originally > 2.0)
        # This effectively fades bright cores to black.
        np.clip(temp_float, 0, None, out=temp_float)
    np.multiply(temp_float, 255, out=temp_float)
    np.clip(temp_float, 0, 255, out=temp_float)
    np.nan_to_num(temp_float, copy=False, nan=config.background_color)
    uint8_image = temp_float.astype(np.uint8)
    del temp_float

    if config.replace_below_black is not None:
        below_mask = (~nan_mask) & (source_data < black_level)
        uint8_image[below_mask] = config.replace_below_black
    if config.replace_above_white is not None and not getattr(config, "mirror_white_overflow", False):
        above_mask = source_data > white_level
        uint8_image[above_mask] = config.replace_above_white
    uint8_image[nan_mask] = config.background_color
    return uint8_image


def process_data(raw_data: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """
    Applies normalization and rescaling to raw FITS data based on a config.
    """
    normalized_data = get_normalized_images(raw_data, config.stretch_function, config.interval_function)
    normalized_data = normalized_data.astype(np.float32)  # prevent upcasting
    return process_data_from_normalized(normalized_data, config)


def process_data_from_normalized(normalized_data: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Rescale only; caller must run `get_normalized_images` for this (stretch, interval) pair."""
    normalized_data = normalized_data.astype(np.float32)
    rescale_cfg = RescaleConfig(
        percentile_black=config.percentile_black,
        percentile_white=config.percentile_white,
        background_color=config.background_color,
        replace_below_black=config.replace_below_black,
        replace_above_white=config.replace_above_white,
        mirror_white_overflow=config.mirror_white_overflow,
    )
    return rescale_image_to_uint(normalized_data, rescale_cfg)


def _worker(config: ProcessingConfig, output_dir: str, data_to_process: np.ndarray) -> Optional[Exception]:
    """
    A single unit of work. Processes and saves one image based on a ProcessingConfig.
    """
    try:
        mir_suffix = "_mir" if config.mirror_white_overflow else ""
        filename = f"b{config.percentile_black}_w{config.percentile_white}_nan{config.background_color}_bb{config.replace_below_black}_aw{config.replace_above_white}_{config.stretch_function[:-7]}_{config.interval_function[:-8]}{mir_suffix}.png"
        full_outpath = os.path.join(output_dir, filename)
        if not os.path.exists(full_outpath):
            processed_data = process_data(data_to_process, config)
            cv2.imwrite(full_outpath, processed_data)
        return None  # Indicate success
    except Exception as e:
        print(f"Failed on config {config}: {e}")
        return e


def unpack_and_run_worker_from_normalized(worker_args: Tuple[ProcessingConfig, str, str, tuple, np.dtype]) -> Optional[Exception]:
    """Worker entry when shared memory holds float32 `get_normalized_images` output (not raw FITS)."""
    config = worker_args[0]
    output_dir = worker_args[1]
    shm_name, shape, dtype = worker_args[-3:]
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    normalized_data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    result = _worker_from_normalized(config, output_dir, normalized_data)
    existing_shm.close()
    return result


def _worker_from_normalized(config: ProcessingConfig, output_dir: str, normalized_data: np.ndarray) -> Optional[Exception]:
    try:
        mir_suffix = "_mir" if config.mirror_white_overflow else ""
        filename = f"b{config.percentile_black}_w{config.percentile_white}_nan{config.background_color}_bb{config.replace_below_black}_aw{config.replace_above_white}_{config.stretch_function[:-7]}_{config.interval_function[:-8]}{mir_suffix}.png"
        full_outpath = os.path.join(output_dir, filename)
        if not os.path.exists(full_outpath):
            processed_data = process_data_from_normalized(normalized_data, config)
            cv2.imwrite(full_outpath, processed_data)
        return None
    except Exception as e:
        print(f"Failed on config {config}: {e}")
        return e
