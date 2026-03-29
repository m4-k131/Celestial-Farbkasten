"""Experimental: rescale + PNG encode on TensorFlow (GPU when available). Astropy normalization stays on CPU."""

import os
import traceback
import warnings
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(
        "TensorFlow failed to import (pip install -r requirements_experimental_gpu.txt; on Windows ensure MSVC runtime / compatible CPU). "
        f"Original: {e!r}"
    ) from e

from lib.extract_png_worker import ProcessingConfig, RescaleConfig, get_normalized_images

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass


def _gpu_device_name() -> str:
    if tf.config.list_physical_devices("GPU"):
        return "/GPU:0"
    return "/CPU:0"


def _rescale_image_to_uint_tf_on_device(source_data: np.ndarray, config: RescaleConfig, device: str) -> np.ndarray:
    """All masking in float32; single uint8 cast at end (avoids GPU SelectV2 on uint8, which often JIT-fails)."""
    nan_mask_np = np.isnan(source_data)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'partition' will ignore the 'mask' of the MaskedArray.*")
        black_level = float(np.nanpercentile(source_data, config.percentile_black))
        white_level = float(np.nanpercentile(source_data, config.percentile_white))
    if white_level <= black_level:
        return np.full(source_data.shape, config.background_color, dtype=np.uint8)
    denominator = float(white_level - black_level)
    if denominator == 0.0:
        denominator = 1.0
    with tf.device(device):
        x = tf.constant(source_data, dtype=tf.float32)
        nan_mask = tf.constant(nan_mask_np, dtype=tf.bool)
        bg_f = tf.cast(config.background_color, tf.float32)
        temp = (x - black_level) / denominator
        if getattr(config, "mirror_white_overflow", False):
            over = temp > 1.0
            temp = tf.where(over, 2.0 - temp, temp)
            temp = tf.maximum(temp, 0.0)
        temp = temp * 255.0
        temp = tf.clip_by_value(temp, 0.0, 255.0)
        temp = tf.where(tf.math.is_nan(temp), bg_f, temp)
        if config.replace_below_black is not None:
            below = tf.logical_and(tf.logical_not(nan_mask), x < black_level)
            rb = tf.cast(config.replace_below_black, tf.float32)
            temp = tf.where(below, rb, temp)
        if config.replace_above_white is not None and not getattr(config, "mirror_white_overflow", False):
            above = x > white_level
            ra = tf.cast(config.replace_above_white, tf.float32)
            temp = tf.where(above, ra, temp)
        temp = tf.where(nan_mask, bg_f, temp)
        temp = tf.clip_by_value(temp, 0.0, 255.0)
        return tf.cast(temp, tf.uint8).numpy()


def rescale_image_to_uint_tf(source_data: np.ndarray, config: RescaleConfig) -> np.ndarray:
    return _rescale_image_to_uint_tf_on_device(source_data, config, _gpu_device_name())


def rescale_only_experimental_gpu(normalized_data: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """TF rescale + encode path when `get_normalized_images` was already run for this (stretch, interval)."""
    rescale_cfg = RescaleConfig(
        percentile_black=config.percentile_black,
        percentile_white=config.percentile_white,
        background_color=config.background_color,
        replace_below_black=config.replace_below_black,
        replace_above_white=config.replace_above_white,
        mirror_white_overflow=config.mirror_white_overflow,
    )
    return rescale_image_to_uint_tf(normalized_data, rescale_cfg)


def process_data_experimental_gpu(raw_data: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    normalized_data = get_normalized_images(raw_data, config.stretch_function, config.interval_function)
    normalized_data = normalized_data.astype(np.float32)
    return rescale_only_experimental_gpu(normalized_data, config)


def write_png_tf(path: str, gray_uint8: np.ndarray) -> None:
    dev = _gpu_device_name()
    with tf.device(dev):
        t = tf.expand_dims(tf.constant(gray_uint8, dtype=tf.uint8), -1)
        encoded = tf.io.encode_png(t)
    tf.io.write_file(path, encoded)


def run_single_png_task_gpu(
    config: ProcessingConfig,
    output_dir: str,
    data_to_process: np.ndarray,
    normalized_float32: Optional[np.ndarray] = None,
) -> Optional[Exception]:
    try:
        mir_suffix = "_mir" if config.mirror_white_overflow else ""
        filename = f"b{config.percentile_black}_w{config.percentile_white}_nan{config.background_color}_bb{config.replace_below_black}_aw{config.replace_above_white}_{config.stretch_function[:-7]}_{config.interval_function[:-8]}{mir_suffix}.png"
        full_outpath = os.path.join(output_dir, filename)
        if not os.path.exists(full_outpath):
            if normalized_float32 is None:
                processed_data = process_data_experimental_gpu(data_to_process, config)
            else:
                processed_data = rescale_only_experimental_gpu(normalized_float32, config)
            write_png_tf(full_outpath, processed_data)
        return None
    except Exception as e:
        if os.environ.get("ASTRO_TF_DEBUG"):
            traceback.print_exc()
        print(f"Failed on config {config}: {e}")
        return e
