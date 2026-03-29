"""Experimental: rescale + PNG encode on TensorFlow (GPU when available). Astropy normalization stays on CPU."""

import os
import traceback
import warnings
from typing import List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(f"TensorFlow failed to import (pip install -r requirements_experimental_gpu.txt; on Windows ensure MSVC runtime / compatible CPU). Original: {e!r}") from e

from lib.extract_png_worker import ProcessingConfig, RescaleConfig, get_normalized_images

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# TF graph uses 0.5 as "bool flag on" for float masks (mir, has_rb, has_ra).
TF_BOOL_MASK_ON = 0.5

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


def _prefilter_active_rescale_cfgs(
    source_data: np.ndarray,
    rescale_cfgs: List[RescaleConfig],
) -> Tuple[List[Optional[np.ndarray]], np.ndarray, List[int], List[RescaleConfig], List[float], List[float]]:
    n = len(rescale_cfgs)
    out: List[Optional[np.ndarray]] = [None] * n
    nan_mask_np = np.isnan(source_data)
    active_idx: List[int] = []
    active_cfgs: List[RescaleConfig] = []
    active_bl: List[float] = []
    active_wl: List[float] = []
    for i, cfg in enumerate(rescale_cfgs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*'partition' will ignore the 'mask' of the MaskedArray.*")
            bl = float(np.nanpercentile(source_data, cfg.percentile_black))
            wl = float(np.nanpercentile(source_data, cfg.percentile_white))
        if wl <= bl:
            out[i] = np.full(source_data.shape, cfg.background_color, dtype=np.uint8)
        else:
            active_idx.append(i)
            active_cfgs.append(cfg)
            active_bl.append(bl)
            active_wl.append(wl)
    return out, nan_mask_np, active_idx, active_cfgs, active_bl, active_wl


def _tf_batched_rescale_stack(
    source_data: np.ndarray,
    nan_mask_np: np.ndarray,
    active_cfgs: List[RescaleConfig],
    active_bl: List[float],
    active_wl: List[float],
) -> np.ndarray:
    B = len(active_cfgs)
    device = _gpu_device_name()
    bls = np.asarray(active_bl, dtype=np.float32).reshape(B, 1, 1)
    wls = np.asarray(active_wl, dtype=np.float32).reshape(B, 1, 1)
    mir = np.asarray([[float(getattr(c, "mirror_white_overflow", False))] for c in active_cfgs], dtype=np.float32).reshape(B, 1, 1)
    has_rb = np.asarray([[1.0 if c.replace_below_black is not None else 0.0] for c in active_cfgs], dtype=np.float32).reshape(B, 1, 1)
    rb_vals = np.asarray([[float(c.replace_below_black) if c.replace_below_black is not None else 0.0] for c in active_cfgs], dtype=np.float32).reshape(B, 1, 1)
    has_ra = np.asarray(
        [[1.0 if (c.replace_above_white is not None and not getattr(c, "mirror_white_overflow", False)) else 0.0] for c in active_cfgs],
        dtype=np.float32,
    ).reshape(B, 1, 1)
    ra_vals = np.asarray([[float(c.replace_above_white) if c.replace_above_white is not None else 0.0] for c in active_cfgs], dtype=np.float32).reshape(B, 1, 1)
    bg_f = np.asarray([[float(c.background_color)] for c in active_cfgs], dtype=np.float32).reshape(B, 1, 1)
    with tf.device(device):
        x = tf.constant(source_data, dtype=tf.float32)
        x_b = tf.tile(tf.expand_dims(x, 0), [B, 1, 1])
        black = tf.constant(bls, dtype=tf.float32)
        white = tf.constant(wls, dtype=tf.float32)
        denom = white - black
        denom = tf.where(tf.equal(denom, 0.0), 1.0, denom)
        base = (x_b - black) / denom
        mir_t = tf.constant(mir, dtype=tf.float32)
        over = base > 1.0
        temp_mir = tf.where(over, 2.0 - base, base)
        temp_mir = tf.maximum(temp_mir, 0.0)
        temp = tf.where(mir_t > TF_BOOL_MASK_ON, temp_mir, base)
        temp = temp * 255.0
        temp = tf.clip_by_value(temp, 0.0, 255.0)
        bg_t = tf.constant(bg_f, dtype=tf.float32)
        temp = tf.where(tf.math.is_nan(temp), bg_t, temp)
        nan_mask_b = tf.tile(tf.expand_dims(tf.constant(nan_mask_np, dtype=tf.bool), 0), [B, 1, 1])
        has_rb_t = tf.constant(has_rb, dtype=tf.float32)
        rb_t = tf.constant(rb_vals, dtype=tf.float32)
        below = tf.logical_and(tf.logical_not(nan_mask_b), x_b < black)
        temp = tf.where(tf.logical_and(below, has_rb_t > TF_BOOL_MASK_ON), rb_t, temp)
        has_ra_t = tf.constant(has_ra, dtype=tf.float32)
        ra_t = tf.constant(ra_vals, dtype=tf.float32)
        above = x_b > white
        temp = tf.where(tf.logical_and(above, has_ra_t > TF_BOOL_MASK_ON), ra_t, temp)
        temp = tf.where(nan_mask_b, bg_t, temp)
        temp = tf.clip_by_value(temp, 0.0, 255.0)
        return tf.cast(temp, tf.uint8).numpy()


def rescale_image_to_uint_tf_batched(source_data: np.ndarray, rescale_cfgs: List[RescaleConfig]) -> List[np.ndarray]:
    """Batched TF rescale: one H2D tile of `source_data`, shared nan mask; CPU nanpercentile per row."""
    out, nan_mask_np, active_idx, active_cfgs, active_bl, active_wl = _prefilter_active_rescale_cfgs(source_data, rescale_cfgs)
    if not active_cfgs:
        return out  # type: ignore[return-value]
    stacked = _tf_batched_rescale_stack(source_data, nan_mask_np, active_cfgs, active_bl, active_wl)
    for k, slot in enumerate(active_idx):
        out[slot] = stacked[k]
    return out  # type: ignore[return-value]


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


def gpu_png_output_path(config: ProcessingConfig, output_dir: str) -> str:
    mir_suffix = "_mir" if config.mirror_white_overflow else ""
    filename = f"b{config.percentile_black}_w{config.percentile_white}_nan{config.background_color}_bb{config.replace_below_black}_aw{config.replace_above_white}_{config.stretch_function[:-7]}_{config.interval_function[:-8]}{mir_suffix}.png"
    return os.path.join(output_dir, filename)


def run_png_variant_batch_gpu(configs: List[ProcessingConfig], normalized_float32: np.ndarray, output_dir: str) -> None:
    """Writes one PNG per config; batched TF rescale then sequential encode+write."""
    rescale_cfgs = [
        RescaleConfig(
            percentile_black=c.percentile_black,
            percentile_white=c.percentile_white,
            background_color=c.background_color,
            replace_below_black=c.replace_below_black,
            replace_above_white=c.replace_above_white,
            mirror_white_overflow=c.mirror_white_overflow,
        )
        for c in configs
    ]
    images = rescale_image_to_uint_tf_batched(normalized_float32, rescale_cfgs)
    for cfg, img in zip(configs, images, strict=True):
        write_png_tf(gpu_png_output_path(cfg, output_dir), img)


def run_single_png_task_gpu(
    config: ProcessingConfig,
    output_dir: str,
    data_to_process: np.ndarray,
    normalized_float32: Optional[np.ndarray] = None,
    overwrite: bool = False,
) -> Optional[Exception]:
    try:
        full_outpath = gpu_png_output_path(config, output_dir)
        if overwrite or not os.path.exists(full_outpath):
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
