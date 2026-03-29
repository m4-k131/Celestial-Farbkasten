"""Experimental entry: PNG variants via TensorFlow GPU rescale/encode (single-process; no ProcessPoolExecutor).

Why GPU usage stays low: Astropy ImageNormalize + ZScale (CPU) dominates wall time; TF only runs cheap
elementwise rescale + PNG encode. The CPU pipeline uses ProcessPoolExecutor across cores; this path is
single-process. We cache one normalized float32 image per (stretch_function, interval_function) so
sweeps over percentiles/mirror do not repeat ImageNormalize.
"""

import argparse
import gc
import itertools
import json
import os
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from reproject import reproject_interp
from tqdm import tqdm

from extract_png_from_fits import load_fits_data, setup_alignment_reference
from lib.extract_png_worker import ProcessingConfig, get_normalized_images
from lib.extract_png_worker_experimental_gpu import run_single_png_task_gpu
from paths import EXTRACTED_PNG_DIR


def process_and_save_pngs_experimental_gpu(data_to_process: np.ndarray, processing_params: Union[str, dict], output_dir: str, overwrite: bool = False) -> None:
    for param_item in processing_params:
        print(param_item)
        loaded_param_set = param_item
        if isinstance(param_item, str) and param_item.endswith(".json"):
            with open(param_item, encoding="utf-8") as f:
                loaded_param_set = json.load(f)
        if not isinstance(loaded_param_set, dict):
            print(f"Warning: Invalid processing parameter set found. Skipping: {loaded_param_set}")
            continue
        mirror_options = loaded_param_set.get("mirror_white_overflow", [False])
        all_params = itertools.product(
            loaded_param_set["percentile_black"],
            loaded_param_set["percentile_white"],
            loaded_param_set["background_color"],
            loaded_param_set["replace_below_black"],
            loaded_param_set["replace_above_white"],
            loaded_param_set["stretch_function"],
            loaded_param_set["interval_function"],
            mirror_options,
        )
        valid_params = [p for p in all_params if p[1] > p[0]]
        tasks_to_run: List[ProcessingConfig] = []
        for params_tuple in valid_params:
            p_b, p_w, bg, r_bb, r_aw, stretch_fn, interval_fn, mirror_bool = params_tuple
            if mirror_bool and p_w == 100:
                continue
            config = ProcessingConfig(
                percentile_black=p_b,
                percentile_white=p_w,
                background_color=bg,
                replace_below_black=r_bb,
                replace_above_white=r_aw,
                stretch_function=stretch_fn,
                interval_function=interval_fn,
                mirror_white_overflow=mirror_bool,
            )
            suffix = "_mir" if mirror_bool else ""
            filename = f"b{p_b}_w{p_w}_nan{bg}_bb{r_bb}_aw{r_aw}_{stretch_fn[:-7]}_{interval_fn[:-8]}{suffix}.png"
            full_outpath = os.path.join(output_dir, filename)
            if overwrite or not os.path.exists(full_outpath):
                tasks_to_run.append(config)
        if not tasks_to_run:
            print("   -> All PNGs already exist. Skipping generation.")
            continue
        by_stretch_interval: dict[Tuple[str, str], List[ProcessingConfig]] = defaultdict(list)
        for cfg in tasks_to_run:
            by_stretch_interval[(cfg.stretch_function, cfg.interval_function)].append(cfg)
        tasks_with_norm: list[tuple[ProcessingConfig, np.ndarray]] = []
        for (stretch_fn, interval_fn), group_cfgs in by_stretch_interval.items():
            normalized_float32 = get_normalized_images(data_to_process, stretch_fn, interval_fn).astype(np.float32)
            for cfg in group_cfgs:
                tasks_with_norm.append((cfg, normalized_float32))
        for config, normalized_float32 in tqdm(tasks_with_norm, desc="   -> Generating PNGs (TF experimental GPU)"):
            run_single_png_task_gpu(config, output_dir, data_to_process, normalized_float32=normalized_float32)


def process_dictionary_wcs_experimental_gpu(dict_to_process: dict, outpath: Optional[str] = None, no_matching: bool = False, overwrite: bool = False) -> None:
    if outpath is None:
        outpath = EXTRACTED_PNG_DIR
    best_ref_filepath, reference_header, reference_shape = (None, None, None)
    if not no_matching:
        best_ref_filepath, reference_header, reference_shape = setup_alignment_reference(dict_to_process)
        if best_ref_filepath is None:
            return
    print("\n--- 2. Processing and Aligning Individual Files (experimental TF GPU) ---")
    for filepath, params in dict_to_process["fits_files"].items():
        print(f"Processing {os.path.basename(filepath)}")
        observation_name = os.path.basename(filepath).split("_")[0]
        dirname = os.path.basename(filepath).split(".")[0]
        image_output_dir = os.path.join(outpath, observation_name, dirname)
        os.makedirs(image_output_dir, exist_ok=True)
        for index in params["fits_indices"]:
            data_to_process = None
            is_reference = filepath == best_ref_filepath
            if no_matching or is_reference:
                print(f"  -> Loading HDU {index} directly.")
                data_to_process = load_fits_data(filepath, index)
            else:
                print(f"  -> Reprojecting HDU {index} to reference...")
                with fits.open(filepath) as src_hdul:
                    data_to_process, _ = reproject_interp(src_hdul[index], reference_header, shape_out=reference_shape)
            if data_to_process is not None:
                process_and_save_pngs_experimental_gpu(data_to_process, params["processing_params"], image_output_dir, overwrite)
                del data_to_process
                gc.collect()
            else:
                print(f"ERROR: Failed to load or align data for HDU {index}. Skipping.")
    print("\nProcessing complete.")


def main(input_json: str, outdir: Optional[str] = None, no_matching: bool = False, overwrite: bool = False) -> None:
    with open(input_json, encoding="utf-8") as f:
        dict_to_process = json.load(f)
    process_dictionary_wcs_experimental_gpu(dict_to_process, outdir, no_matching, overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PNGs with TensorFlow GPU rescale/encode (experimental).")
    parser.add_argument("input_json", type=str)
    parser.add_argument("--outdir", required=False, type=str)
    parser.add_argument("--no_matching", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.input_json, args.outdir, args.no_matching, args.overwrite)
