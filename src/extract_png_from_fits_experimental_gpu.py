"""Experimental entry: PNG variants via TensorFlow GPU rescale/encode (single-process; no ProcessPoolExecutor).

Why GPU usage stays low: Astropy ImageNormalize + ZScale (CPU) dominates wall time; TF only runs cheap
elementwise rescale + PNG encode. The CPU pipeline uses ProcessPoolExecutor across cores; this path is
single-process. We cache one normalized float32 image per (stretch_function, interval_function) so
sweeps over percentiles/mirror do not repeat ImageNormalize. Variants are processed in batches of
``--batch-size`` (default 16) so one TF graph sees B stacked images per forward pass.
"""

import argparse
import contextlib
import gc
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from reproject import reproject_interp
from tqdm import tqdm

from extract_png_from_fits import load_fits_data, setup_alignment_reference
from lib.extract_benchmark import ExtractBenchmark, default_benchmark_path
from lib.extract_png_worker import PERCENTILE_WHITE_FULL_FOR_MIRROR_SKIP, ProcessingConfig, get_normalized_images
from lib.extract_png_worker_experimental_gpu import gpu_png_output_path, run_png_variant_batch_gpu, run_single_png_task_gpu
from paths import EXTRACTED_PNG_DIR


@dataclass
class ExperimentalGpuPipelineOptions:
    overwrite: bool = False
    bench: Optional[ExtractBenchmark] = None
    batch_size: int = 16


def _collect_tasks_from_loaded_param_set_gpu(loaded_param_set: dict, output_dir: str, overwrite: bool) -> List[ProcessingConfig]:
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
        if mirror_bool and p_w == PERCENTILE_WHITE_FULL_FOR_MIRROR_SKIP:
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
    return tasks_to_run


def _run_stretch_interval_groups_gpu(
    tasks_to_run: List[ProcessingConfig],
    data_to_process: np.ndarray,
    output_dir: str,
    opts: ExperimentalGpuPipelineOptions,
) -> None:
    by_stretch_interval: dict[Tuple[str, str], List[ProcessingConfig]] = defaultdict(list)
    for cfg in tasks_to_run:
        by_stretch_interval[(cfg.stretch_function, cfg.interval_function)].append(cfg)
    bs = max(1, opts.batch_size)
    bench = opts.bench
    overwrite = opts.overwrite
    t_var = time.perf_counter()
    for (stretch_fn, interval_fn), group_cfgs in by_stretch_interval.items():
        t_norm = time.perf_counter()
        normalized_float32 = get_normalized_images(data_to_process, stretch_fn, interval_fn).astype(np.float32)
        if bench:
            bench.record(
                "astropy_image_normalize",
                time.perf_counter() - t_norm,
                f"{stretch_fn}/{interval_fn} variants={len(group_cfgs)}",
            )
        n = len(group_cfgs)
        with tqdm(total=n, desc="   -> Generating PNGs (TF experimental GPU)") as pbar:
            for start in range(0, n, bs):
                batch = group_cfgs[start : start + bs]
                pending = [c for c in batch if overwrite or not os.path.exists(gpu_png_output_path(c, output_dir))]
                if not pending:
                    pbar.update(len(batch))
                    continue
                if len(pending) == 1:
                    run_single_png_task_gpu(pending[0], output_dir, data_to_process, normalized_float32=normalized_float32, overwrite=overwrite)
                else:
                    run_png_variant_batch_gpu(pending, normalized_float32, output_dir)
                pbar.update(len(batch))
    if bench:
        bench.record(
            "tf_rescale_encode_write_png",
            time.perf_counter() - t_var,
            f"variants={len(tasks_to_run)} batch_size={bs}",
        )


def process_and_save_pngs_experimental_gpu(
    data_to_process: np.ndarray,
    processing_params: Union[str, dict],
    output_dir: str,
    opts: ExperimentalGpuPipelineOptions,
) -> None:
    for param_item in processing_params:
        print(param_item)
        loaded_param_set = param_item
        if isinstance(param_item, str) and param_item.endswith(".json"):
            with open(param_item, encoding="utf-8") as f:
                loaded_param_set = json.load(f)
        if not isinstance(loaded_param_set, dict):
            print(f"Warning: Invalid processing parameter set found. Skipping: {loaded_param_set}")
            continue
        tasks_to_run = _collect_tasks_from_loaded_param_set_gpu(loaded_param_set, output_dir, opts.overwrite)
        if not tasks_to_run:
            print("   -> All PNGs already exist. Skipping generation.")
            continue
        _run_stretch_interval_groups_gpu(tasks_to_run, data_to_process, output_dir, opts)


def process_dictionary_wcs_experimental_gpu(
    dict_to_process: dict,
    outpath: Optional[str] = None,
    no_matching: bool = False,
    opts: Optional[ExperimentalGpuPipelineOptions] = None,
) -> None:
    if opts is None:
        opts = ExperimentalGpuPipelineOptions()
    if outpath is None:
        outpath = EXTRACTED_PNG_DIR
    best_ref_filepath, reference_header, reference_shape = (None, None, None)
    if not no_matching:
        with contextlib.nullcontext() if opts.bench is None else opts.bench.span("alignment_setup"):
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
            t_io = time.perf_counter()
            if no_matching or is_reference:
                print(f"  -> Loading HDU {index} directly.")
                data_to_process = load_fits_data(filepath, index)
            else:
                print(f"  -> Reprojecting HDU {index} to reference...")
                with fits.open(filepath) as src_hdul:
                    data_to_process, _ = reproject_interp(src_hdul[index], reference_header, shape_out=reference_shape)
            if opts.bench:
                opts.bench.record(
                    "load_or_reproject",
                    time.perf_counter() - t_io,
                    f"{os.path.basename(filepath)} HDU{index}",
                )
            if data_to_process is not None:
                process_and_save_pngs_experimental_gpu(data_to_process, params["processing_params"], image_output_dir, opts)
                del data_to_process
                gc.collect()
            else:
                print(f"ERROR: Failed to load or align data for HDU {index}. Skipping.")
    print("\nProcessing complete.")


def main(
    input_json: str,
    outdir: Optional[str] = None,
    no_matching: bool = False,
    opts: Optional[ExperimentalGpuPipelineOptions] = None,
) -> None:
    with open(input_json, encoding="utf-8") as f:
        dict_to_process = json.load(f)
    process_dictionary_wcs_experimental_gpu(dict_to_process, outdir, no_matching, opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PNGs with TensorFlow GPU rescale/encode (experimental).")
    parser.add_argument("input_json", type=str)
    parser.add_argument("--outdir", required=False, type=str)
    parser.add_argument("--no_matching", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16, metavar="N", help="TF rescale variants per batched GPU call (default: 16).")
    parser.add_argument("--benchmark", nargs="?", const="", default=None, metavar="FILE", help="Write step timings to FILE (default: outputs/benchmark/extract_gpu_<timestamp>.txt).")
    args = parser.parse_args()
    bench: Optional[ExtractBenchmark] = None
    bench_path: Optional[str] = None
    if args.benchmark is not None:
        bench_path = args.benchmark if args.benchmark else default_benchmark_path("gpu")
        bench = ExtractBenchmark("extract_png_from_fits_experimental_gpu (TF GPU)")
    gpu_opts = ExperimentalGpuPipelineOptions(overwrite=args.overwrite, bench=bench, batch_size=args.batch_size)
    main(args.input_json, args.outdir, args.no_matching, gpu_opts)
    if bench is not None and bench_path is not None:
        bench.write_report(bench_path, input_json=args.input_json, argv=sys.argv)
        print(f"Benchmark report written to {bench_path}")
