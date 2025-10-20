import argparse
import gc
import itertools
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import astropy
import cv2
import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from paths import EXTRACTED_PNG_DIR
from reproject import reproject_interp
from shapely.geometry import Polygon
from skimage.transform import AffineTransform, warp
from tqdm import tqdm

STRETCH_FUNCTIONS = {
    'BaseStretch': astropy.visualization.stretch.BaseStretch,
    'LinearStretch': astropy.visualization.stretch.LinearStretch,
    'SqrtStretch': astropy.visualization.stretch.SqrtStretch,
    'PowerStretch': astropy.visualization.stretch.PowerStretch,
    'PowerDistStretch': astropy.visualization.stretch.PowerDistStretch,
    'SquaredStretch': astropy.visualization.stretch.SquaredStretch,
    'LogStretch': astropy.visualization.stretch.LogStretch,
    'AsinhStretch': astropy.visualization.stretch.AsinhStretch,
    'SinhStretch': astropy.visualization.stretch.SinhStretch,
    'HistEqStretch': astropy.visualization.stretch.HistEqStretch,
    'ContrastBiasStretch': astropy.visualization.stretch.ContrastBiasStretch,
}

INTERVAL_FUNCTIONS = {
    'ManualInterval': astropy.visualization.interval.ManualInterval,
    'MinMaxInterval': astropy.visualization.interval.MinMaxInterval,
    'AsymmetricPercentileInterval': astropy.visualization.interval.AsymmetricPercentileInterval,
    'ZScaleInterval': astropy.visualization.interval.ZScaleInterval,
}

DEFAULT_MATCHING_PARAMS = {
    "detection_sigma": 1.0,
    "min_area": 4,
    "max_control_points": 150
}


def unpack_and_run_worker(worker_args):
    """
    Helper function to unpack arguments and call the main worker.
    This is needed because executor.map passes a single argument.
    """
    shm_name, shape, dtype = worker_args[-3:]
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    data_to_process = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    result = result = _worker(worker_args[0], worker_args[1], data_to_process)
    existing_shm.close()
    return result


def load_fits_data(fits_path, index=1):
    with fits.open(fits_path) as hdul:
        data = hdul[index].data
    data = np.nan_to_num(data, nan=0.0)
    dtype_kind = data.dtype.kind
    native_float32_dtype = np.dtype(f'={dtype_kind}4')
    data = data.astype(native_float32_dtype, copy=True)
    # data += 1e-9 #Not needed anymore?
    return np.ascontiguousarray(data)


def get_normalized_images(data, stretch_function="AsinhStretch", interval_function="ZScaleInterval"):
    if stretch_function not in STRETCH_FUNCTIONS:
        print(f"{interval_function=} is not a valid stretch function. Available stretch functions are: {STRETCH_FUNCTIONS.keys()}. Using default AsinhStretch")
        stretch_function = 'AsinhStretch'
    if interval_function not in INTERVAL_FUNCTIONS:
        print(f"{interval_function=} is not a valid interval function. Available interval functions are: {INTERVAL_FUNCTIONS.keys()}. Using default ZScaleInterval")
        interval_function = 'ZScaleInterval'
    norm = ImageNormalize(data, interval=INTERVAL_FUNCTIONS[interval_function](
    ), stretch=STRETCH_FUNCTIONS[stretch_function]())
    return norm(data)


def rescale_image_to_uint(source_data, percentile_black=1.0, percentile_white=99.0, background_color=0, replace_below_black=None, replace_above_white=None):
    """
    Rescales a float image to uint8, with options to replace out-of-band values.
    """
    nan_mask = np.isnan(source_data)
    with warnings.catch_warnings():  # Afaik ignoring the masked values is what we want
        warnings.filterwarnings(
            'ignore', message=".*'partition' will ignore the 'mask' of the MaskedArray.*")
        black_level = np.nanpercentile(source_data, percentile_black)
        white_level = np.nanpercentile(source_data, percentile_white)
    if white_level <= black_level:
        uint8_image = np.full(
            source_data.shape, background_color, dtype=np.uint8)
        return uint8_image
    temp_float = np.empty(source_data.shape, dtype=np.float32)

    # Perform all arithmetic and clipping operations in-place on the temp array.
    denominator = white_level - black_level
    if denominator == 0:
        denominator = 1  # Avoid division by zero

    np.subtract(source_data, black_level, out=temp_float)
    np.divide(temp_float, denominator, out=temp_float)
    np.multiply(temp_float, 255, out=temp_float)
    np.clip(temp_float, 0, 255, out=temp_float)
    np.nan_to_num(temp_float, copy=False, nan=background_color)
    uint8_image = temp_float.astype(np.uint8)
    del temp_float

    if replace_below_black is not None:
        below_mask = (~nan_mask) & (source_data < black_level)
        uint8_image[below_mask] = replace_below_black
    if replace_above_white is not None:
        above_mask = source_data > white_level
        uint8_image[above_mask] = replace_above_white
    uint8_image[nan_mask] = background_color
    return uint8_image


def apply_transormation(source_data, transformation_params, output_shape):
    tform = AffineTransform(matrix=transformation_params)
    transformed_data = warp(source_data, inverse_map=tform.inverse,
                            preserve_range=True, output_shape=output_shape, cval=0)
    return transformed_data


def process_data(raw_data, percentile_black=0.1, percentile_white=0.9, background_color=0, replace_below_black=None, replace_above_white=None, stretch_function="AsinhStretch", interval_function="ZScaleInterval"):
    normalized_data = get_normalized_images(
        raw_data, stretch_function, interval_function)
    normalized_data = normalized_data.astype(np.float32)  # prevent upcasting
    rescaled_image = rescale_image_to_uint(
        normalized_data, percentile_black, percentile_white, background_color, replace_below_black, replace_above_white)
    return rescaled_image


def _get_wcs_footprint_polygon(filepath, hdu_index=1):
    """Helper function to get the sky footprint of a FITS file as a Shapely Polygon."""
    try:
        with fits.open(filepath) as hdul:
            wcs = WCS(hdul[hdu_index].header)
            # calc_footprint returns the RA/Dec of the corners
            footprint = wcs.calc_footprint()
            return Polygon(footprint)
    except Exception as e:
        print(
            f"Warning: Could not get WCS footprint for {filepath}. Skipping. Error: {e}")
        return None


def find_optimal_reference_image(dict_to_process, hdu_index=1):
    """
    Finds the optimal reference image by first identifying the highest resolution group,
    then finding the image with the best geometric overlap within that group.

    Args:
        dict_to_process (dict): The input dictionary containing filepaths.
        hdu_index (int): The index of the HDU containing the science data and WCS.

    Returns:
        str: The filepath of the optimal reference FITS file.
    """
    filepaths = list(dict_to_process["fits_files"].keys())
    if not filepaths:
        return None
    print("--- 1. Identifying Highest Resolution Group ---")
    # Step 1: Calculate pixel scale for every image
    scales = {}
    for filepath in filepaths:
        try:
            with fits.open(filepath) as hdul:
                wcs = WCS(hdul[hdu_index].header)
                if not wcs.is_celestial:
                    continue
                # proj_plane_pixel_scales returns scale in degrees/pixel
                pixel_scale = np.mean(proj_plane_pixel_scales(wcs))
                scales[filepath] = pixel_scale
                print(
                    f"  - '{os.path.basename(filepath)}': Scale={pixel_scale * 3600:.4f} arcsec/pixel")
        except Exception:
            continue
    if not scales:
        print("Error: Could not determine pixel scales for any images.")
        return None
    # Step 2: Find the maximum scale (lowest resolution) and identify all candidates
    max_pixel_scale = max(scales.values())
    lowest_res_candidates = [
        path for path, scale in scales.items()
        if np.isclose(scale, max_pixel_scale)
    ]
    print(f"\nFound {len(lowest_res_candidates)} candidate(s) at lowest resolution (~{max_pixel_scale * 3600:.4f} arcsec/pixel).")
    if len(lowest_res_candidates) == 1:
        best_reference_path = lowest_res_candidates[0]
        print(
            f"Optimal reference is the sole lowest-resolution image: '{os.path.basename(best_reference_path)}'")
        return best_reference_path
    # --- 2. Optimizing Within the High-Resolution Group ---
    print("\n--- 2. Finding Best Overlap Within Lowest-Res Group ---")
    # Pre-calculate footprints for the candidates
    footprints = {path: _get_wcs_footprint_polygon(
        path, hdu_index) for path in lowest_res_candidates}
    overlap_scores = {}
    for candidate_path in lowest_res_candidates:
        candidate_poly = footprints[candidate_path]
        if candidate_poly is None:
            continue
        total_overlap_area = 0.0
        for other_path in lowest_res_candidates:
            if candidate_path == other_path:
                continue
            other_poly = footprints[other_path]
            if other_poly is None:
                continue
            total_overlap_area += candidate_poly.intersection(other_poly).area
        overlap_scores[candidate_path] = total_overlap_area
        print(
            f"  - Overlap score for {os.path.basename(candidate_path)}: {total_overlap_area:.4f}")
    if not overlap_scores:
        print("Error: Could not calculate overlap scores. Returning first candidate.")
        return lowest_res_candidates[0]
    best_reference_path = max(overlap_scores, key=overlap_scores.get)
    print(
        f"Optimal reference is '{os.path.basename(best_reference_path)}' (best overlap in low-res group).")
    return best_reference_path


def setup_alignment_reference(dict_to_process):
    """
    Finds the best reference FITS file and returns its path, header, and data shape.

    Args:
        dict_to_process (dict): The input dictionary containing FITS file info.

    Returns:
        tuple: A tuple containing (best_ref_filepath, reference_header, reference_shape),
               or (None, None, None) if it fails.
    """
    print("--- 1. Setting Up Alignment Reference ---")
    best_ref_filepath = find_optimal_reference_image(dict_to_process)
    if best_ref_filepath is None:
        print("Error: Could not determine a reference image.")
        return None, None, None
    print(
        f"Using '{os.path.basename(best_ref_filepath)}' as the reference for alignment.")
    # Get the primary science HDU index for the reference file
    ref_hdu_index = dict_to_process["fits_files"][best_ref_filepath]["fits_indices"][0]
    with fits.open(best_ref_filepath) as ref_hdul:
        reference_header = ref_hdul[ref_hdu_index].header
        reference_shape = ref_hdul[ref_hdu_index].data.shape
    return best_ref_filepath, reference_header, reference_shape


def find_best_reference_image(transformations):
    """
    Finds the most central image to use as a reference frame, handling
    both dictionary and NumPy array transformation types.

    Args:
        transformations (dict): A nested dictionary of transformation parameters.

    Returns:
        str: The filepath of the best reference image.
    """
    centrality_scores = {}
    filepaths = list(transformations.keys())
    for ref_path in filepaths:
        total_distance = 0
        for other_path in filepaths:
            if ref_path == other_path:
                continue
            params = transformations[other_path].get(ref_path)
            tx, ty = None, None
            if isinstance(params, dict) and 'translation' in params:
                tx, ty = params['translation']
            elif isinstance(params, np.ndarray) and params.shape == (3, 3):
                # Translation in x is the 3rd element of the 1st row
                tx = params[0, 2]
                # Translation in y is the 3rd element of the 2nd row
                ty = params[1, 2]
            if tx is not None and ty is not None:
                distance = np.sqrt(tx**2 + ty**2)
                total_distance += distance
            else:
                print(
                    f"Warning: Could not extract translation from {other_path} to {ref_path}")
                total_distance += np.inf
        centrality_scores[ref_path] = total_distance
    if not centrality_scores:
        return None
    print("\nCentrality Scores (lower is better):")
    for path, score in centrality_scores.items():
        filename = path.split('/')[-1]
        print(f"  - {filename:<45}: {score:.2f}")
    best_reference = min(centrality_scores, key=centrality_scores.get)
    return best_reference


def _worker(params, output_dir, data_to_process):
    """
    A single unit of work. Processes and saves one image based on params.
    Now receives the reconstructed numpy array.
    """
    p_b, p_w, bg, r_bb, r_aw, stretch_fn, interval_fn = params
    try:
        filename = f"b{p_b}_w{p_w}_nan{bg}_bb{r_bb}_aw{r_aw}_{stretch_fn[:-7]}_{interval_fn[:-8]}.png"
        full_outpath = os.path.join(output_dir, filename)

        # The overwrite check is better handled in the main process
        # before submitting the task, but for simplicity, we can
        # leave it here. If you need to optimize further, you
        # could filter out existing files before creating the task list.
        # Assuming overwrite=False behavior for this example
        if not os.path.exists(full_outpath):
            processed_data = process_data(
                data_to_process, p_b, p_w, bg, r_bb, r_aw, stretch_fn, interval_fn)
            cv2.imwrite(full_outpath, processed_data)
        return None  # Indicate success
    except Exception as e:
        print(f"Failed on params {params}: {e}")
        return e


def process_and_save_pngs(data_to_process, processing_params, output_dir, overwrite=False):
    """
    Takes a single FITS data array and generates all specified PNG variants.

    Args:
        data_to_process (np.ndarray): The aligned FITS data.
        processing_params (list): A list of processing parameter sets (dicts or JSON paths).
        output_dir (str): The directory to save the generated PNG files.
        overwrite (bool): Whether to overwrite existing files.
    """
    shm = shared_memory.SharedMemory(create=True, size=data_to_process.nbytes)
    # Create a NumPy array that uses the shared memory buffer
    shm_np_array = np.ndarray(data_to_process.shape,
                              dtype=data_to_process.dtype, buffer=shm.buf)
    # Copy the FITS data into the shared memory array
    shm_np_array[:] = data_to_process[:]
    try:
        for param_set in processing_params:
            print(param_set)
            if isinstance(param_set, str) and param_set.endswith(".json"):
                with open(param_set, "r", encoding="utf-8") as f:
                    param_set = json.load(f)
            if not isinstance(param_set, dict):
                print(
                    f"Warning: Invalid processing parameter set found. Skipping: {param_set}")
                continue
            all_params = itertools.product(
                param_set["percentile_black"], param_set["percentile_white"],
                param_set["background_color"], param_set["replace_below_black"],
                param_set["replace_above_white"], param_set["stretch_function"],
                param_set["interval_function"]
            )
            valid_params = [p for p in all_params if p[1]
                            > p[0]]  # p[1] is p_w, p[0] is p_b
            # print(valid_params, valid_params)
            tasks_to_run = []
            for params in valid_params:
                p_b, p_w, bg, r_bb, r_aw, stretch_fn, interval_fn = params
                filename = f"b{p_b}_w{p_w}_nan{bg}_bb{r_bb}_aw{r_aw}_{stretch_fn[:-7]}_{interval_fn[:-8]}.png"
                full_outpath = os.path.join(output_dir, filename)
                if overwrite or not os.path.exists(full_outpath):
                    # Pass params, output_dir, and the SHARED MEMORY info, not the actual data
                    tasks_to_run.append(
                        (params, output_dir, shm.name, data_to_process.shape, data_to_process.dtype))
            if not tasks_to_run:
                print("   -> All PNGs already exist. Skipping generation.")
                continue
            with ProcessPoolExecutor() as executor:
                list(tqdm(executor.map(unpack_and_run_worker, tasks_to_run),
                     total=len(tasks_to_run), desc="   -> Generating PNGs"))
    finally:
        shm.close()
        shm.unlink()


def process_dictionary_wcs(dict_to_process, outpath=None, no_matching=False, overwrite=False):
    """
    Processes a dictionary of FITS files, aligning them using WCS and generating PNGs.
    """
    if outpath is None:
        outpath = EXTRACTED_PNG_DIR  # Assuming this is defined elsewhere
    best_ref_filepath, reference_header, reference_shape = (None, None, None)
    if not no_matching:
        best_ref_filepath, reference_header, reference_shape = setup_alignment_reference(
            dict_to_process)
        if best_ref_filepath is None:
            return  # Abort if reference setup failed
    print("\n--- 2. Processing and Aligning Individual Files ---")
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
                    data_to_process, _ = reproject_interp(
                        src_hdul[index], reference_header, shape_out=reference_shape
                    )
            if data_to_process is not None:
                process_and_save_pngs(
                    data_to_process,
                    params["processing_params"],
                    image_output_dir,
                    overwrite
                )
                del data_to_process
                gc.collect()
            else:
                print(
                    f"ERROR: Failed to load or align data for HDU {index}. Skipping.")
    print("\nProcessing complete.")


def main(input_json, outdir=None, no_matching=False, overwrite=False):
    with open(input_json, "r", encoding="utf-8") as f:
        dict_to_process = json.load(f)
    process_dictionary_wcs(dict_to_process, outdir, no_matching, overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--outdir", required=False)
    parser.add_argument("--no_matching", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.input_json, args.outdir, args.no_matching, args.overwrite)
