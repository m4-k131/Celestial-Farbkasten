import os 
import itertools
import numpy as np 
import cv2 
import argparse
import json

from tqdm import tqdm
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch
from skimage.transform import AffineTransform, warp
import astroalign
import astropy

from paths import EXTRACTED_PNG_DIR


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
    "detection_sigma":3.0,
    "min_area":4,
    "max_control_points": 150
}


 
def load_fits_data(fits_path, index=1):
    with fits.open(fits_path) as hdul:
        data = hdul[index].data
    native_data = data.astype(data.dtype.name)
    return native_data


def get_normalized_images(data, stretch_function="AsinhStretch", interval_function="ZScaleInterval"):
    if stretch_function not in STRETCH_FUNCTIONS:
        print(f"{interval_function=} is not a valid stretch function. Available stretch functions are: {STRETCH_FUNCTIONS.keys()}. Using default AsinhStretch")
        stretch_function='AsinhStretch'
    if interval_function not in INTERVAL_FUNCTIONS:
        print(f"{interval_function=} is not a valid interval function. Available interval functions are: {INTERVAL_FUNCTIONS.keys()}. Using default ZScaleInterval")
        interval_function='ZScaleInterval'
    
    norm = ImageNormalize(data, interval=INTERVAL_FUNCTIONS[interval_function](), stretch=STRETCH_FUNCTIONS[stretch_function]())
    return norm(data)


def rescale_image_to_uint(source_data, percentile_black=1.0, percentile_white=99.0, background_color=0, replace_below_black=None, replace_above_white=None):
    """
    Rescales a float image to uint8, with options to replace out-of-band values.
    """
    nan_mask = np.isnan(source_data)
    black_level = np.nanpercentile(source_data, percentile_black)
    white_level = np.nanpercentile(source_data, percentile_white)
    if white_level <= black_level:
        uint8_image = np.full(source_data.shape, background_color, dtype=np.uint8)
        return uint8_image
    
    rescaled_float = 255 * (source_data - black_level) / (white_level - black_level)
    clipped_float = np.clip(rescaled_float, 0, 255)
    np.nan_to_num(clipped_float, copy=False, nan=0)
    uint8_image = clipped_float.astype(np.uint8)

    if replace_below_black is not None:
        below_mask = source_data < black_level
        uint8_image[below_mask] = replace_below_black
    if replace_above_white is not None:
        above_mask = source_data > white_level
        uint8_image[above_mask] = replace_above_white

    uint8_image[nan_mask] = background_color
    return uint8_image


def apply_transormation(source_data, transformation_params, output_shape):
    tform = AffineTransform(matrix=transformation_params)
    transformed_data = warp(source_data, inverse_map=tform.inverse, preserve_range=True, output_shape=output_shape, cval=0)
    return transformed_data


def process_data(raw_data, percentile_black=0.1, percentile_white=0.9 , background_color = 0, replace_below_black=None, replace_above_white=None, stretch_function="AsinhStretch", interval_function="ZScaleInterval"):
    normalized_data = get_normalized_images(raw_data, stretch_function, interval_function)
    rescaled_image = rescale_image_to_uint(normalized_data, percentile_black, percentile_white, background_color, replace_below_black, replace_above_white)
    return rescaled_image


def find_transformations(dict_to_process):
    """
    Finds transformations between all pairs of FITS files.

    Args:
        dict_to_process (dict): A dictionary where keys are filepaths and
                                values are parameter dictionaries.

    Returns:
        dict: A nested dictionary with transformation parameters.
              Returns None for pairs that could not be aligned.
    """
    transformations = {}
    for filepath, params in dict_to_process["fits_files"].items():
        print(f"Source: {filepath}")
        transformations[filepath] = {}
        try:
            # Assuming the first index is always valid for loading
            source_fits = load_fits_data(filepath, params["fits_indices"][0])
        except Exception as e:
            print(f"  Could not load source FITS {filepath}: {e}")
            continue

        for other_filepath, other_params in dict_to_process["fits_files"].items():
            if filepath == other_filepath:
                transformations[filepath][other_filepath] = {'scale': (1.0, 1.0), 'translation': (0.0, 0.0), 'rotation': 0.0}
                continue
            print(f"  Target: {other_filepath}")
            try:
                target_fits = load_fits_data(other_filepath, other_params["fits_indices"][0])
                transformation, (source_list, target_list) = astroalign.find_transform(
                    source_fits, 
                    target_fits,
                    **DEFAULT_MATCHING_PARAMS
                )
                print(f"    -> Found transformation with {len(source_list)} matching pairs.")
                transformations[filepath][other_filepath] = transformation.params

            except astroalign.MaxIterError:
                print("    -> Warning: Could not find transformation. Max iterations reached.")
                transformations[filepath][other_filepath] = None # Store None if no match is found
            except Exception as e:
                print(f"    -> ERROR: An unexpected error occurred: {e}")
                transformations[filepath][other_filepath] = None
    return transformations


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
                tx = params[0, 2]  # Translation in x is the 3rd element of the 1st row
                ty = params[1, 2]  # Translation in y is the 3rd element of the 2nd row
            if tx is not None and ty is not None:
                distance = np.sqrt(tx**2 + ty**2)
                total_distance += distance
            else:
                print(f"Warning: Could not extract translation from {other_path} to {ref_path}")
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
           

def process_dictionary(dict_to_process, outpath=None, no_matching=False, overwrite=False):
    if outpath is None:
        outpath = EXTRACTED_PNG_DIR

    if not no_matching:
        transformations = find_transformations(dict_to_process)
        best_ref_filepath = find_best_reference_image(transformations)

        reference_data = load_fits_data(best_ref_filepath, dict_to_process["fits_files"][best_ref_filepath]["fits_indices"][0])
        reference_shape = reference_data.shape

    for filepath, params in dict_to_process["fits_files"].items():
        print(f"Processing {filepath}")
        observation_name = os.path.basename(filepath).split("_")[0]
        dirname=os.path.basename(filepath).split(".")[0]
        os.makedirs(os.path.join(outpath, observation_name, dirname), exist_ok=True)
        print(filepath, params) #DEBUG
        for index in params["fits_indices"]:
            raw_data = load_fits_data(filepath, index)
            if not no_matching: 
                if filepath != best_ref_filepath:
                    transformation_params = transformations[filepath][best_ref_filepath]
                    raw_data = apply_transormation(raw_data, transformation_params, output_shape=reference_shape)
            print("raw data shape", raw_data.shape)
            for processing_param_set in params["processing_params"]:
                if isinstance(processing_param_set, str):
                    if processing_param_set.endswith(".json"):# and os.path.isfile(processing_param_set):
                        with open(processing_param_set, "r", encoding="utf-8") as f:
                            processing_param_set = json.load(f)
                if not isinstance(processing_param_set, dict): #TODO Validate keys
                    print(f"{processing_param_set} is not a valid processing_param_set")
                    continue

                #print("Param set", processing_param_set)
                percentile_black = processing_param_set["percentile_black"]
                percentile_white = processing_param_set["percentile_white"]
                background_color = processing_param_set["background_color"]
                replace_below_black = processing_param_set["replace_below_black"]
                replace_above_white = processing_param_set["replace_above_white"]
                stretch_functions = processing_param_set["stretch_function"]
                interval_functions = processing_param_set["interval_function"]
                iterator = itertools.product(percentile_black, percentile_white, background_color, replace_below_black, replace_above_white, stretch_functions, interval_functions)
                total_iterations = len(percentile_black) * len(percentile_white) * len(background_color) * len(replace_below_black) * len(replace_above_white) * len(stretch_functions) * len(interval_functions)
                with tqdm(total=total_iterations, desc="Processing Images") as pbar:
                    for p_b, p_w, bg, r_bb, r_aw, stretch_fn, interval_fn in iterator:
                        filename = f"b{p_b}_w{p_w}_nan{bg}_bb{r_bb}_aw{r_aw}_{stretch_fn[:-7]}_{interval_fn[:-8]}.png"
                        full_outpath = os.path.join(outpath, observation_name, dirname, filename)
                        if p_w > p_b:
                            if overwrite or not os.path.isfile(full_outpath):
                                processed_data = process_data(raw_data, p_b, p_w, bg, r_bb, r_aw, stretch_fn, interval_fn)
                                cv2.imwrite(full_outpath, processed_data)
                                pbar.set_postfix({
                                'pb': f'{p_b:.2f}', 
                                'pw': f'{p_w:.2f}', 
                                'bg': bg,
                                'stretch': stretch_fn[:-7],
                                "interval": interval_fn[:-8]
                                })
                                pbar.update(1)
                            else:
                                print(f"{filename} already exists. Run this script with --overwrite to overwrite existing files. Skipping")
                                pbar.total -= 1
                                pbar.refresh()
                        else:
                            pbar.total -= 1
                            pbar.refresh()


def main(input_json, outdir=None, no_matching=False, overwrite=False):
    with open(input_json, "r", encoding="utf-8") as f:
        dict_to_process = json.load(f)
    process_dictionary(dict_to_process, outdir, no_matching, overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--outdir", required=False)
    parser.add_argument("--no_matching", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.input_json, args.outdir, args.no_matching, args.overwrite)

