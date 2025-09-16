import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch
from skimage.transform import AffineTransform, warp
import astroalign
import astropy
import numpy as np 
import os 
import cv2 

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
    'BaseInterval': astropy.visualization.interval.BaseInterval,
    'ManualInterval': astropy.visualization.interval.ManualInterval,
    'MinMaxInterval': astropy.visualization.interval.MinMaxInterval,
    'AsymmetricPercentileInterval': astropy.visualization.interval.AsymmetricPercentileInterval,
    'PercentileInterval': astropy.visualization.interval.PercentileInterval,
    'ZScaleInterval': astropy.visualization.interval.ZScaleInterval,
}
DEFUALT_PROCESSING_PARAMS = [

    {
    "percentile_black": [20, 40, 60, 80],
    "percentile_white": [90, 95, 97, 98, 99, 100],
    "background_color": [0],
    "replace_below_black": [0],
    "replace_above_white": [0, 255],
    }
]
 
PIPELINE_PARAMS = {
    "C:/Users/malte/Code/astro_project/M-74/jw02107-o039_t018_miri_f770w_i2d.fits": {
   "fits_indices":[1],
   "processing_params": DEFUALT_PROCESSING_PARAMS
    },
    "C:/Users/malte/Code/astro_project/M-74/jw02107-o039_t018_miri_f1000w_i2d.fits": {
    "fits_indices":[1],
    "processing_params": DEFUALT_PROCESSING_PARAMS
    },
    "C:/Users/malte/Code/astro_project/M-74/jw02107-o039_t018_miri_f1130w_i2d.fits": {
    "fits_indices":[1],
    "processing_params": DEFUALT_PROCESSING_PARAMS
    },
    "C:/Users/malte/Code/astro_project/M-74/jw02107-o039_t018_miri_f2100w_i2d.fits": {
    "fits_indices":[1],
    "processing_params": DEFUALT_PROCESSING_PARAMS
    },
}

def load_fits_data(fits_path, index=1):
    with fits.open(fits_path) as hdul:
        # Assuming the image data is in the primary HDU
        data = hdul[index].data
    return data 

def get_normalized_images(data, plot_normalized=False):
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
    if plot_normalized:
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(data, origin='lower', cmap='gray', norm=norm)
        # Add a colorbar to see the data values
        fig.colorbar(im)
        plt.show()
    return norm(data)


import numpy as np

import numpy as np

def rescale_image_to_uint(source_data, percentile_black=1.0, percentile_white=99.0, background_color=0, replace_below_black=None, replace_above_white=None):
    """
    Rescales a float image to uint8, with options to replace out-of-band values.
    """
    # 1. Handle NaNs and calculate levels.
    nan_mask = np.isnan(source_data)
    black_level = np.nanpercentile(source_data, percentile_black)
    white_level = np.nanpercentile(source_data, percentile_white)

    if white_level <= black_level:
        uint8_image = np.full(source_data.shape, background_color, dtype=np.uint8)
        return uint8_image

    # 2. Perform scaling.
    rescaled_float = 255 * (source_data - black_level) / (white_level - black_level)
    
    # 3. Clip the scaled values. The array still contains NaNs at this point.
    clipped_float = np.clip(rescaled_float, 0, 255)

    # --- THE FIX: Explicitly handle NaNs before casting to integer ---
    # Replace any NaN values in the float array with 0. This makes the cast safe.
    np.nan_to_num(clipped_float, copy=False, nan=0)
    # --- END FIX ---

    # 4. Now, this conversion is safe and will not produce a warning.
    uint8_image = clipped_float.astype(np.uint8)

    # 5. Overwrite clipped values if replacement is requested.
    if replace_below_black is not None:
        below_mask = source_data < black_level
        uint8_image[below_mask] = replace_below_black

    if replace_above_white is not None:
        above_mask = source_data > white_level
        uint8_image[above_mask] = replace_above_white

    # 6. Handle NaNs last to ensure the correct background_color is set.
    uint8_image[nan_mask] = background_color

    return uint8_image

def get_transformation(source_data, reference_data):
    transformation, (source_list, target_list) = astroalign.find_transform(source_data, reference_data)
    # transformation.params will give you a 3x3 NumPy array.  matrix, scale, rotation, and translation.
    return transformation.params

def apply_transormation(source_data, transformation_params, output_shape):
    tform = AffineTransform(matrix=transformation_params)
    transformed_data = warp(source_data, inverse_map=tform.inverse, preserve_range=True, output_shape=output_shape, cval=0)
    return transformed_data

def process_data(raw_data, percentile_black=0.1, percentile_white=0.9 , background_color = 0, replace_below_black=None, replace_above_white=None):
    normalized_data = get_normalized_images(raw_data)
    rescaled_image = rescale_image_to_uint(normalized_data, percentile_black, percentile_white, background_color, replace_below_black, replace_above_white)
    return rescaled_image


from astroalign import MaxIterError

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
    for filepath, params in dict_to_process.items():
        print(f"Source: {filepath}")
        transformations[filepath] = {}
        try:
            # Assuming the first index is always valid for loading
            source_fits = load_fits_data(filepath, params["fits_indices"][0])
        except Exception as e:
            print(f"  Could not load source FITS {filepath}: {e}")
            continue

        for other_filepath, other_params in dict_to_process.items():
            if filepath == other_filepath:
                transformations[filepath][other_filepath] = {'scale': (1.0, 1.0), 'translation': (0.0, 0.0), 'rotation': 0.0}
                continue
            print(f"  Target: {other_filepath}")
            try:
                target_fits = load_fits_data(other_filepath, other_params["fits_indices"][0])
                transformation, (source_list, target_list) = astroalign.find_transform(
                    source_fits, 
                    target_fits,
                    detection_sigma=3.0,  # Lowered to find more (potentially fainter) sources
                    min_area=4,           # Helps eliminate single hot pixels/cosmic rays
                    max_control_points=150 # Use more stars for matching
                )
                print(f"    -> Found transformation with {len(source_list)} matching pairs.")
                transformations[filepath][other_filepath] = transformation.params

            except MaxIterError:
                print("    -> ERROR: Could not find transformation. Max iterations reached.")
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

            # === MODIFICATION START ===
            # Check if params is a dictionary (for identity transforms)
            if isinstance(params, dict) and 'translation' in params:
                tx, ty = params['translation']
            # Check if it's a NumPy array (for all other transforms)
            elif isinstance(params, np.ndarray) and params.shape == (3, 3):
                tx = params[0, 2]  # Translation in x is the 3rd element of the 1st row
                ty = params[1, 2]  # Translation in y is the 3rd element of the 2nd row
            # === MODIFICATION END ===

            if tx is not None and ty is not None:
                distance = np.sqrt(tx**2 + ty**2)
                total_distance += distance
            else:
                print(f"Warning: Could not extract translation from {other_path} to {ref_path}")
                total_distance += np.inf

        centrality_scores[ref_path] = total_distance

    if not centrality_scores:
        return None

    # For debugging, you can print the scores
    print("\nCentrality Scores (lower is better):")
    for path, score in centrality_scores.items():
        # A bit of string manipulation to make the output readable
        filename = path.split('/')[-1]
        print(f"  - {filename:<45}: {score:.2f}")

    best_reference = min(centrality_scores, key=centrality_scores.get)
    
    return best_reference
            


def process_dictionary(dict_to_process, outpath):
    #todo: less lines for params after params are finished

    transformations = find_transformations(dict_to_process)
    best_ref_filepath = find_best_reference_image(transformations)

    reference_data = load_fits_data(best_ref_filepath, dict_to_process[best_ref_filepath]["fits_indices"][0])
    reference_shape = reference_data.shape

    for filepath, params in dict_to_process.items():
        dirname=os.path.basename(filepath).split(".")[0]
        os.makedirs(os.path.join(outpath, dirname), exist_ok=True)
        print(filepath, params) #DEBUG
        for index in params["fits_indices"]:
            print("index", index)
            raw_data = load_fits_data(filepath, index)
            if filepath != best_ref_filepath:
                transformation_params = transformations[filepath][best_ref_filepath]
                raw_data = apply_transormation(raw_data, transformation_params, output_shape=reference_shape)
            print("raw data shape", raw_data.shape)
            processing_params = params["processing_params"]
            #yeah, maybe a few too many loops
            for processing_param in processing_params:
                for percentile_black in processing_param["percentile_black"]:
                    for percentile_white in processing_param["percentile_white"]:
                        for background_color in processing_param["background_color"]:
                            for replace_below_black in processing_param["replace_below_black"]:
                                for replace_above_white in processing_param["replace_above_white"]:
                                    processed_data = process_data(raw_data, percentile_black, percentile_white, background_color, replace_below_black, replace_above_white)
                                    filename = f"b{percentile_black}_w{percentile_white}_nan{background_color}_bb{replace_below_black}_aw{replace_above_white}.png"
                                    cv2.imwrite(os.path.join(outpath, dirname, filename), processed_data)


def main():
    outpath="C:/Users/malte/Code/astro_project/m74out"
    dict_to_process=PIPELINE_PARAMS
    process_dictionary(dict_to_process, outpath)

#main()

