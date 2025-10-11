import os
import json
import glob

def generate_fits_config(target_dir, output_file=None):
    """
    Generates a JSON configuration for all .fits files in a directory.

    Args:
        target_dir (str): The path to the directory containing .fits files.
        output_file (str, optional): Path to save the JSON file. 
                                     If None, prints to console.
    """
    default_params = {
        "fits_indices": [1],
        "processing_params": ["configs/extract/__default_extract_a_few.json"]
    }
    search_pattern = os.path.join(target_dir, '*.fits')
    fits_files = glob.glob(search_pattern)
    fits_paths = [path.replace('\\', '/') for path in fits_files]
    fits_files_dict = {path: default_params for path in sorted(fits_paths)}
    final_json = {
        "fits_files": fits_files_dict
    }
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(final_json, f, indent=4)
        print(f"Configuration saved to {output_file}")
    else:
        print(json.dumps(final_json, indent=4))

directory_path = "outputs/download/NGC5236-NIRCAM"
generate_fits_config(directory_path)

