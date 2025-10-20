import os
import json
import glob
import argparse


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


def generate_combine_config(target_dir, output_file=None):
    final_json = {
        "operand":  "+",
        "images": []}
    for folder in sorted(os.listdir(target_dir)):
        if os.path.isdir(os.path.join(target_dir, folder)):
            image_dict = {"path": os.path.join(target_dir, folder, "b5_w100_nan0_bb0_aw255_Asinh_ZScale.png"),  # jw01783-o001
                          "color": [255, 255, 255],
                          "factor": 1/len(os.listdir(target_dir))
                          }
            final_json["images"].append(image_dict)
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(final_json, f, indent=4)
        print(f"Configuration saved to {output_file}")
    else:
        print(json.dumps(final_json, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir")
    parser.add_argument("--output_file")
    parser.add_argument("--combine_config", action="store_true")
    parser.add_argument("--fits_config", action="store_true")
    args = parser.parse_args()
    if args.combine_config:
        generate_combine_config(args.target_dir, args.output_file)
    if args.fits_config:
        generate_fits_config(args.target_dir, args.output_file)
