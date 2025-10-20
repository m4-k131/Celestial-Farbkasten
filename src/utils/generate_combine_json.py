import json
import argparse
from pathlib import Path

from lib.colors import PALETTES


def get_wavelength_from_folder(folder_name):
    """Extracts the wavelength string from a folder name."""
    try:
        return folder_name.split("_")[-2]
    except IndexError:
        return None


def select_png_for_region(region):
    """Selects PNG file parameters based on the desired intensity region."""
    if region == "high":
        return "b80_w99.5_nan0_bb0_aw255_Asinh_ZScale.png"
    elif region == "low":
        return "b20_w99_nan0_bb0_aw255_Asinh_ZScale.png"
    else:  # 'both' or any other value
        return "b40_w99.7_nan0_bb0_aw255_Asinh_ZScale.png"


def create_combiner_config(palette, png_folders, output_dir, palette_name):
    """Creates a single combiner JSON configuration file."""
    images = []
    for wavelength_keyword, region, color, factor in palette:
        for folder in png_folders:
            folder_path = Path(folder)
            if wavelength_keyword in folder_path.name:
                png_filename = select_png_for_region(region)
                image_path = folder_path / png_filename

                if image_path.exists():
                    images.append({
                        "path": str(image_path),
                        "color": color,
                        "factor": factor
                    })
                else:
                    print(f"Warning: Could not find {image_path}. Skipping.")

    if not images:
        print(
            f"Warning: No images found for palette '{palette_name}'. Skipping config generation.")
        return

    config = {
        "operand": "+",
        "colorspace": "bgr",
        "images": images
    }

    output_filename = output_dir / f"combiner_config_{palette_name}.json"
    with open(output_filename, 'w', encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Successfully created {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate combiner.py configs for a folder of extracted PNGs.")
    parser.add_argument("target_dir", type=Path,
                        help="The directory containing folders of extracted PNGs.")
    parser.add_argument("--output_dir", type=Path, default=Path("."),
                        help="The directory to save the generated JSON configs.")
    args = parser.parse_args()

    if not args.target_dir.is_dir():
        print(f"Error: {args.target_dir} is not a valid directory.")
        return

    args.output_dir.mkdir(exist_ok=True)

    subfolders = [d for d in args.target_dir.iterdir() if d.is_dir()]

    for palette_name, palette in PALETTES.items():
        create_combiner_config(palette, subfolders,
                               args.output_dir, palette_name)


if __name__ == "__main__":
    main()
