import os
import json
import argparse
from pathlib import Path

# --- Color Definitions ---
COLORS = {
    "$CosmicGold": (20, 150, 255),
    "$DeepSpaceBlue": (100, 30, 20),
    "$NebulaMagenta": (200, 40, 180),
    "$CyanGas": (220, 200, 0),
    "$Starlight": (150, 223, 255),
    "$RoyalVoid": (130, 0, 75),
    "$OxidizedRust": (20, 90, 200),
    "$OxygenTeal": (160, 180, 40),
    "$PaleHotYellow": (205, 250, 255),
    "$DeepCrimson": (30, 10, 150),
    "$SunsetOrange": (0, 120, 255),
    "$ElectricViolet": (211, 0, 148),
    "$LuminousMint": (175, 255, 100),
    "$CharcoalVoid": (30, 25, 25),
    "$StellarCrimson": (0, 50, 255),
    "$DeepRuby": (60, 0, 240),
    # Pure, overwhelming red. Subtracts blue and green.
    "$AggressiveHydrogenAlpha": (-150, -150, 255),
    # Intense orange-red for sulphur emissions (SII).
    "$SulphurBurn": (-100, 50, 255),
    # A piercing cyan for oxygen (OIII), subtracting red.
    "$OxygenGlow": (255, 200, -100),
    # Darkens everything it touches, enhancing shadows.
    "$VoidCrusher": (-50, -50, -50),
    # A brighter, more intense gold that suppresses blue.
    "$StarfireGold": (-50, 200, 255),
    # A vibrant teal that removes green, useful for specific nebula gases.
    "$PlasmaTeal": (255, -100, 0),
}

# --- Palette Definitions ---
# Each palette is a list of tuples: (wavelength_keyword, region, color, factor)
PALETTES = {
    "cosmic_embers": [
        ("f140m", "high", "$StellarCrimson", 0.5),
        ("f182m", "high", "$SunsetOrange", 0.9),  # Focus Color
        ("f212n", "low", "$CosmicGold", 0.6),
        ("f277w", "low", "$PaleHotYellow", 0.4),
        ("f300m", "both", "$Starlight", 0.3),
        ("f335m", "both", "$CyanGas", 0.4),
        ("f444w", "low", "$DeepSpaceBlue", 0.7),
    ],
    "gas_and_stars": [
        ("f187n", "high", "$NebulaMagenta", 0.6),
        ("f212n", "high", "$ElectricViolet", 0.5),
        ("f470n", "high", "$DeepRuby", 0.4),
        ("f115w", "low", "$Starlight", 0.3),
        ("f150w", "low", "$PaleHotYellow", 0.5),
        ("f200w", "low", "$CosmicGold", 0.9),  # Focus Color
        ("f277w", "low", "$OxygenTeal", 0.6),
        ("f335m", "low", "$CyanGas", 0.5),
        ("f444w", "low", "$DeepSpaceBlue", 0.7),
    ],
    "golden_nebula": [
        ("f140m", "high", "$DeepCrimson", 0.5),
        ("f182m", "low", "$OxidizedRust", 0.6),
        ("f277w", "low", "$CosmicGold", 0.8),
        ("f335m", "both", "$PaleHotYellow", 0.9),  # Focus Color
        ("f444w", "both", "$Starlight", 0.4),
    ],
}


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
    with open(output_filename, 'w') as f:
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
