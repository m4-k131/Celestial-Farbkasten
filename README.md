# Astro-Imaging Scripts

A collection of scripts to query, download, process, and combine astronomical data into a final image.
Create false-color images with a few easy-to-use scripts around **astroquery** and **astropy**. The flow has been tested mainly with **JWST** data from MAST.

## Setup

Install dependencies from the repo root, for example:

```bash
pip install -r requirements.txt
```

Run the pipeline scripts as `python src/<script>.py` from the project root so imports resolve.

---

# Step by step example

To create a false-color image, use the scripts in the following order.
The commands below illustrate the **LDN-1527** example; downloading and extracting can take a while.

## 1. `query_observation_lists.py`

This script queries the MAST archive for observation data for a target and saves the results to CSV. You need a MAST token for querying.

Export **`MAST_API_TOKEN`** in your environment, or create a **`.env`** file next to this `README.md` with:

```env
MAST_API_TOKEN=your_token_here
```

Or enter the token when the script prompts (if the variable is unset).

To create a token, use the [MAST token page](https://auth.mast.stsci.edu/token?suggested_name=Astroquery&suggested_scope=mast:exclusive_access) (same flow as in the script’s help text).

Finding the exact `target_name` can be tricky: for example Messier 74 is queried as **`NGC-628`**. Searching for `"NGC 628"`, `"M-74"`, or `"M74"` may not match. Use the [MAST Portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html) if you need to confirm the name.

Arguments:

- `target_name` (required): celestial object name for the query.
- `--exact_name`: do not wrap the name in wildcards; CSV filename gets an `_actual` suffix (e.g. `LDN-1527_actual.csv`).
- `--calib_level` (optional, default: `3`): calibration level.
- `--project` (optional, default: `JWST`): telescope or project.
- `--outdir` (optional, default: **`outputs/csv`**): output directory for the CSV.

### Example

```bash
python src/query_observation_lists.py LDN-1527 --exact_name
```

---

## 2. `download_from_csv.py`

Downloads FITS and JPEG products listed in the CSV from step 1.

- `csv` (required): path to the CSV.
- `--outdir` (optional): download directory (default: **`outputs/download/<csv_basename>`**).
- `--ignore_jpgs`: skip JPEG previews.
- `--ignore_fits`: skip FITS (useful to browse previews first).
- `--must_contain`: only download if this substring appears in the file URI (helps pick one observation without re-querying).

FITS files are large and slow to download; trying **`--ignore_fits`** first is often practical. Related products are often distinguished by a substring in the filename or URI—use **`--must_contain`** to narrow downloads.

### Examples

```bash
python src/download_from_csv.py outputs/csv/LDN-1527_actual.csv --ignore_fits
python src/download_from_csv.py outputs/csv/LDN-1527_actual.csv
python src/download_from_csv.py outputs/csv/LDN-1527_actual.csv --must_contain=YOUR_SUBSTRING
```

---

## 3. `extract_png_from_fits.py`

Processes and aligns scientific FITS files into 8-bit PNGs. Configuration is **JSON** (which files, HDUs, stretch/interval presets, output layout).

- `input_json` (required): extract config JSON (see **`configs/extract/`** for examples).
- `--outdir` (optional): overrides the default extracted-PNG root (**`outputs/extracted_png`** from `src/paths.py`).
- `--no_matching`: skip cross-image matching where applicable.
- `--overwrite`: allow overwriting existing outputs.

### Example

```bash
python src/extract_png_from_fits.py configs/extract/LDN-1527/LDN-1527_extract_a_few.json
```

---

## 4. `combiner.py`

Combines processed grayscale PNGs into one false-color composite. Configuration is **JSON** (paths, colors, per-layer factors, optional nesting).

- `input_json` (required): combine config JSON (see **`configs/combine/`** for examples).
- `--imagename` (optional): base name for the output PNG (defaults from the input JSON filename).
- `--suffix` (optional): extra suffix on the filename.
- `--outdir` (optional, default: **`outputs/color_image`**).
- `--overwrite`: write the output even if the target file already exists.

The script also writes a sidecar **`.json`** next to the PNG with the resolved configuration.

### Example

```bash
python src/combiner.py configs/combine/LDN-1527/ldn-1527_combined_1.json
```

### JSON configuration

There is no separate schema document yet. Use the checked-in files under **`configs/extract/`** and **`configs/combine/`**, and sample combine JSON under **`dev/`**, as references. Helpers:

- `src/utils/generate_extract_json.py` — scaffold extract (and simple combine) JSON from folders.
- `src/utils/generate_combine_json.py` — build combine configs from **`src/lib/colors.PALETTES`** and extracted PNG folder names.
