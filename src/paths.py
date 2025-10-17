# src/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs" 
DOWNLOAD_DIR = PROJECT_ROOT / OUTPUT_DIR/ "download"
CSV_DIR = PROJECT_ROOT / OUTPUT_DIR/ "csv"
EXTRACTED_PNG_DIR = PROJECT_ROOT / OUTPUT_DIR/ "extracted_png"
COLOR_IMAGE = PROJECT_ROOT / OUTPUT_DIR/ "color_image"
