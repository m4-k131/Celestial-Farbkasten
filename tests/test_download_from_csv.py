import os
from unittest.mock import call, patch

import pandas as pd
import pytest

from src.download_from_csv import BASE_URL
from src.download_from_csv import main as download_main

DOWNLOAD_ALL_DOWNLOAD_ASSERTION = 5
DOWNLOAD_ALL_ISFILE_ASSERTION = 5
DOWNLOAD_INGORE_FITS_DOWNLOAD_ASSERTION = 2


@pytest.fixture
def mock_csv_data():
    """Provides a mock pandas DataFrame."""
    dummy_data = {"intentType": ["science", "science", "calibration", "science"], "jpegURL": ["mast:JWST/jpg/jw01.jpg", "mast:JWST/jpg/jw02-filter.jpg", "mast:JWST/jpg/cal.jpg", pd.NA], "dataURL": ["mast:JWST/fits/jw01.fits", "mast:JWST/fits/jw02-filter.fits", "mast:JWST/fits/cal.fits", "mast:JWST/fits/jw03.fits"]}
    return pd.DataFrame(dummy_data)


@pytest.fixture
def mock_dependencies(mock_csv_data):
    """Mocks all external dependencies for the download script."""
    # We patch all dependencies in the 'src.download_from_csv' namespace
    with patch("src.download_from_csv.pd.read_csv", return_value=mock_csv_data) as mock_read_csv, patch("src.download_from_csv.os.makedirs") as mock_mkdir, patch("src.download_from_csv.os.path.isfile", return_value=False) as mock_isfile, patch("src.download_from_csv.download_with_tqdm", return_value=True) as mock_download:
        yield {"read_csv": mock_read_csv, "mkdir": mock_mkdir, "isfile": mock_isfile, "download": mock_download}


def test_download_all(mock_dependencies, tmp_path):
    """
    Tests the default case: downloads FITS and JPGs for 'science' intents.
    """
    test_csv = "dummy.csv"
    test_outdir = str(tmp_path)
    download_main(csv=test_csv, outdir=test_outdir)
    # --- Assertions ---
    mock_dependencies["mkdir"].assert_called_with(test_outdir, exist_ok=True)
    mock_dependencies["read_csv"].assert_called_with(test_csv)
    assert mock_dependencies["isfile"].call_count == DOWNLOAD_ALL_DOWNLOAD_ASSERTION
    assert mock_dependencies["download"].call_count == DOWNLOAD_ALL_ISFILE_ASSERTION
    expected_calls = [call(f"{BASE_URL}mast:JWST/jpg/jw01.jpg", os.path.join(test_outdir, "jw01.jpg")), call(f"{BASE_URL}mast:JWST/fits/jw01.fits", os.path.join(test_outdir, "jw01.fits")), call(f"{BASE_URL}mast:JWST/jpg/jw02-filter.jpg", os.path.join(test_outdir, "jw02-filter.jpg")), call(f"{BASE_URL}mast:JWST/fits/jw02-filter.fits", os.path.join(test_outdir, "jw02-filter.fits")), call(f"{BASE_URL}mast:JWST/fits/jw03.fits", os.path.join(test_outdir, "jw03.fits"))]
    mock_dependencies["download"].assert_has_calls(expected_calls)


def test_download_ignore_fits(mock_dependencies, tmp_path):
    """
    Tests the --ignore_fits flag (download_fits=False).
    """
    test_csv = "dummy.csv"
    test_outdir = str(tmp_path)
    download_main(csv=test_csv, outdir=test_outdir, download_fits=False)
    assert mock_dependencies["download"].call_count == DOWNLOAD_INGORE_FITS_DOWNLOAD_ASSERTION
    expected_calls = [call(f"{BASE_URL}mast:JWST/jpg/jw01.jpg", os.path.join(test_outdir, "jw01.jpg")), call(f"{BASE_URL}mast:JWST/jpg/jw02-filter.jpg", os.path.join(test_outdir, "jw02-filter.jpg"))]
    mock_dependencies["download"].assert_has_calls(expected_calls)


def test_download_must_contain(mock_dependencies, tmp_path):
    """
    Tests the --must_contain flag.
    """
    test_csv = "dummy.csv"
    test_outdir = str(tmp_path)

    download_main(csv=test_csv, outdir=test_outdir, must_contain="filter")
