import json
import os
import shutil

import cv2
import numpy as np
from src.combiner import main as combiner_main


def test_combiner_real_output(tmp_path):
    # Setup temporary directory structure
    extract_dir = tmp_path / "outputs/extracted_png/test_obj/test_filter"
    color_dir = tmp_path / "outputs/color_image"
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    try:
        # Create a "fake" numpy array (50% gray) and write as PNG
        test_image_path = extract_dir / "b5_w100_nan0_bb0_aw255_Asinh_ZScale.png"
        fake_data = np.ones((100, 100), dtype=np.uint8) * 127
        cv2.imwrite(str(test_image_path), fake_data)

        # Create a test JSON referencing this image
        config = {
            "colorspace": "bgr",
            "images": [
                {
                    "path": str(test_image_path),
                    "color": "$StellarCrimson",  # BGR: (0, 50, 255)
                    "factor": 1.0,
                }
            ],
        }
        config_path = tmp_path / "test_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        # Call the actual combiner script
        combiner_main(str(config_path), imagename="integration_result", outdir=str(color_dir))

        # Verification: Read the result and check average values
        result_path = color_dir / "integration_result.png"
        assert os.path.exists(result_path)

        result_img = cv2.imread(str(result_path))
        avg_bgr = np.mean(result_img, axis=(0, 1))

        # Expected: 127/255 * (0, 50, 255) ≈ (0, 25, 127)
        assert np.isclose(avg_bgr[0], 0, atol=1)
        assert np.isclose(avg_bgr[1], 25, atol=1)
        assert np.isclose(avg_bgr[2], 127, atol=1)

    finally:
        # Completeness: Delete created files and folders
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
