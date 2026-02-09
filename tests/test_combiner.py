import os
import json
import cv2
import numpy as np
import shutil
from src.combiner import main as combiner_main


def test_combiner_integration(tmp_path):
    # 1. Setup temporary directory structure
    extract_dir = tmp_path / "extracted"
    color_dir = tmp_path / "final_image"
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    try:
        # 2. Create fake "extracted" PNGs (8-bit grayscale)
        path_a = str(extract_dir / "white_source.png")
        img_a = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.imwrite(path_a, img_a)

        path_b = str(extract_dir / "gray_source.png")
        img_b = np.ones((100, 100), dtype=np.uint8) * 127
        cv2.imwrite(path_b, img_b)

        # 3. Create the JSON configuration
        config = {
            "operand": "+",
            "colorspace": "bgr",
            "images": [
                {
                    "path": path_a,
                    "color": "$StellarCrimson",  # (0, 50, 255) in BGR
                    "factor": 1.0,
                },
                {
                    "path": path_b,
                    "color": "$OxygenTeal",  # (160, 180, 40) in BGR
                    "factor": 0.5,
                },
            ],
        }
        config_path = tmp_path / "test_combine.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        # 4. Run the combiner
        combiner_main(str(config_path), imagename="test_result", outdir=str(color_dir))

        # 5. Verify the output
        result_path = color_dir / "test_result.png"
        assert os.path.exists(result_path)

        result_img = cv2.imread(str(result_path))
        # Calculate expected values
        # B: 0 + (0.5 * (127/255) * 160) ≈ 40
        # G: 50 + (0.5 * (127/255) * 180) ≈ 95
        # R: 255 + (0.5 * (127/255) * 40) ≈ 255 (clipped)
        avg_bgr = np.mean(result_img, axis=(0, 1))
        assert np.isclose(avg_bgr[0], 40, atol=1)
        assert np.isclose(avg_bgr[1], 95, atol=1)
        assert np.isclose(avg_bgr[2], 255, atol=1)

    finally:
        # 6. Explicit Cleanup
        # Deletes the entire temporary tree created during this test
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
