#!/usr/bin/env python3
"""Исправление искажений одного изображения по camera_calib.yml (без ROS)."""
import argparse
import sys
from pathlib import Path

import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import load_calibration  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Undistort a single image using OpenCV calib YAML.")
    parser.add_argument("--calib", required=True, help="Path to camera_calib.yml")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    img = cv2.imread(args.input)
    if img is None:
        print(f"Cannot read image: {args.input}")
        return 1

    h, w = img.shape[:2]
    if (w, h) != (calib.image_width, calib.image_height):
        print(
            f"Warning: image size {w}x{h} differs from calib "
            f"{calib.image_width}x{calib.image_height}. Remap may be wrong."
        )

    out = calib.undistort(img)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, out)
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
