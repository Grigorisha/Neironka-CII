#!/usr/bin/env python3
"""
Конвертация camera_calib.yml (OpenCV) в YAML camera_info для ROS.
"""
import argparse
from pathlib import Path

import cv2
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True, help="OpenCV camera_calib.yml")
    parser.add_argument("--out", required=True, help="Output ROS camera_info YAML")
    parser.add_argument("--camera_name", default="realsense_rgb")
    args = parser.parse_args()

    fs = cv2.FileStorage(args.calib, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"Cannot open {args.calib}")
        return 1

    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeffs").mat().reshape(-1)
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    fs.release()

    data = {
        "image_width": w,
        "image_height": h,
        "camera_name": args.camera_name,
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": K.flatten().tolist(),
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(D.size),
            "data": D.tolist(),
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [
                K[0, 0],
                0.0,
                K[0, 2],
                0.0,
                0.0,
                K[1, 1],
                K[1, 2],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    print("Saved:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
