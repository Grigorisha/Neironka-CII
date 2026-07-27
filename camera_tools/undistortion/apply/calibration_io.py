"""Загрузка camera_calib.yml (OpenCV FileStorage) и построение карт undistort."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_width: int
    image_height: int
    map1: Optional[np.ndarray] = None
    map2: Optional[np.ndarray] = None

    def build_maps(self) -> None:
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            (self.image_width, self.image_height),
            cv2.CV_16SC2,
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.map1 is None or self.map2 is None:
            self.build_maps()
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)


def load_calibration(path: str | Path) -> CameraCalibration:
    path = Path(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Calibration file not found or invalid: {path}")

    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeffs").mat()
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    fs.release()

    if K is None or D is None or w <= 0 or h <= 0:
        raise ValueError(f"Incomplete calibration in {path}")

    calib = CameraCalibration(K, D, w, h)
    calib.build_maps()
    return calib
