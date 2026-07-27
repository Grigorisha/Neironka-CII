#!/usr/bin/env python3
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2

CALIB_APPLY_DIR = Path(__file__).resolve().parent / "undistortion" / "apply"
if str(CALIB_APPLY_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_APPLY_DIR))

from calibration_io import CameraCalibration, load_calibration


RGB_FOURCC_HINTS = ("MJPG", "YUYV", "UYVY", "RGB3", "BGR3")
DEPTH_FOURCC_HINTS = ("Z16", "Y16", "GREY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Снять один кадр с RGB-камеры Intel RealSense и сохранить как фото."
    )
    parser.add_argument(
        "output_dir",
        help="Папка для сохранения фото. Будет создана автоматически.",
    )
    parser.add_argument(
        "--device",
        help="Явно указать устройство, например /dev/video2 (если автоопределение не подходит).",
    )
    parser.add_argument("--width", type=int, default=1280, help="Желаемая ширина кадра.")
    parser.add_argument("--height", type=int, default=720, help="Желаемая высота кадра.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Сколько кадров пропустить для стабилизации.")
    parser.add_argument(
        "--calib",
        required=True,
        help="Путь к файлу калибровки OpenCV YAML (camera_calib.yml).",
    )
    return parser.parse_args()


def list_realsense_video_nodes() -> List[Path]:
    nodes: List[Path] = []
    for video in sorted(Path("/sys/class/video4linux").glob("video*")):
        name_file = video / "name"
        if not name_file.exists():
            continue
        name = name_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
        if "realsense" in name:
            nodes.append(Path("/dev") / video.name)
    return nodes


def get_v4l2_formats(device: Path) -> str:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", str(device), "--list-formats-ext"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return result.stdout or ""


def looks_like_rgb_stream(formats_text: str) -> bool:
    upper = formats_text.upper()
    has_rgb = any(code in upper for code in RGB_FOURCC_HINTS)
    has_depth_only = any(code in upper for code in DEPTH_FOURCC_HINTS) and not has_rgb
    return has_rgb and not has_depth_only


def pick_rgb_devices(explicit_device: Optional[str]) -> List[Path]:
    if explicit_device:
        return [Path(explicit_device)]

    candidates = list_realsense_video_nodes()
    if not candidates:
        raise RuntimeError("Не найдены /dev/video* ноды RealSense.")

    rgb_candidates: List[Path] = []
    for dev in candidates:
        formats = get_v4l2_formats(dev)
        if looks_like_rgb_stream(formats):
            rgb_candidates.append(dev)

    if rgb_candidates:
        return rgb_candidates

    # Фолбэк: все ноды RealSense, если форматы не удалось распарсить.
    return candidates


def _load_required_calibration(path_str: str) -> CameraCalibration:
    calib_path = Path(path_str).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Файл калибровки не найден: {calib_path}")
    try:
        return load_calibration(calib_path)
    except Exception as exc:
        raise RuntimeError(f"Не удалось загрузить калибровку из {calib_path}: {exc}") from exc


def _ensure_resolution_matches(frame, calib: CameraCalibration) -> None:
    frame_h, frame_w = frame.shape[:2]
    if (frame_w, frame_h) != (calib.image_width, calib.image_height):
        raise RuntimeError(
            "Размер кадра не совпадает с калибровкой: "
            f"frame={frame_w}x{frame_h}, calib={calib.image_width}x{calib.image_height}"
        )


def main() -> int:
    args = parse_args()
    try:
        calib = _load_required_calibration(args.calib)
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device_candidates = pick_rgb_devices(args.device)
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    frame = None
    used_device: Optional[Path] = None
    open_errors: List[str] = []
    for device in device_candidates:
        cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        if not cap.isOpened():
            open_errors.append(f"{device}: open failed")
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        for _ in range(max(1, args.warmup_frames)):
            ok, frame = cap.read()
            if ok:
                used_device = device
                break
        cap.release()

        if used_device is not None and frame is not None:
            break
        open_errors.append(f"{device}: frame read failed")

    if frame is None or used_device is None:
        print("Ошибка: не удалось получить кадр с RGB камеры RealSense.", file=sys.stderr)
        if open_errors:
            print("Детали:", file=sys.stderr)
            for err in open_errors:
                print(f"  - {err}", file=sys.stderr)
        return 1
    try:
        _ensure_resolution_matches(frame, calib)
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    frame = calib.undistort(frame)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"realsense_rgb_{ts}.jpg"
    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        print(f"Ошибка: не удалось сохранить файл {output_path}", file=sys.stderr)
        return 1

    print(f"RGB устройство: {used_device}")
    print(f"Фото сохранено: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
