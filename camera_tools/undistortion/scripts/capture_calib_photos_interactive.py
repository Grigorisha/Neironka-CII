#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2

RGB_FOURCC_HINTS = ("MJPG", "YUYV", "UYVY", "RGB3", "BGR3")
DEPTH_FOURCC_HINTS = ("Z16", "Y16", "GREY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Интерактивная съемка калибровочных фото с live preview."
    )
    parser.add_argument(
        "output_dir",
        help="Папка для кадров калибровки (например calib_001.jpg, calib_002.jpg).",
    )
    parser.add_argument("--device", help="Явно указать устройство, например /dev/video2.")
    parser.add_argument("--width", type=int, default=1280, help="Ширина кадра.")
    parser.add_argument("--height", type=int, default=720, help="Высота кадра.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Число прогревочных кадров.")
    parser.add_argument("--prefix", default="calib", help="Префикс имени выходных файлов.")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="Расширение файлов.")
    parser.add_argument(
        "--max-shots",
        type=int,
        default=0,
        help="Максимум снимков (0 = без ограничения).",
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
    return candidates


def next_index(output_dir: Path, prefix: str, ext: str) -> int:
    pattern = f"{prefix}_*.{ext}"
    max_idx = 0
    for path in output_dir.glob(pattern):
        stem = path.stem
        if not stem.startswith(prefix + "_"):
            continue
        suffix = stem[len(prefix) + 1 :]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def open_camera(candidates: List[Path], width: int, height: int, warmup_frames: int) -> tuple[cv2.VideoCapture, Path]:
    open_errors: List[str] = []
    for device in candidates:
        cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        if not cap.isOpened():
            open_errors.append(f"{device}: open failed")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok_any = False
        for _ in range(max(1, warmup_frames)):
            ok, _ = cap.read()
            if ok:
                ok_any = True
                break
        if ok_any:
            return cap, device
        cap.release()
        open_errors.append(f"{device}: frame read failed")

    details = "\n".join(f"  - {x}" for x in open_errors)
    raise RuntimeError(f"Не удалось открыть RGB-камеру.\n{details}")


def _mouse_capture_callback(event: int, _x: int, _y: int, flags: int, state: dict) -> None:
    # В OpenCV боковые кнопки могут приходить как XBUTTON-события.
    xbutton_down = getattr(cv2, "EVENT_XBUTTONDOWN", -1)
    xbutton1_flag = getattr(cv2, "EVENT_FLAG_XBUTTON1", 0)
    xbutton2_flag = getattr(cv2, "EVENT_FLAG_XBUTTON2", 0)
    mbutton_down = getattr(cv2, "EVENT_MBUTTONDOWN", -1)

    if event == xbutton_down and flags & (xbutton1_flag | xbutton2_flag):
        state["capture_requested"] = True
        return

    # Фолбэк для систем, где XBUTTON-события не приходят.
    if event == mbutton_down:
        state["capture_requested"] = True


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device_candidates = pick_rgb_devices(args.device)
        cap, used_device = open_camera(device_candidates, args.width, args.height, args.warmup_frames)
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"RGB устройство: {used_device}")
    print("Управление: SPACE/s или боковая кнопка мыши = сохранить кадр, q/ESC = выход")

    idx = next_index(output_dir, args.prefix, args.ext)
    saved_count = 0
    window_name = "Calibration capture"
    mouse_state = {"capture_requested": False}
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_capture_callback, mouse_state)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Ошибка: кадр не получен.", file=sys.stderr)
                break

            preview = frame.copy()
            cv2.putText(
                preview,
                f"saved={saved_count} next={args.prefix}_{idx:03d}.{args.ext}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            capture_by_key = key in (ord("s"), 32)
            capture_by_mouse = bool(mouse_state["capture_requested"])
            if capture_by_key or capture_by_mouse:
                mouse_state["capture_requested"] = False
                output_path = output_dir / f"{args.prefix}_{idx:03d}.{args.ext}"
                if cv2.imwrite(str(output_path), frame):
                    saved_count += 1
                    idx += 1
                    print(f"Сохранено: {output_path}")
                    if args.max_shots > 0 and saved_count >= args.max_shots:
                        print(f"Достигнут лимит max-shots={args.max_shots}.")
                        break
                else:
                    print(f"Ошибка: не удалось сохранить {output_path}", file=sys.stderr)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Итого сохранено кадров: {saved_count}")
    print(f"Папка: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
