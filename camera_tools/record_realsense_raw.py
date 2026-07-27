#!/usr/bin/env python3
"""Превью и запись видео с RGB-камеры RealSense: старт/стоп записи — пробел.

Разрешение по умолчанию — максимум, который отдаёт RGB-сенсор D415
(см. `rs-enumerate-devices`): 1920x1080. FPS по умолчанию — 6: на 1920x1080
камера поддерживает только дискретные 30/15/6, и 6 — минимальный из них,
чтобы не гнаться за размером файла. Кадры пишутся без сжатия (ffmpeg, codec
rawvideo, pix_fmt bgr24) в контейнер .avi (Matroska сырой BGR нативно не
поддерживает).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Только однозначные форматы цветного сенсора. У D415 инфракрасный/depth-модуль
# тоже отдаёт 'UYVY' (но не на полном разрешении) и другие похожие с виду коды,
# поэтому широкий список меток по всему тексту (как в capture_realsense_rgb_photo.py)
# ложно принимает ИК-модуль за RGB-камеру. Здесь вместо этого проверяется конкретный
# формат-блок: код пикселей из белого списка ДОЛЖЕН поддерживать нужное разрешение.
COLOR_FOURCC_WHITELIST = ("YUYV", "RGB8", "RGB3", "BGR3", "MJPG")
FOURCC_BLOCK_RE = re.compile(r"\[\d+\]:\s*'(\w{2,4})'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Превью и запись видео с RGB-камеры RealSense (без сжатия). Пробел — старт/стоп, Q — выход."
    )
    parser.add_argument("output_dir", help="Папка для сохранения видео. Будет создана автоматически.")
    parser.add_argument("--device", help="Явно указать устройство, например /dev/video2.")
    parser.add_argument("--width", type=int, default=1920, help="Ширина кадра (макс. для D415 RGB).")
    parser.add_argument("--height", type=int, default=1080, help="Высота кадра (макс. для D415 RGB).")
    parser.add_argument(
        "--fps", type=float, default=6.0,
        help="Частота кадров (камера на 1920x1080 умеет только дискретные 30/15/6).",
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


def _format_blocks(formats_text: str) -> List[Tuple[str, str]]:
    """Разбивает вывод --list-formats-ext на блоки (код пикселей, текст блока)."""
    blocks: List[Tuple[str, str]] = []
    lines = formats_text.splitlines()
    current_code: Optional[str] = None
    current_lines: List[str] = []
    for line in lines:
        match = FOURCC_BLOCK_RE.search(line)
        if match:
            if current_code is not None:
                blocks.append((current_code, "\n".join(current_lines)))
            current_code = match.group(1).strip()
            current_lines = [line]
        elif current_code is not None:
            current_lines.append(line)
    if current_code is not None:
        blocks.append((current_code, "\n".join(current_lines)))
    return blocks


def supports_color_resolution(formats_text: str, width: int, height: int) -> bool:
    target = f"{width}x{height}"
    for code, block in _format_blocks(formats_text):
        if code.upper() in COLOR_FOURCC_WHITELIST and target in block:
            return True
    return False


def supports_any_color_format(formats_text: str) -> bool:
    return any(code.upper() in COLOR_FOURCC_WHITELIST for code, _ in _format_blocks(formats_text))


def pick_rgb_devices(explicit_device: Optional[str], width: int, height: int) -> List[Path]:
    if explicit_device:
        return [Path(explicit_device)]

    candidates = list_realsense_video_nodes()
    if not candidates:
        raise RuntimeError("Не найдены /dev/video* ноды RealSense.")

    formats_by_device = {d: get_v4l2_formats(d) for d in candidates}

    exact_match = [d for d in candidates if supports_color_resolution(formats_by_device[d], width, height)]
    if exact_match:
        return exact_match

    any_color = [d for d in candidates if supports_any_color_format(formats_by_device[d])]
    if any_color:
        return any_color

    raise RuntimeError(
        "Не удалось найти цветную (RGB) ноду RealSense среди "
        f"{[str(d) for d in candidates]}. Укажите её явно через --device "
        "(проверьте `v4l2-ctl -d /dev/videoN --list-formats-ext`, ищите 'YUYV'/'RGB8')."
    )


def open_rgb_capture(
    device_candidates: List[Path], width: int, height: int, fps: float
) -> Tuple[Optional[cv2.VideoCapture], Optional[Path]]:
    for device in device_candidates:
        cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, device
        cap.release()
    return None, None


class RawRecorder:
    """Пишет кадры без сжатия (ffmpeg, codec rawvideo) в .mkv."""

    def __init__(self, output_dir: Path, width: int, height: int, fps: float) -> None:
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.proc: Optional[subprocess.Popen] = None
        self.path: Optional[Path] = None

    @property
    def is_recording(self) -> bool:
        return self.proc is not None

    def start(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"realsense_raw_{ts}.avi"
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", "-",
            "-c:v", "rawvideo",
            "-pix_fmt", "bgr24",
            str(self.path),
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return self.path

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except BrokenPipeError:
            self.stop()

    def stop(self) -> Optional[Path]:
        if self.proc is None:
            return None
        finished_path = self.path
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        self.proc = None
        self.path = None
        return finished_path


def main() -> int:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        print("Ошибка: ffmpeg не найден в PATH (нужен для записи без сжатия).", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device_candidates = pick_rgb_devices(args.device, args.width, args.height)
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    cap, used_device = open_rgb_capture(device_candidates, args.width, args.height, args.fps)
    if cap is None:
        print("Ошибка: не удалось открыть RGB-камеру RealSense.", file=sys.stderr)
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps

    print(f"Устройство: {used_device}")
    print(f"Разрешение: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    print(f"Видео сохраняется без сжатия (ffmpeg rawvideo) в {output_dir}")
    print("Пробел — старт/стоп записи, Q — выход.")

    recorder = RawRecorder(output_dir, actual_w, actual_h, actual_fps)
    window_name = "RealSense D415 -- SPACE: rec, Q: quit"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Предупреждение: кадр не получен.", file=sys.stderr)
                break

            if recorder.is_recording:
                recorder.write(frame)

            display = frame.copy()
            if recorder.is_recording:
                cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(
                    display, "REC", (48, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
                )
            cv2.putText(
                display, "SPACE: start/stop rec | Q: quit",
                (10, display.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if recorder.is_recording:
                    saved = recorder.stop()
                    print(f"Запись остановлена: {saved}")
                else:
                    path = recorder.start()
                    print(f"Запись начата: {path}")
            elif key == ord("q"):
                break
    finally:
        if recorder.is_recording:
            saved = recorder.stop()
            print(f"Запись остановлена: {saved}")
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
