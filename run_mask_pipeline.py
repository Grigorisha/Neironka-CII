#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Пайплайн масок OWL-ViT + SAM для видеофайла или камеры."
    )
    parser.add_argument("--mode", choices=["video", "camera"], required=True)
    parser.add_argument("--input", help="Путь к входному видео (для --mode video).")
    parser.add_argument("--camera", type=int, default=0, help="Индекс камеры (для --mode camera).")
    parser.add_argument("--fps", type=float, default=10.0, help="Целевой FPS выхода.")
    parser.add_argument("--width", type=int, default=1280, help="Ширина входа для камеры.")
    parser.add_argument("--height", type=int, default=720, help="Высота входа для камеры.")
    parser.add_argument("--threshold", type=float, default=0.15, help="Порог OWL-ViT.")
    parser.add_argument(
        "--texts",
        default="",
        help="Список классов через запятую. Пусто = DEFAULT_TEXTS из mask_pipeline.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/mask_output.mp4",
        help="Путь к выходному видео маски (для --mode video).",
    )
    parser.add_argument(
        "--stream-url",
        default="udp://127.0.0.1:5000?pkt_size=1316",
        help="URL для ffmpeg-стрима (например udp://... или rtsp://...).",
    )
    parser.add_argument(
        "--stream-format",
        choices=["udp", "rtsp"],
        default="udp",
        help="Формат контейнера для streaming-выхода.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Показывать окно: слева исходный кадр, справа маска.",
    )
    return parser.parse_args()


class FfmpegGrayWriter:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        destination: str,
        fmt: str,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.destination = destination
        self.proc: Optional[subprocess.Popen] = None

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg не найден в PATH.")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "gray",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
        ]
        if fmt == "rtsp":
            cmd.extend(["-f", "rtsp", "-rtsp_transport", "tcp"])
        elif fmt == "mp4":
            cmd.extend(["-f", "mp4", "-movflags", "+faststart"])
        else:
            cmd.extend(["-f", "mpegts"])
        cmd.append(destination)

        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print("ffmpeg writer:", shlex.join(cmd))

    def write(self, gray_mask: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("ffmpeg writer не инициализирован.")
        if gray_mask.dtype != np.uint8:
            raise ValueError("Ожидается mask dtype uint8.")
        if gray_mask.shape != (self.height, self.width):
            raise ValueError(
                f"Ожидается shape {(self.height, self.width)}, получено {gray_mask.shape}"
            )
        self.proc.stdin.write(gray_mask.tobytes())

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _parse_texts(arg_texts: str) -> list[str] | None:
    cleaned = [x.strip() for x in arg_texts.split(",") if x.strip()]
    return cleaned or None


def _open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.mode == "video":
        if not args.input:
            raise ValueError("--input обязателен для --mode video.")
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть источник видео.")
    return cap


def _build_writer(args: argparse.Namespace, width: int, height: int, fps: float) -> FfmpegGrayWriter:
    if args.mode == "video":
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        return FfmpegGrayWriter(width, height, fps, str(output), fmt="mp4")
    return FfmpegGrayWriter(width, height, fps, args.stream_url, fmt=args.stream_format)


def _build_preview_frame(frame_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    left = frame_bgr
    right = mask_bgr
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
    panel = np.hstack([left, right])
    cv2.putText(
        panel,
        "Camera",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Mask 0/255",
        (left.shape[1] + 16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def run() -> int:
    args = parse_args()
    texts = _parse_texts(args.texts)

    from local_config import MODEL_TYPE, SAM_CHECKPOINT
    from mask_pipeline import SamOwlVitMaskPipeline

    pipeline = SamOwlVitMaskPipeline(
        sam_checkpoint=SAM_CHECKPOINT,
        model_type=MODEL_TYPE,
        texts=texts,
        threshold=args.threshold,
    )
    print(f"Device: {pipeline.device}")

    cap = _open_capture(args)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out_fps = args.fps if args.fps > 0 else (src_fps if src_fps > 0 else 10.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _build_writer(args, frame_width, frame_height, out_fps)

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.mode == "video":
                    break
                print("Камера не вернула кадр, завершаю.")
                break

            mask, detections = pipeline.process_frame(frame)
            writer.write(mask)

            frame_idx += 1
            elapsed = max(time.time() - t0, 1e-6)
            if frame_idx % 10 == 0:
                proc_fps = frame_idx / elapsed
                print(
                    f"frame={frame_idx} detections={len(detections)} "
                    f"source_fps={src_fps:.2f} proc_fps={proc_fps:.2f}"
                )

            if args.preview:
                preview = _build_preview_frame(frame, mask)
                cv2.imshow("Camera + Segmentation Mask", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Остановлено пользователем (Ctrl+C).")
    finally:
        cap.release()
        writer.close()
        cv2.destroyAllWindows()

    if args.mode == "video":
        print(f"Готово: {Path(args.output).resolve()}")
    else:
        print(f"Стрим отправлен в: {args.stream_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
