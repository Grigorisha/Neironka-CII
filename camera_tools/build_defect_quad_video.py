#!/usr/bin/env python3
"""Склеивает серию отснятых видео и строит сравнительный ролик 2x2:

    [ оригинал           ] [ маска: закрашенный бокс   ]
    [ оригинал + боксы    ] [ маска: точная (YOLO+SAM)  ]

На каждый кадр запускается YoloSamMaskPipeline (YOLO находит бокс+класс
дефекта, SAM строит по этому боксу точную маску формы). Выходное видео
сжимается (H.264), в отличие от исходных несжатых записей с камеры.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Склейка + YOLO/SAM разметка серии видео в сравнительный ролик 2x2."
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="Видеофайлы для склейки, в нужном порядке. Если не заданы — берутся все *.avi из --input-dir по имени (хронологически).",
    )
    parser.add_argument(
        "--input-dir", default="",
        help="Папка с видео для автосклейки (используется, если inputs не заданы).",
    )
    parser.add_argument("--output", required=True, help="Путь к выходному .mp4.")
    parser.add_argument(
        "--weights-dir", default="",
        help="Папка с .pt весами YOLO. Пусто = YOLO_WEIGHTS_DIR из local_config.py.",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Порог уверенности YOLO.")
    parser.add_argument("--fps", type=float, default=0.0, help="FPS выходного видео. 0 = взять из первого входного файла.")
    parser.add_argument("--limit-frames", type=int, default=0, help="Обработать не больше N кадров (0 = все). Полезно для быстрой проверки.")
    return parser.parse_args()


def _draw_boxes(frame_bgr: np.ndarray, detections) -> np.ndarray:
    from yolo_pipeline import color_for_model_key

    annotated = frame_bgr.copy()
    for det in detections:
        color = color_for_model_key(det.model_key)
        x1, y1, x2, y2 = (int(v) for v in det.box_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        caption = f"{det.label}: {det.score:.2f}"
        cv2.putText(
            annotated, caption, (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return annotated


def _build_mask_panel(detections, masks: List[Optional[np.ndarray]], shape_hw: Tuple[int, int]) -> np.ndarray:
    from yolo_pipeline import color_for_model_key

    height, width = shape_hw
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    for det, mask in zip(detections, masks):
        if mask is None:
            continue
        panel[mask] = color_for_model_key(det.model_key)
    return panel


def _label(text: str, panel: np.ndarray) -> np.ndarray:
    labeled = panel.copy()
    cv2.putText(labeled, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return labeled



def _resolve_inputs(args: argparse.Namespace) -> List[Path]:
    if args.inputs:
        paths = [Path(p).expanduser().resolve() for p in args.inputs]
    else:
        if not args.input_dir:
            raise ValueError("Нужно указать либо список видео, либо --input-dir.")
        input_dir = Path(args.input_dir).expanduser().resolve()
        paths = sorted(input_dir.glob("*.avi"))
        if not paths:
            paths = sorted(input_dir.glob("*.mp4"))
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Входной файл не найден: {p}")
    if not paths:
        raise FileNotFoundError("Не найдено ни одного входного видео.")
    return paths


def iter_concatenated_frames(paths: List[Path]) -> Iterator[np.ndarray]:
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Предупреждение: не удалось открыть {path}, пропуск.", file=sys.stderr)
            continue
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


def probe_first_frame(paths: List[Path]) -> Tuple[int, int]:
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            return w, h
    raise RuntimeError("Не удалось прочитать ни одного кадра из входных видео.")


def probe_fps(path: Path, default_fps: float = 6.0) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default_fps


class QuadVideoWriter:
    """Пишет сжатое (H.264) видео из готовых кадров через ffmpeg-пайп."""

    def __init__(self, output_path: Path, width: int, height: int, fps: float) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            str(output_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, frame_bgr: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=120)


def run() -> int:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        print("Ошибка: ffmpeg не найден в PATH.", file=sys.stderr)
        return 1

    try:
        input_paths = _resolve_inputs(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"Входных видео: {len(input_paths)}")
    for p in input_paths:
        print(f"  - {p.name}")

    frame_w, frame_h = probe_first_frame(input_paths)
    fps = args.fps if args.fps > 0 else probe_fps(input_paths[0])
    print(f"Разрешение кадра: {frame_w}x{frame_h}, выходной FPS: {fps}")

    from local_config import MODEL_TYPE, SAM_CHECKPOINT, YOLO_WEIGHTS_DIR
    from yolo_sam_pipeline import YoloSamMaskPipeline

    weights_dir = Path(args.weights_dir).expanduser().resolve() if args.weights_dir else Path(YOLO_WEIGHTS_DIR)
    pipeline = YoloSamMaskPipeline(
        yolo_weights_dir=weights_dir,
        sam_checkpoint=SAM_CHECKPOINT,
        sam_model_type=MODEL_TYPE,
        threshold=args.threshold,
    )
    print(f"Device: {pipeline.device}")

    output_path = Path(args.output).expanduser().resolve()
    writer = QuadVideoWriter(output_path, frame_w * 2, frame_h * 2, fps)

    shape_hw = (frame_h, frame_w)
    frame_idx = 0
    total_detections = 0
    start = time.time()
    try:
        for frame in iter_concatenated_frames(input_paths):
            if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
                frame = cv2.resize(frame, (frame_w, frame_h))

            results = pipeline.detect_and_segment(frame)
            total_detections += len(results)
            detections = [r.detection for r in results]
            box_masks = [r.box_mask for r in results]
            sam_masks = [r.sam_mask for r in results]

            top_left = _label("Original", frame)
            bottom_left = _label("Detections", _draw_boxes(frame, detections))
            top_right = _label("Mask: box-fill", _build_mask_panel(detections, box_masks, shape_hw))
            bottom_right = _label("Mask: SAM (precise)", _build_mask_panel(detections, sam_masks, shape_hw))

            grid = np.vstack([
                np.hstack([top_left, top_right]),
                np.hstack([bottom_left, bottom_right]),
            ])
            writer.write(grid)

            frame_idx += 1
            if frame_idx % 20 == 0:
                elapsed = time.time() - start
                print(f"Кадров обработано: {frame_idx} ({elapsed / frame_idx:.2f} с/кадр), детекций всего: {total_detections}")

            if args.limit_frames and frame_idx >= args.limit_frames:
                print(f"Достигнут --limit-frames={args.limit_frames}, остановка.")
                break
    finally:
        writer.close()

    elapsed = time.time() - start
    print(f"Готово: {output_path}")
    print(f"Кадров: {frame_idx}, время: {elapsed:.1f} с ({elapsed / max(1, frame_idx):.2f} с/кадр), детекций всего: {total_detections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
