#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обработка одного PNG: OWL-ViT + SAM -> цветная instance-маска PNG."
    )
    parser.add_argument(
        "--input-image",
        required=True,
        help="Путь к входному PNG (или другому формату, читаемому OpenCV).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Директория для сохранения результата (будет создана при необходимости).",
    )
    parser.add_argument(
        "--output-name",
        default="instance_mask.png",
        help="Имя выходного PNG файла в output-dir.",
    )
    parser.add_argument("--threshold", type=float, default=0.15, help="Порог OWL-ViT.")
    parser.add_argument(
        "--texts",
        default="",
        help="Список классов через запятую. Пусто = DEFAULT_TEXTS из mask_pipeline.py",
    )
    return parser.parse_args()


def _parse_texts(arg_texts: str) -> list[str] | None:
    cleaned = [x.strip() for x in arg_texts.split(",") if x.strip()]
    return cleaned or None


def _make_color_bgr(idx: int) -> tuple[int, int, int]:
    # Детеминированная палитра по индексу объекта.
    hue = (idx * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _build_instance_color_mask(masks: list[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    colored = np.zeros((height, width, 3), dtype=np.uint8)

    for idx, mask in enumerate(masks, start=1):
        if mask.shape != (height, width):
            continue
        color = _make_color_bgr(idx)
        colored[mask.astype(bool)] = color
    return colored


def run() -> int:
    args = parse_args()

    from local_config import MODEL_TYPE, SAM_CHECKPOINT
    from mask_pipeline import SamOwlVitMaskPipeline

    input_path = Path(args.input_image).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    frame_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {input_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")

    pipeline = SamOwlVitMaskPipeline(
        sam_checkpoint=SAM_CHECKPOINT,
        model_type=MODEL_TYPE,
        texts=_parse_texts(args.texts),
        threshold=args.threshold,
    )
    print(f"Device: {pipeline.device}")

    detections = pipeline.detect_boxes(frame_bgr)
    masks = pipeline.segment_boxes(frame_bgr, (d.box_xyxy for d in detections))
    colored_mask = _build_instance_color_mask(masks, frame_bgr.shape[:2])

    ok = cv2.imwrite(str(output_path), colored_mask)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить PNG: {output_path}")

    print(f"detections={len(detections)} instances={len(masks)}")
    print(f"Готово: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
