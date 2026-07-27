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
CALIB_APPLY_DIR = Path(__file__).resolve().parent / "undistortion" / "apply"
if str(CALIB_APPLY_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_APPLY_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Обработка одного изображения набором YOLO-моделей дефектов "
            "покрытия (трещины/яма/разметка) -> PNG с боксами."
        )
    )
    parser.add_argument(
        "--input-image",
        required=True,
        help="Путь к входному изображению (PNG/JPG, всё что читает OpenCV).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Директория для сохранения результата (будет создана при необходимости).",
    )
    parser.add_argument(
        "--output-name",
        default="yolo_detections.png",
        help="Имя выходного PNG файла в output-dir.",
    )
    parser.add_argument(
        "--weights-dir",
        default="",
        help="Папка с .pt весами. Пусто = YOLO_WEIGHTS_DIR из local_config.py.",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Порог уверенности YOLO.")
    parser.add_argument(
        "--calib",
        default="",
        help=(
            "Опционально: файл калибровки OpenCV YAML (camera_calib.yml) для undistort "
            "перед детекцией. По умолчанию не применяется — YOLO-модели обучены на "
            "необработанных кадрах видеорегистратора, а не на кадрах с камеры этого стенда."
        ),
    )
    return parser.parse_args()


def _make_color_bgr(idx: int) -> tuple[int, int, int]:
    # Детерминированная палитра по индексу модели (не объекта — так у каждого
    # типа дефекта свой цвет на всех кадрах).
    hue = (idx * 61) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _load_required_calibration(path_str: str):
    from calibration_io import load_calibration

    calib_path = Path(path_str).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Файл калибровки не найден: {calib_path}")
    try:
        return load_calibration(calib_path)
    except Exception as exc:
        raise RuntimeError(f"Не удалось загрузить калибровку из {calib_path}: {exc}") from exc


def _ensure_resolution_matches(frame: np.ndarray, calib) -> None:
    frame_h, frame_w = frame.shape[:2]
    if (frame_w, frame_h) != (calib.image_width, calib.image_height):
        raise RuntimeError(
            "Размер изображения не совпадает с калибровкой: "
            f"frame={frame_w}x{frame_h}, calib={calib.image_width}x{calib.image_height}"
        )


def run() -> int:
    args = parse_args()

    from local_config import YOLO_WEIGHTS_DIR
    from yolo_pipeline import RoadDefectYoloPipeline

    input_path = Path(args.input_image).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    frame_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {input_path}")

    if args.calib:
        calib = _load_required_calibration(args.calib)
        _ensure_resolution_matches(frame_bgr, calib)
        frame_bgr = calib.undistort(frame_bgr)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")

    weights_dir = Path(args.weights_dir).expanduser().resolve() if args.weights_dir else Path(YOLO_WEIGHTS_DIR)
    pipeline = RoadDefectYoloPipeline(weights_dir=weights_dir, threshold=args.threshold)
    print(f"Моделей загружено: {len(pipeline.models)} из {weights_dir}")

    detections = pipeline.detect(frame_bgr)

    annotated = frame_bgr.copy()
    model_keys = list(pipeline.models.keys())
    for det in detections:
        color = _make_color_bgr(model_keys.index(det.model_key))
        x1, y1, x2, y2 = (int(v) for v in det.box_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        caption = f"{det.label}: {det.score:.2f}"
        cv2.putText(
            annotated,
            caption,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    ok = cv2.imwrite(str(output_path), annotated)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить PNG: {output_path}")

    print(f"detections={len(detections)}")
    for det in detections:
        print(f"  [{det.model_key}] {det.label}: {det.score:.3f} box={det.box_xyxy.tolist()}")
    print(f"Готово: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
