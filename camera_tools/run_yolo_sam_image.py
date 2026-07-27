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
            "YOLO находит бокс+класс дефекта; сравнение двух вариантов маски внутри "
            "бокса — наивная заливка всего прямоугольника vs точная маска SAM."
        )
    )
    parser.add_argument("--input-image", required=True, help="Путь к входному изображению.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Директория для сохранения результатов (будет создана при необходимости).",
    )
    parser.add_argument(
        "--weights-dir",
        default="",
        help="Папка с .pt весами YOLO. Пусто = YOLO_WEIGHTS_DIR из local_config.py.",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Порог уверенности YOLO.")
    parser.add_argument(
        "--calib",
        default="",
        help="Опционально: файл калибровки OpenCV YAML для undistort перед детекцией.",
    )
    return parser.parse_args()


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


def _build_mask_panel(detections, masks: list[np.ndarray | None], shape_hw: tuple[int, int]) -> np.ndarray:
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


def run() -> int:
    args = parse_args()

    from local_config import MODEL_TYPE, SAM_CHECKPOINT, YOLO_WEIGHTS_DIR
    from yolo_sam_pipeline import YoloSamMaskPipeline

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

    weights_dir = Path(args.weights_dir).expanduser().resolve() if args.weights_dir else Path(YOLO_WEIGHTS_DIR)
    pipeline = YoloSamMaskPipeline(
        yolo_weights_dir=weights_dir,
        sam_checkpoint=SAM_CHECKPOINT,
        sam_model_type=MODEL_TYPE,
        threshold=args.threshold,
    )
    print(f"Device: {pipeline.device}")

    results = pipeline.detect_and_segment(frame_bgr)
    print(f"detections={len(results)}")
    for r in results:
        sam_info = f"sam_score={r.sam_score:.3f}" if r.sam_score is not None else "SAM: маска не построена"
        print(f"  [{r.detection.model_key}] {r.detection.label}: yolo_score={r.detection.score:.3f} {sam_info}")

    detections = [r.detection for r in results]
    box_masks: list[np.ndarray | None] = [r.box_mask for r in results]
    sam_masks: list[np.ndarray | None] = [r.sam_mask for r in results]

    shape_hw = frame_bgr.shape[:2]
    boxes_panel = _draw_boxes(frame_bgr, detections)
    box_mask_panel = _build_mask_panel(detections, box_masks, shape_hw)
    sam_mask_panel = _build_mask_panel(detections, sam_masks, shape_hw)

    cv2.imwrite(str(output_dir / "boxes.png"), boxes_panel)
    cv2.imwrite(str(output_dir / "mask_box_fill.png"), box_mask_panel)
    cv2.imwrite(str(output_dir / "mask_sam.png"), sam_mask_panel)

    comparison = np.hstack([
        _label("YOLO boxes", boxes_panel),
        _label("Box-fill mask", box_mask_panel),
        _label("SAM mask", sam_mask_panel),
    ])
    comparison_path = output_dir / "comparison.png"
    cv2.imwrite(str(comparison_path), comparison)

    print(f"Готово: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
