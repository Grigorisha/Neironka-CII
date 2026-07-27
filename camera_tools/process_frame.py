#!/usr/bin/env python3
"""Обработка одиночного кадра: на вход путь к изображению, на выход одна картинка.

Что окажется на выходе, определяют две группы флагов (см. docs/SPEC_process_frame.md):

  Группа 1 — что показывать (обязательна, ровно один флаг):
    --boxes-damage   рамки повреждений покрытия поверх оригинала (5 YOLO-моделей)
    --boxes-infra    рамки знаков, столбов, светофоров, люков поверх оригинала (OWL-ViT)
    --boxes-all      рамки обоих источников на одном кадре
    --masks          чёрный фон, цветные объекты; чем заполнять — группа 2

  Группа 2 — каким пайплайном (только вместе с --masks, ровно один флаг):
    --pipe1 / --pipe-seg-all       повреждения масками SAM  + инфраструктура масками SAM
    --pipe2 / --pipe-box-all       повреждения заливкой бокса + инфраструктура масками SAM
    --pipe3 / --pipe-seg-damage    только повреждения, масками SAM
    --pipe4 / --pipe-seg-infra     только инфраструктура, масками SAM
    --pipe5 / --pipe-box-damage    только повреждения, заливкой бокса
    --pipe6 / --pipe-box-infra     только инфраструктура, заливкой бокса

  Эквивалентная явная форма (выбор по сути двумерный):
    --damage {seg,box,off} --infra {seg,box,off}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CALIB_APPLY_DIR = Path(__file__).resolve().parent / "undistortion" / "apply"
if str(CALIB_APPLY_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_APPLY_DIR))


# Именованный режим -> (что делать с повреждениями, что делать с инфраструктурой)
PIPE_PRESETS = {
    "seg-all": ("seg", "seg"),
    "box-all": ("box", "seg"),
    "seg-damage": ("seg", "off"),
    "seg-infra": ("off", "seg"),
    "box-damage": ("box", "off"),
    "box-infra": ("off", "box"),
}
PIPE_NUMBERS = {
    "pipe1": "seg-all",
    "pipe2": "box-all",
    "pipe3": "seg-damage",
    "pipe4": "seg-infra",
    "pipe5": "box-damage",
    "pipe6": "box-infra",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обработка одного кадра детекторами дорожных дефектов и инфраструктуры.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Путь к входному изображению.")
    parser.add_argument("--output", required=True, help="Путь к выходному изображению (.png).")

    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--boxes-damage", dest="kind", action="store_const", const="boxes-damage",
                      help="Рамки повреждений покрытия поверх оригинала.")
    kind.add_argument("--boxes-infra", dest="kind", action="store_const", const="boxes-infra",
                      help="Рамки знаков/столбов/светофоров/люков поверх оригинала.")
    kind.add_argument("--boxes-all", dest="kind", action="store_const", const="boxes-all",
                      help="Рамки обоих источников поверх оригинала.")
    kind.add_argument("--masks", dest="kind", action="store_const", const="masks",
                      help="Маски: чёрный фон, цветные объекты.")

    pipe = parser.add_mutually_exclusive_group()
    for num, name in PIPE_NUMBERS.items():
        pipe.add_argument(f"--{num}", dest="pipe", action="store_const", const=name,
                          help=f"То же, что --pipe-{name}.")
    for name in PIPE_PRESETS:
        pipe.add_argument(f"--pipe-{name}", dest="pipe", action="store_const", const=name,
                          help=argparse.SUPPRESS)

    parser.add_argument("--damage", choices=["seg", "box", "off"],
                        help="Явная форма: как показывать повреждения покрытия.")
    parser.add_argument("--infra", choices=["seg", "box", "off"],
                        help="Явная форма: как показывать инфраструктуру.")

    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Порог уверенности YOLO (повреждения). По умолчанию 0.25.")
    parser.add_argument("--infra-threshold", type=float, default=0.15,
                        help="Порог уверенности OWL-ViT (инфраструктура). По умолчанию 0.15.")
    parser.add_argument("--weights-dir", default="",
                        help="Папка с .pt весами YOLO. Пусто = YOLO_WEIGHTS_DIR из local_config.py.")
    parser.add_argument("--device", default="", help="cuda | cpu. Пусто = автоопределение.")
    parser.add_argument("--calib", default="",
                        help="Файл калибровки OpenCV YAML — устранить дисторсию перед обработкой.")
    return parser.parse_args()


def resolve_modes(args: argparse.Namespace) -> Tuple[str, str]:
    """Сводит флаги к паре (damage_mode, infra_mode)."""
    if args.damage or args.infra:
        if args.pipe:
            raise ValueError("Нельзя одновременно использовать --pipeN и --damage/--infra.")
        if args.kind != "masks":
            raise ValueError("--damage/--infra применимы только вместе с --masks.")
        damage = args.damage or "off"
        infra = args.infra or "off"
        if damage == "off" and infra == "off":
            raise ValueError("--damage off вместе с --infra off не даст никакого результата.")
        return damage, infra

    if args.kind == "masks":
        if not args.pipe:
            raise ValueError(
                "Для --masks нужно указать пайплайн: --pipe1 … --pipe6 "
                "(или явно --damage/--infra). Список см. в --help."
            )
        return PIPE_PRESETS[args.pipe]

    # Режимы с рамками: пайплайн выбирать нечего, сегментация на рамки не влияет.
    if args.pipe:
        raise ValueError("Флаг пайплайна (--pipeN) действует только вместе с --masks.")
    if args.kind == "boxes-damage":
        return "box", "off"
    if args.kind == "boxes-infra":
        return "off", "box"
    return "box", "box"  # boxes-all


def load_calibration_or_die(path_str: str):
    from calibration_io import load_calibration

    calib_path = Path(path_str).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Файл калибровки не найден: {calib_path}")
    return load_calibration(calib_path)


def draw_boxes(frame_bgr: np.ndarray, items: List[Tuple[str, float, np.ndarray, Tuple[int, int, int]]]) -> np.ndarray:
    annotated = frame_bgr.copy()
    for label, score, box, color in items:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label}: {score:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return annotated


def build_mask_panel(shape_hw: Tuple[int, int],
                     items: List[Tuple[Optional[np.ndarray], Tuple[int, int, int]]]) -> np.ndarray:
    height, width = shape_hw
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    for mask, color in items:
        if mask is None or mask.shape != (height, width):
            continue
        panel[mask] = color
    return panel


def box_fill_mask(box_xyxy: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    x1, x2 = sorted(int(round(v)) for v in (box_xyxy[0], box_xyxy[2]))
    y1, y2 = sorted(int(round(v)) for v in (box_xyxy[1], box_xyxy[3]))
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def run() -> int:
    args = parse_args()

    try:
        damage_mode, infra_mode = resolve_modes(args)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Ошибка: входной файл не найден: {input_path}", file=sys.stderr)
        return 1
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Ошибка: не удалось прочитать изображение: {input_path}", file=sys.stderr)
        return 1

    if args.calib:
        try:
            calib = load_calibration_or_die(args.calib)
        except Exception as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            return 1
        h, w = frame.shape[:2]
        if (w, h) != (calib.image_width, calib.image_height):
            print(f"Ошибка: размер кадра {w}x{h} не совпадает с калибровкой "
                  f"{calib.image_width}x{calib.image_height}", file=sys.stderr)
            return 1
        frame = calib.undistort(frame)

    shape_hw = frame.shape[:2]
    device = args.device or None

    from local_config import MODEL_TYPE, SAM_CHECKPOINT, YOLO_WEIGHTS_DIR

    box_items: List[Tuple[str, float, np.ndarray, Tuple[int, int, int]]] = []
    mask_items: List[Tuple[Optional[np.ndarray], Tuple[int, int, int]]] = []
    n_damage = n_infra = 0

    # --- Повреждения покрытия: 5 YOLO-моделей (+ SAM в режиме seg) ---
    if damage_mode != "off":
        from yolo_pipeline import color_for_model_key
        weights_dir = Path(args.weights_dir).expanduser().resolve() if args.weights_dir else Path(YOLO_WEIGHTS_DIR)

        if damage_mode == "seg":
            from yolo_sam_pipeline import YoloSamMaskPipeline
            pipeline = YoloSamMaskPipeline(
                yolo_weights_dir=weights_dir, sam_checkpoint=SAM_CHECKPOINT,
                sam_model_type=MODEL_TYPE, threshold=args.threshold, device=device,
            )
            results = pipeline.detect_and_segment(frame)
            n_damage = len(results)
            for r in results:
                color = color_for_model_key(r.detection.model_key)
                box_items.append((r.detection.label, r.detection.score, r.detection.box_xyxy, color))
                mask_items.append((r.sam_mask, color))
        else:  # box — сегментацию не запускаем, SAM не нужен
            from yolo_pipeline import RoadDefectYoloPipeline
            pipeline = RoadDefectYoloPipeline(
                weights_dir=weights_dir, threshold=args.threshold, device=device,
            )
            detections = pipeline.detect(frame)
            n_damage = len(detections)
            for d in detections:
                color = color_for_model_key(d.model_key)
                box_items.append((d.label, d.score, d.box_xyxy, color))
                mask_items.append((box_fill_mask(d.box_xyxy, shape_hw), color))

    # --- Инфраструктура: OWL-ViT (+ SAM в режиме seg) ---
    if infra_mode != "off":
        from mask_pipeline import INFRA_TEXTS, SamOwlVitMaskPipeline, color_for_infra_label

        infra_pipeline = SamOwlVitMaskPipeline(
            sam_checkpoint=SAM_CHECKPOINT, model_type=MODEL_TYPE, texts=INFRA_TEXTS,
            threshold=args.infra_threshold, device=device,
        )
        detections = infra_pipeline.detect_boxes(frame)
        n_infra = len(detections)
        for d in detections:
            color = color_for_infra_label(d.label)
            box_items.append((d.label, d.score, d.box_xyxy, color))

        if infra_mode == "seg":
            masks = infra_pipeline.segment_boxes(frame, [d.box_xyxy for d in detections])
            # segment_boxes пропускает детекции, на которых SAM упал, поэтому
            # длины могут разойтись — сопоставляем по порядку, сколько есть.
            for mask, d in zip(masks, detections):
                mask_items.append((mask, color_for_infra_label(d.label)))
        else:
            for d in detections:
                mask_items.append((box_fill_mask(d.box_xyxy, shape_hw), color_for_infra_label(d.label)))

    # --- Отрисовка ---
    if args.kind == "masks":
        result = build_mask_panel(shape_hw, mask_items)
    else:
        result = draw_boxes(frame, box_items)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        print(f"Ошибка: не удалось сохранить {output_path}", file=sys.stderr)
        return 1

    print(f"Режим: {args.kind} (повреждения={damage_mode}, инфраструктура={infra_mode})")
    print(f"Найдено: повреждений {n_damage}, инфраструктуры {n_infra}")
    for label, score, _, _ in box_items:
        print(f"  {label}: {score:.3f}")
    print(f"Сохранено: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
