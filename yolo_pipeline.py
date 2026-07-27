#!/usr/bin/env python3
"""Пайплайн детекции дефектов покрытия набором специализированных YOLO-моделей.

В отличие от mask_pipeline.py (OWL-ViT + SAM, open-vocabulary), здесь каждая
модель — узкоспециализированный YOLO-детектор одного типа дефекта. Ровно
такой же набор весов и та же логика выбора класса используются в боевом
Airflow DAG (см. "Новые файлы/vav_dag_1.py", функция get_predicts_from_models).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Имена файлов весов как в vav_dag_1.py (s3_model_0..3,5_key).
DEFAULT_MODEL_FILES: Dict[str, str] = {
    "longitude_crack": "best_111_longtitude.pt",
    "transverse_crack": "best_104_transversive.pt",
    "alligator_crack": "best_105_alligator_crack.pt",
    "pothole": "best_116_pothole.pt",
    "road_marks": "best_112_road_marks.pt",
}

# Русские названия — для консоли/чата.
MODEL_KEY_RU_NAMES: Dict[str, str] = {
    "longitude_crack": "Продольная трещина",
    "transverse_crack": "Поперечная трещина",
    "alligator_crack": "Крокодиловая трещина (сетка)",
    "pothole": "Яма",
    "road_marks": "Разметка",
}

# Фиксированный цвет (BGR) на каждый тип дефекта — один и тот же цвет должен
# всегда означать один и тот же класс во всех кадрах и во всех инструментах
# (run_yolo_sam_image.py, build_defect_quad_video.py), а не зависеть от
# порядка детекций внутри конкретного кадра.
MODEL_KEY_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "pothole": (0, 0, 255),            # красный
    "alligator_crack": (255, 0, 255),  # пурпурный
    "longitude_crack": (0, 255, 255),  # жёлтый
    "transverse_crack": (255, 255, 0), # голубой
    "road_marks": (0, 255, 0),         # зелёный
}


def color_for_model_key(model_key: str) -> Tuple[int, int, int]:
    if model_key in MODEL_KEY_COLORS_BGR:
        return MODEL_KEY_COLORS_BGR[model_key]
    # Фолбэк на случай новой модели без цвета в карте — стабильный по ключу,
    # но не гарантированно различимый со списком выше.
    digest = sum(ord(c) for c in model_key)
    hsv = np.uint8([[[(digest * 37) % 180, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


@dataclass
class Detection:
    label: str       # имя класса, зашитое в саму модель (model.names[...])
    model_key: str    # ключ модели из DEFAULT_MODEL_FILES, чей это детектор
    score: float
    box_xyxy: np.ndarray


# Подавление пересечений между детекциями разных моделей — портировано как есть
# из "Новые файлы/vav_dag_1.py" (iou/suppress_duplicates/is_inside/suppress_nested_boxes,
# вызывались там как suppress_duplicates(results, 0.70) -> suppress_nested_boxes(results)).
# Там функции работали со списком по всем кадрам сразу и сверяли ещё имя файла;
# здесь detect() уже вызывается на одном кадре, так что сверка по кадру не нужна.

def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    return inter_area / float(box1_area + box2_area - inter_area)


def suppress_duplicates(detections: List[Detection], iou_thresh: float = 0.70) -> List[Detection]:
    """Из пары пересекающихся боксов (IoU > iou_thresh) оставляет только с более высоким score."""
    if not detections:
        return []

    preds_sorted = sorted(detections, key=lambda d: -d.score)
    kept: List[Detection] = []
    used = [False] * len(preds_sorted)

    for i, p1 in enumerate(preds_sorted):
        if used[i]:
            continue
        for j in range(i + 1, len(preds_sorted)):
            if used[j]:
                continue
            if _iou(p1.box_xyxy, preds_sorted[j].box_xyxy) > iou_thresh:
                used[j] = True
        kept.append(p1)

    return kept


def _is_inside(box_inner: np.ndarray, box_outer: np.ndarray) -> bool:
    return (box_outer[0] <= box_inner[0] <= box_inner[2] <= box_outer[2] and
            box_outer[1] <= box_inner[1] <= box_inner[3] <= box_outer[3])


def suppress_nested_boxes(detections: List[Detection]) -> List[Detection]:
    """Убирает боксы, полностью лежащие внутри другого бокса из того же списка."""
    if not detections:
        return []

    kept: List[Detection] = []
    for i, p1 in enumerate(detections):
        is_nested = False
        for j, p2 in enumerate(detections):
            if i == j:
                continue
            if _is_inside(p1.box_xyxy, p2.box_xyxy):
                is_nested = True
                break
        if not is_nested:
            kept.append(p1)

    return kept


class RoadDefectYoloPipeline:
    """Прогоняет кадр через набор YOLO-моделей и объединяет их детекции."""

    def __init__(
        self,
        weights_dir: str | Path,
        model_files: Dict[str, str] | None = None,
        threshold: float = 0.25,
        device: str | None = None,
        dedup_iou_thresh: float = 0.70,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.model_files = dict(model_files) if model_files else dict(DEFAULT_MODEL_FILES)
        if not self.model_files:
            raise ValueError("Список моделей пуст.")
        self.threshold = float(threshold)
        self.device = device  # None -> ultralytics сам выберет cuda/cpu
        self.dedup_iou_thresh = float(dedup_iou_thresh)

        self.models: Dict[str, YOLO] = {}
        for key, filename in self.model_files.items():
            path = self.weights_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Веса модели '{key}' не найдены: {path}")
            model = YOLO(str(path))
            if self.device:
                model.to(self.device)
            self.models[key] = model

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Детекция на одном кадре (BGR, как из cv2.imread/cv2.VideoCapture)."""
        frame_rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(frame_rgb)

        detections: List[Detection] = []
        for key, model in self.models.items():
            results = model.predict(source=image, conf=self.threshold, verbose=False)
            result = results[0]
            boxes = result.boxes
            if boxes is None or boxes.cls.numel() == 0:
                continue
            for box in boxes:
                score = float(box.conf[0])
                if score < self.threshold:
                    continue
                cls_idx = int(box.cls[0])
                label = self._resolve_label(model, cls_idx, key)
                box_xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
                detections.append(
                    Detection(label=label, model_key=key, score=score, box_xyxy=box_xyxy)
                )

        detections = suppress_duplicates(detections, self.dedup_iou_thresh)
        detections = suppress_nested_boxes(detections)
        return detections

    @staticmethod
    def _resolve_label(model: YOLO, cls_idx: int, model_key: str) -> str:
        names = model.names
        if isinstance(names, dict):
            return names.get(cls_idx, f"{model_key}_{cls_idx}")
        try:
            return names[cls_idx]
        except (IndexError, TypeError):
            return f"{model_key}_{cls_idx}"
