#!/usr/bin/env python3
"""Обработка кадра детекторами дефектов и инфраструктуры — импортируемый API.

Модели загружаются один раз в конструкторе, дальше `process()` вызывается на
каждый кадр. Именно так это следует использовать из ROS-ноды или любого другого
долгоживущего процесса: запуск CLI `camera_tools/process_frame.py` на каждый кадр
означал бы перезагрузку моделей (~7 секунд) при каждом вызове.

    from frame_processor import FrameProcessor

    proc = FrameProcessor(kind="masks", damage="seg", infra="off")  # один раз
    result = proc.process(frame_bgr)                                 # на каждый кадр
    cv_image = result.image        # готовая картинка (BGR, как у OpenCV)
    for item in result.items:      # и сами детекции, если нужны отдельно
        print(item.source, item.label, item.score, item.box_xyxy)

Замеры скорости и почему обработка в реальном времени невозможна — см.
docs/INTEGRATION_ROS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Допустимые значения
KINDS = ("boxes-damage", "boxes-infra", "boxes-all", "masks")
SOURCE_MODES = ("seg", "box", "off")

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


@dataclass
class ProcessedItem:
    """Одна найденная сущность: дефект покрытия или объект инфраструктуры."""

    label: str                          # класс, как его назвала модель
    score: float                        # уверенность 0..1
    box_xyxy: np.ndarray                # [x1, y1, x2, y2] в пикселях кадра
    color_bgr: Tuple[int, int, int]     # цвет этого класса в палитре
    source: str                         # "damage" (5 YOLO) | "infra" (OWL-ViT)
    model_key: Optional[str] = None     # для damage — какая из 5 моделей
    mask: Optional[np.ndarray] = None   # bool-маска HxW, если режим её строит
    mask_score: Optional[float] = None  # уверенность SAM, если маска от SAM


@dataclass
class ProcessResult:
    image: np.ndarray                   # готовое изображение BGR
    items: List[ProcessedItem] = field(default_factory=list)

    @property
    def damage_items(self) -> List[ProcessedItem]:
        return [i for i in self.items if i.source == "damage"]

    @property
    def infra_items(self) -> List[ProcessedItem]:
        return [i for i in self.items if i.source == "infra"]


def box_fill_mask(box_xyxy: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Маска-прямоугольник: бокс, закрашенный целиком."""
    height, width = shape_hw
    x1, x2 = sorted(int(round(v)) for v in (box_xyxy[0], box_xyxy[2]))
    y1, y2 = sorted(int(round(v)) for v in (box_xyxy[1], box_xyxy[3]))
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def resolve_modes(kind: str, pipe: Optional[str] = None,
                  damage: Optional[str] = None, infra: Optional[str] = None) -> Tuple[str, str]:
    """Сводит вариант вызова к паре (damage_mode, infra_mode). Кидает ValueError."""
    if kind not in KINDS:
        raise ValueError(f"kind должен быть одним из {KINDS}, получено {kind!r}")

    if damage is not None or infra is not None:
        if pipe is not None:
            raise ValueError("Нельзя одновременно задавать pipe и damage/infra.")
        if kind != "masks":
            raise ValueError("damage/infra применимы только при kind='masks'.")
        damage_mode = damage or "off"
        infra_mode = infra or "off"
        for name, value in (("damage", damage_mode), ("infra", infra_mode)):
            if value not in SOURCE_MODES:
                raise ValueError(f"{name} должен быть одним из {SOURCE_MODES}, получено {value!r}")
        if damage_mode == "off" and infra_mode == "off":
            raise ValueError("damage='off' вместе с infra='off' не даст никакого результата.")
        return damage_mode, infra_mode

    if kind == "masks":
        if pipe is None:
            raise ValueError("Для kind='masks' нужно указать pipe (pipe1..pipe6) или damage/infra.")
        name = PIPE_NUMBERS.get(pipe, pipe)
        if name not in PIPE_PRESETS:
            raise ValueError(f"Неизвестный pipe {pipe!r}. Допустимо: "
                             f"{list(PIPE_NUMBERS)} или {list(PIPE_PRESETS)}.")
        return PIPE_PRESETS[name]

    if pipe is not None:
        raise ValueError("pipe действует только при kind='masks'.")
    if kind == "boxes-damage":
        return "box", "off"
    if kind == "boxes-infra":
        return "off", "box"
    return "box", "box"  # boxes-all


class FrameProcessor:
    """Держит загруженные модели и обрабатывает кадры по одному.

    Грузятся только те модели, которые нужны выбранному режиму: при
    damage='off' не поднимаются YOLO-модели, при infra='off' — OWL-ViT.
    """

    def __init__(
        self,
        kind: str = "masks",
        pipe: Optional[str] = None,
        damage: Optional[str] = None,
        infra: Optional[str] = None,
        weights_dir: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: Optional[str] = None,
        threshold: float = 0.25,
        infra_threshold: float = 0.15,
        infra_texts: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.kind = kind
        self.damage_mode, self.infra_mode = resolve_modes(kind, pipe, damage, infra)
        self.threshold = float(threshold)
        self.infra_threshold = float(infra_threshold)
        self.device = device

        # Пути по умолчанию берём из local_config.py — как и все скрипты проекта.
        import local_config
        weights_dir = weights_dir or local_config.YOLO_WEIGHTS_DIR
        sam_checkpoint = sam_checkpoint or local_config.SAM_CHECKPOINT
        sam_model_type = sam_model_type or local_config.MODEL_TYPE

        self._damage_pipeline = None
        self._infra_pipeline = None

        if self.damage_mode == "seg":
            from yolo_sam_pipeline import YoloSamMaskPipeline
            self._damage_pipeline = YoloSamMaskPipeline(
                yolo_weights_dir=weights_dir, sam_checkpoint=sam_checkpoint,
                sam_model_type=sam_model_type, threshold=self.threshold, device=device,
            )
        elif self.damage_mode == "box":
            from yolo_pipeline import RoadDefectYoloPipeline
            self._damage_pipeline = RoadDefectYoloPipeline(
                weights_dir=weights_dir, threshold=self.threshold, device=device,
            )

        if self.infra_mode != "off":
            from mask_pipeline import INFRA_TEXTS, SamOwlVitMaskPipeline
            self._infra_pipeline = SamOwlVitMaskPipeline(
                sam_checkpoint=sam_checkpoint, model_type=sam_model_type,
                texts=list(infra_texts) if infra_texts else INFRA_TEXTS,
                threshold=self.infra_threshold, device=device,
            )

    # --- основной метод ---

    def process(self, frame_bgr: np.ndarray) -> ProcessResult:
        """Обрабатывает один кадр BGR (как отдаёт cv2). Возвращает картинку и детекции."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Ожидается изображение BGR формы (H, W, 3).")

        shape_hw = frame_bgr.shape[:2]
        items: List[ProcessedItem] = []

        if self._damage_pipeline is not None:
            items.extend(self._collect_damage(frame_bgr, shape_hw))
        if self._infra_pipeline is not None:
            items.extend(self._collect_infra(frame_bgr, shape_hw))

        if self.kind == "masks":
            image = self.render_masks(shape_hw, items)
        else:
            image = self.render_boxes(frame_bgr, items)
        return ProcessResult(image=image, items=items)

    # --- сбор детекций ---

    def _collect_damage(self, frame_bgr: np.ndarray, shape_hw: Tuple[int, int]) -> List[ProcessedItem]:
        from yolo_pipeline import color_for_model_key

        out: List[ProcessedItem] = []
        if self.damage_mode == "seg":
            for r in self._damage_pipeline.detect_and_segment(frame_bgr):
                out.append(ProcessedItem(
                    label=r.detection.label, score=r.detection.score,
                    box_xyxy=r.detection.box_xyxy,
                    color_bgr=color_for_model_key(r.detection.model_key),
                    source="damage", model_key=r.detection.model_key,
                    mask=r.sam_mask, mask_score=r.sam_score,
                ))
        else:
            for d in self._damage_pipeline.detect(frame_bgr):
                out.append(ProcessedItem(
                    label=d.label, score=d.score, box_xyxy=d.box_xyxy,
                    color_bgr=color_for_model_key(d.model_key),
                    source="damage", model_key=d.model_key,
                    mask=box_fill_mask(d.box_xyxy, shape_hw),
                ))
        return out

    def _collect_infra(self, frame_bgr: np.ndarray, shape_hw: Tuple[int, int]) -> List[ProcessedItem]:
        from mask_pipeline import color_for_infra_label

        detections = self._infra_pipeline.detect_boxes(frame_bgr)
        out = [
            ProcessedItem(label=d.label, score=d.score, box_xyxy=d.box_xyxy,
                          color_bgr=color_for_infra_label(d.label), source="infra")
            for d in detections
        ]

        if self.infra_mode == "seg" and detections:
            masks = self._infra_pipeline.segment_boxes(frame_bgr, [d.box_xyxy for d in detections])
            # segment_boxes пропускает детекции, на которых SAM упал, поэтому длины
            # могут разойтись — сопоставляем по порядку, сколько получилось.
            for item, mask in zip(out, masks):
                item.mask = mask
        elif self.infra_mode == "box":
            for item in out:
                item.mask = box_fill_mask(item.box_xyxy, shape_hw)
        return out

    # --- отрисовка (статические: можно вызывать и на своих данных) ---

    @staticmethod
    def render_boxes(frame_bgr: np.ndarray, items: Sequence[ProcessedItem]) -> np.ndarray:
        annotated = frame_bgr.copy()
        for item in items:
            x1, y1, x2, y2 = (int(v) for v in item.box_xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), item.color_bgr, 2)
            cv2.putText(annotated, f"{item.label}: {item.score:.2f}", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, item.color_bgr, 2, cv2.LINE_AA)
        return annotated

    @staticmethod
    def render_masks(shape_hw: Tuple[int, int], items: Sequence[ProcessedItem]) -> np.ndarray:
        height, width = shape_hw
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        for item in items:
            if item.mask is None or item.mask.shape != (height, width):
                continue
            panel[item.mask] = item.color_bgr
        return panel
