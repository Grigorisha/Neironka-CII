#!/usr/bin/env python3
"""YOLO (бокс + класс дефекта) + SAM (точная маска внутри бокса).

Совмещает RoadDefectYoloPipeline (yolo_pipeline.py) с сегментацией SAM из
mask_pipeline.py: YOLO находит бокс и определяет класс дефекта, SAM по этому
боксу как prompt строит маску формы дефекта внутри него.

Дополнительно для каждой детекции строится "наивная" маска — бокс, закрашенный
целиком, — чтобы можно было сравнить, насколько SAM реально уточняет форму
относительно простой заливки прямоугольника.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from yolo_pipeline import Detection, RoadDefectYoloPipeline


@dataclass
class DefectMask:
    detection: Detection
    box_mask: np.ndarray  # bool HxW — бокс, закрашенный целиком (наивный вариант)
    sam_mask: np.ndarray | None  # bool HxW — точная маска SAM внутри бокса
    sam_score: float | None


class YoloSamMaskPipeline:
    def __init__(
        self,
        yolo_weights_dir: str | Path,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = RoadDefectYoloPipeline(
            weights_dir=yolo_weights_dir, threshold=threshold, device=self.device
        )
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(sam_model.to(self.device))

    @staticmethod
    def _box_fill_mask(box_xyxy: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        height, width = shape_hw
        x1, x2 = sorted(int(round(v)) for v in (box_xyxy[0], box_xyxy[2]))
        y1, y2 = sorted(int(round(v)) for v in (box_xyxy[1], box_xyxy[3]))
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask

    def detect_and_segment(self, frame_bgr: np.ndarray) -> List[DefectMask]:
        detections = self.yolo.detect(frame_bgr)
        if not detections:
            return []

        frame_rgb = frame_bgr[:, :, ::-1]
        self.sam_predictor.set_image(frame_rgb)
        shape_hw = frame_bgr.shape[:2]

        results: List[DefectMask] = []
        for det in detections:
            box_mask = self._box_fill_mask(det.box_xyxy, shape_hw)

            sam_mask = None
            sam_score = None
            try:
                raw_masks, sam_scores, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(det.box_xyxy, dtype=np.float32)[None, :],
                    multimask_output=True,
                )
                best_idx = int(np.argmax(sam_scores))
                sam_mask = raw_masks[best_idx].astype(bool)
                sam_score = float(sam_scores[best_idx])
            except Exception:
                pass

            results.append(
                DefectMask(detection=det, box_mask=box_mask, sam_mask=sam_mask, sam_score=sam_score)
            )
        return results

    @staticmethod
    def build_combined_mask(masks: List[np.ndarray | None], shape_hw: Tuple[int, int]) -> np.ndarray:
        height, width = shape_hw
        merged = np.zeros((height, width), dtype=np.uint8)
        for mask in masks:
            if mask is None or mask.shape != (height, width):
                continue
            merged[mask] = 255
        return merged
