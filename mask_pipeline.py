#!/usr/bin/env python3
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import OwlViTForObjectDetection, OwlViTProcessor


DEFAULT_TEXTS: List[str] = [
    "car",
    "truck",
    "road sign",
    "tree",
    "person",
    "lamp post",
    "street light",
    "light pole",
    "pole",
    "utility pole",
    "electric pole",
    "telephone pole",
    "road markings",
    "lane markings",
    "street markings",
    "road lines",
    "traffic lines",
    "pedestrian crossing",
    "road stripe",
    "traffic lane",
    "lane marking",
    "road line",
    "solid white line",
    "dashed white line",
    "yellow road line",
    "double yellow line",
    "lane divider",
    "center line",
    "crosswalk",
    "zebra crossing",
    "stop line",
    "turn arrow on road",
    "bicycle lane marking",
    "bus lane marking",
    "road sign painted on asphalt",
    "traffic symbols on road",
    "no parking marking",
    "speed limit painted on road",
    "chevron road marking",
    "merge arrows on road",
    "intersection marking",
    "pedestrian marking",
    "crack in the road",
    "asphalt crack",
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "block crack",
    "edge crack",
    "pothole",
    "shallow pothole",
    "deep pothole",
    "sinkhole",
    "rutting",
    "road rut",
    "uneven road surface",
    "bumpy road",
    "road depression",
    "surface wear",
    "road erosion",
    "faded road surface",
    "delamination of asphalt",
    "loose gravel on road",
    "missing asphalt patch",
    "repaired patch",
    "rough surface",
    "collapsed manhole",
    "open manhole",
    "broken drain grate",
    "sunken manhole",
    "raised utility cover",
    "damaged curb",
    "eroded road shoulder",
    "damaged speed bump",
    "damaged pedestrian crossing",
    "faded road marking",
    "damaged crosswalk marking",
    "erased lane line",
    "temporary patch on road",
    "temporary repair",
    "construction damage",
    "roadwork trace",
]


# Придорожная инфраструктура: знаки, столбы, светофоры, люки и решётки.
# Отдельный от DEFAULT_TEXTS список — он короче (быстрее прогон OWL-ViT) и не
# содержит повреждений покрытия и разметки, которые ищут специализированные
# YOLO-модели. При совместном режиме дубли между пайплайнами сейчас намеренно
# не подавляются — сравниваем качество, чистку отложили.
INFRA_TEXTS: List[str] = [
    # Знаки
    "road sign",
    "traffic sign",
    "stop sign",
    "speed limit sign",
    "warning sign",
    # Столбы и опоры
    "lamp post",
    "street light",
    "light pole",
    "pole",
    "utility pole",
    "electric pole",
    "telephone pole",
    # Светофоры
    "traffic light",
    # Люки и водоотвод
    "manhole",
    "manhole cover",
    "drain grate",
    "storm drain",
]

# Русские названия групп — для консоли и отчётов.
INFRA_GROUP_RU_NAMES: Dict[str, str] = {
    "sign": "Дорожный знак",
    "pole": "Столб / опора",
    "traffic_light": "Светофор",
    "manhole": "Люк / решётка",
}

# К какой группе относится каждый текстовый класс.
INFRA_LABEL_GROUPS: Dict[str, str] = {
    "road sign": "sign",
    "traffic sign": "sign",
    "stop sign": "sign",
    "speed limit sign": "sign",
    "warning sign": "sign",
    "lamp post": "pole",
    "street light": "pole",
    "light pole": "pole",
    "pole": "pole",
    "utility pole": "pole",
    "electric pole": "pole",
    "telephone pole": "pole",
    "traffic light": "traffic_light",
    "manhole": "manhole",
    "manhole cover": "manhole",
    "drain grate": "manhole",
    "storm drain": "manhole",
}

# Палитра инфраструктуры (BGR). Намеренно не пересекается с цветами повреждений
# из yolo_pipeline.MODEL_KEY_COLORS_BGR (красный, пурпурный, жёлтый, голубой,
# зелёный) — иначе в совместном режиме объекты двух пайплайнов не различить.
INFRA_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "sign": (0, 165, 255),          # оранжевый
    "pole": (255, 0, 0),            # синий
    "traffic_light": (226, 43, 138),  # фиолетовый
    "manhole": (255, 255, 255),     # белый
}


def infra_group_for_label(label: str) -> str:
    return INFRA_LABEL_GROUPS.get(label.strip().lower(), "other")


def color_for_infra_label(label: str) -> Tuple[int, int, int]:
    group = infra_group_for_label(label)
    if group in INFRA_COLORS_BGR:
        return INFRA_COLORS_BGR[group]
    # Фолбэк для классов вне карты: стабильный по имени, в тёплой части спектра,
    # чтобы не совпасть с зелёным/голубым/жёлтым у повреждений.
    digest = sum(ord(c) for c in label)
    hsv = np.uint8([[[(digest * 23) % 30, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


@dataclass
class Detection:
    label: str
    score: float
    box_xyxy: np.ndarray


class SamOwlVitMaskPipeline:
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        owl_model_name: str = "google/owlvit-base-patch32",
        texts: Sequence[str] | None = None,
        threshold: float = 0.15,
        device: str | None = None,
    ) -> None:
        self.texts = list(texts) if texts else list(DEFAULT_TEXTS)
        if not self.texts:
            raise ValueError("Список текстовых классов пуст.")
        self.threshold = float(threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = OwlViTProcessor.from_pretrained(owl_model_name)
        self.detector = OwlViTForObjectDetection.from_pretrained(owl_model_name).to(self.device)
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(sam_model.to(self.device))

    def detect_boxes(self, frame_bgr: np.ndarray) -> List[Detection]:
        frame_rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(frame_rgb)
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)

        height, width = frame_rgb.shape[:2]
        target_sizes = torch.tensor([(height, width)], device=self.device)
        results = self._post_process_detections(outputs, inputs, target_sizes)

        detections: List[Detection] = []
        for score_t, label_t, box_t in zip(results["scores"], results["labels"], results["boxes"]):
            score = self._to_float(score_t)
            if score < self.threshold:
                continue
            label_name = self._resolve_label(label_t)
            box = self._to_numpy_box(box_t)
            box = self._sanitize_box(box, width, height)
            if box is None:
                continue
            detections.append(Detection(label=label_name, score=score, box_xyxy=box))
        return detections

    def _post_process_detections(
        self,
        outputs: Any,
        inputs: Any,
        target_sizes: torch.Tensor,
    ) -> Any:
        if hasattr(self.processor, "post_process_object_detection"):
            return self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes
            )[0]

        if hasattr(self.processor, "post_process_grounded_object_detection"):
            fn = self.processor.post_process_grounded_object_detection
            kwargs: dict[str, Any] = {
                "outputs": outputs,
                "target_sizes": target_sizes,
            }
            params = inspect.signature(fn).parameters

            if "threshold" in params:
                kwargs["threshold"] = self.threshold
            if "box_threshold" in params:
                kwargs["box_threshold"] = self.threshold
            if "text_threshold" in params:
                kwargs["text_threshold"] = self.threshold
            if "input_ids" in params and "input_ids" in inputs:
                kwargs["input_ids"] = inputs["input_ids"]
            if "text_labels" in params:
                kwargs["text_labels"] = [self.texts]

            return fn(**kwargs)[0]

        raise AttributeError(
            "Processor не поддерживает post_process_object_detection/"
            "post_process_grounded_object_detection"
        )

    def _resolve_label(self, label_value: Any) -> str:
        if isinstance(label_value, torch.Tensor):
            label_value = label_value.detach().cpu().item()

        if isinstance(label_value, (int, np.integer)):
            label_id = int(label_value)
            if 0 <= label_id < len(self.texts):
                return self.texts[label_id]
            return f"label_{label_id}"

        return str(label_value)

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _to_numpy_box(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    def segment_boxes(self, frame_bgr: np.ndarray, boxes_xyxy: Iterable[np.ndarray]) -> List[np.ndarray]:
        frame_rgb = frame_bgr[:, :, ::-1]
        self.sam_predictor.set_image(frame_rgb)
        masks: List[np.ndarray] = []
        for box in boxes_xyxy:
            try:
                raw_masks, sam_scores, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(box, dtype=np.float32)[None, :],
                    multimask_output=True,
                )
            except Exception:
                continue
            if raw_masks.size == 0:
                continue
            best_idx = int(np.argmax(sam_scores))
            masks.append(raw_masks[best_idx].astype(bool))
        return masks

    @staticmethod
    def build_binary_mask(masks: Sequence[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
        height, width = shape_hw
        merged = np.zeros((height, width), dtype=np.uint8)
        for mask in masks:
            if mask.shape != merged.shape:
                continue
            merged[mask] = 255
        return merged

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        detections = self.detect_boxes(frame_bgr)
        masks = self.segment_boxes(frame_bgr, (d.box_xyxy for d in detections))
        mask = self.build_binary_mask(masks, frame_bgr.shape[:2])
        return mask, detections

    @staticmethod
    def _sanitize_box(box_xyxy: np.ndarray, width: int, height: int) -> np.ndarray | None:
        x1, y1, x2, y2 = box_xyxy
        x1 = np.clip(x1, 0, width - 1)
        x2 = np.clip(x2, 0, width - 1)
        y1 = np.clip(y1, 0, height - 1)
        y2 = np.clip(y2, 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)
