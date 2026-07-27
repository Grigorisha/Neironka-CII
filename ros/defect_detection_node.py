#!/usr/bin/env python3
"""ROS 2 нода: берёт кадр из входного топика, публикует обработанный в выходной.

Режим обработки задаётся JSON-конфигом (см. ros/config/default.json), поэтому
менять пайплайн можно без правки кода.

Запуск:
    source /opt/ros2_humble/install/setup.bash
    ~/workspace/.venvs/detection/bin/python ros/defect_detection_node.py \
        --config ros/config/default.json

Почему так: rclpy собран для Python 3.8, наш venv — тоже Python 3.8, поэтому
интерпретатор venv видит rclpy через PYTHONPATH, выставленный setup.bash.

Обработка кадра занимает 0.4-1.9 с (замеры в docs/INTEGRATION_ROS.md), то есть
заведомо медленнее съёмки. Поэтому нода не пытается обработать каждый кадр:
приёмный колбэк только запоминает последний кадр, а обработка идёт в отдельном
потоке. Кадры, пришедшие во время обработки, вытесняются — очередь не растёт,
задержка не накапливается.

cv_bridge намеренно не используется: на этой машине он не установлен, а для
bgr8/rgb8 конвертация тривиальна и делается вручную.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

from frame_processor import FrameProcessor  # noqa: E402


DEFAULT_CONFIG: Dict[str, Any] = {
    "input_topic": "/camera/color/image_raw",
    "output_topic": "/defect_detection/image",
    "kind": "masks",
    "pipe": "pipe3",
    "damage": None,
    "infra": None,
    "threshold": 0.25,
    "infra_threshold": 0.15,
    "weights_dir": None,
    "device": None,
    "input_reliability": "best_effort",
    "output_reliability": "reliable",
    "log_every": 10,
}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if not path:
        return config
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        user_config = json.load(fh)
    unknown = set(user_config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Неизвестные ключи в конфиге: {sorted(unknown)}. "
                         f"Допустимые: {sorted(DEFAULT_CONFIG)}")
    config.update(user_config)
    return config


def qos_profile(reliability: str, depth: int = 1) -> QoSProfile:
    policy = (QoSReliabilityPolicy.BEST_EFFORT if reliability == "best_effort"
              else QoSReliabilityPolicy.RELIABLE)
    return QoSProfile(reliability=policy, history=QoSHistoryPolicy.KEEP_LAST, depth=depth)


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> numpy BGR. Учитывает выравнивание строк (msg.step)."""
    encoding = msg.encoding.lower()
    if encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"Поддерживаются только bgr8 и rgb8, получено {msg.encoding!r}")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    # msg.step — байт на строку, может быть больше width*3 из-за паддинга
    rows = buf.reshape(msg.height, msg.step)[:, : msg.width * 3]
    img = rows.reshape(msg.height, msg.width, 3)
    if encoding == "rgb8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)  # копия: буфер сообщения только для чтения


def bgr_to_imgmsg(img: np.ndarray, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


class DefectDetectionNode(Node):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__("defect_detection")
        self.config = config

        self.get_logger().info(
            f"Загрузка моделей (kind={config['kind']}, pipe={config['pipe']}, "
            f"damage={config['damage']}, infra={config['infra']})…"
        )
        t0 = time.time()
        self.processor = FrameProcessor(
            kind=config["kind"], pipe=config["pipe"],
            damage=config["damage"], infra=config["infra"],
            weights_dir=config["weights_dir"],
            threshold=config["threshold"], infra_threshold=config["infra_threshold"],
            device=config["device"],
        )
        self.get_logger().info(
            f"Модели загружены за {time.time() - t0:.1f} с "
            f"(повреждения={self.processor.damage_mode}, "
            f"инфраструктура={self.processor.infra_mode})"
        )

        self._lock = threading.Lock()
        self._pending: Optional[Image] = None      # последний непринятый кадр
        self._wake = threading.Event()
        self._running = True
        self._n_done = 0
        self._n_dropped = 0

        self.publisher = self.create_publisher(
            Image, config["output_topic"], qos_profile(config["output_reliability"]))
        self.subscription = self.create_subscription(
            Image, config["input_topic"], self.on_frame,
            qos_profile(config["input_reliability"]))

        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"Вход: {config['input_topic']} -> выход: {config['output_topic']}")

    def on_frame(self, msg: Image) -> None:
        """Колбэк не обрабатывает кадр, только запоминает последний."""
        with self._lock:
            if self._pending is not None:
                self._n_dropped += 1     # предыдущий не успели взять — вытесняем
            self._pending = msg
        self._wake.set()

    def _work_loop(self) -> None:
        while self._running:
            if not self._wake.wait(timeout=0.5):
                continue
            self._wake.clear()
            with self._lock:
                msg, self._pending = self._pending, None
            if msg is None:
                continue
            try:
                self._process_and_publish(msg)
            except Exception as exc:  # нода не должна падать из-за одного кадра
                self.get_logger().error(f"Ошибка обработки кадра: {exc}")

    def _process_and_publish(self, msg: Image) -> None:
        t0 = time.time()
        frame = imgmsg_to_bgr(msg)
        result = self.processor.process(frame)
        self.publisher.publish(bgr_to_imgmsg(result.image, msg.header))

        self._n_done += 1
        every = max(1, int(self.config["log_every"]))
        if self._n_done % every == 0:
            elapsed = time.time() - t0
            self.get_logger().info(
                f"кадров обработано {self._n_done}, пропущено {self._n_dropped} | "
                f"последний {elapsed:.2f} с (~{1 / elapsed:.1f} FPS) | "
                f"повреждений {len(result.damage_items)}, "
                f"инфраструктуры {len(result.infra_items)}"
            )

    def destroy_node(self) -> bool:
        self._running = False
        self._wake.set()
        self._worker.join(timeout=5.0)
        return super().destroy_node()


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS 2 нода детекции дорожных дефектов.")
    parser.add_argument("--config", default="", help="Путь к JSON-конфигу.")
    args, ros_args = parser.parse_known_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2

    rclpy.init(args=ros_args)
    node = None
    try:
        node = DefectDetectionNode(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
