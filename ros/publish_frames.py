#!/usr/bin/env python3
"""Публикует кадры в топик — источник картинок для ноды детекции.

Нужен потому, что ROS-драйвера камеры на этой машине нет: без него входной топик
пустой и ноду не на чем проверить. Умеет брать кадры из папки с изображениями
или из видеофайла.

Запуск:
    source /opt/ros2_humble/install/setup.bash
    ~/workspace/.venvs/detection/bin/python ros/publish_frames.py \
        --source outputs/recordings/some.avi --rate 6

    ~/workspace/.venvs/detection/bin/python ros/publish_frames.py \
        --source path/to/frames_dir --topic /camera/color/image_raw --loop
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mkv", ".mov"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Публикация кадров в ROS-топик.")
    parser.add_argument("--source", required=True,
                        help="Видеофайл или папка с изображениями.")
    parser.add_argument("--topic", default="/camera/color/image_raw",
                        help="Куда публиковать. По умолчанию /camera/color/image_raw.")
    parser.add_argument("--rate", type=float, default=6.0,
                        help="Частота публикации, Гц. По умолчанию 6 (темп съёмки).")
    parser.add_argument("--loop", action="store_true",
                        help="Зациклить источник.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Опубликовать не больше N кадров (0 = все).")
    parser.add_argument("--reliability", choices=["best_effort", "reliable"],
                        default="best_effort",
                        help="QoS. Должен совпадать с input_reliability в конфиге ноды.")
    parser.add_argument("--frame-id", default="camera", help="frame_id в заголовке.")
    return parser.parse_args()


def iter_frames(source: Path, loop: bool) -> Iterator[np.ndarray]:
    if source.is_dir():
        files: List[Path] = sorted(
            p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"В папке нет изображений: {source}")
        while True:
            for path in files:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img
            if not loop:
                return

    if source.suffix.lower() not in VIDEO_SUFFIXES and not source.is_file():
        raise FileNotFoundError(f"Источник не найден: {source}")

    while True:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
        if not loop:
            return


def bgr_to_imgmsg(img: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"Ошибка: источник не найден: {source}", file=sys.stderr)
        return 1

    policy = (QoSReliabilityPolicy.BEST_EFFORT if args.reliability == "best_effort"
              else QoSReliabilityPolicy.RELIABLE)
    qos = QoSProfile(reliability=policy, history=QoSHistoryPolicy.KEEP_LAST, depth=1)

    rclpy.init()
    node = Node("frame_publisher")
    pub = node.create_publisher(Image, args.topic, qos)
    node.get_logger().info(f"Источник: {source}")
    node.get_logger().info(f"Публикую в {args.topic} с частотой {args.rate} Гц")
    time.sleep(1.0)  # дать подписчикам обнаружиться

    period = 1.0 / max(0.1, args.rate)
    sent = 0
    try:
        for frame in iter_frames(source, args.loop):
            pub.publish(bgr_to_imgmsg(frame, node.get_clock().now().to_msg(), args.frame_id))
            sent += 1
            if sent % 10 == 0:
                node.get_logger().info(f"опубликовано кадров: {sent}")
            if args.limit and sent >= args.limit:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    finally:
        node.get_logger().info(f"Всего опубликовано: {sent}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
