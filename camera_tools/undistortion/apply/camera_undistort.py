#!/usr/bin/env python3
"""
ROS2-нода для исправления искажений камеры.

Подписка:  /camera/image_raw
Публикация: /camera/image_undistorted

Требует: rclpy, cv_bridge, sensor_msgs.
"""
import sys
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibration_io import load_calibration  # noqa: E402


class CameraUndistortNode(Node):
    def __init__(self):
        super().__init__("camera_undistort")

        self.declare_parameter("calib_file", "camera_calib.yml")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/camera/image_undistorted")

        calib_file = self.get_parameter("calib_file").get_parameter_value().string_value
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )

        self.bridge = CvBridge()
        self.calib = None

        try:
            self.calib = load_calibration(calib_file)
            self.get_logger().info(f"Calibration loaded from {calib_file}")
        except (FileNotFoundError, ValueError) as exc:
            self.get_logger().warn(str(exc))

        self.sub = self.create_subscription(Image, self.input_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(
            f"camera_undistort: input={self.input_topic} output={self.output_topic}"
        )

    def image_callback(self, msg: Image):
        if self.calib is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        undistorted = self.calib.undistort(frame)
        out_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding="bgr8")
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraUndistortNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
