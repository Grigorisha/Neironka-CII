#!/usr/bin/env python3
"""
Simple V4L2 camera viewer for Jetson/Linux.

Usage:
  python3 simple_camera_viewer.py
  python3 simple_camera_viewer.py --device /dev/video0
  python3 simple_camera_viewer.py --list
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import time
from typing import List, Optional, Tuple

import cv2


def list_video_devices() -> List[Tuple[str, str]]:
    devices = []
    for dev in sorted(glob.glob("/dev/video*")):
        name_path = f"/sys/class/video4linux/{os.path.basename(dev)}/name"
        name = "Unknown camera"
        if os.path.exists(name_path):
            try:
                with open(name_path, "r", encoding="utf-8") as f:
                    name = f.read().strip()
            except OSError:
                pass
        devices.append((dev, name))
    return devices


def open_first_working_device(preferred_device: Optional[str]) -> Tuple[cv2.VideoCapture, str]:
    candidates = [preferred_device] if preferred_device else [d for d, _ in list_video_devices()]

    if not candidates:
        raise RuntimeError("No /dev/video* devices found.")

    for dev in candidates:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, _frame = cap.read()
            if ok:
                return cap, dev
        cap.release()

    raise RuntimeError(f"Could not open a working stream from: {', '.join(candidates)}")


def run_gstreamer_window(device: str, width: int, height: int, fps: int) -> int:
    if shutil.which("gst-launch-1.0") is None:
        print("Error: gst-launch-1.0 is not installed.")
        print("Install with: sudo apt-get install -y gstreamer1.0-tools")
        return 1

    # Generic V4L2 pipeline that works with most UVC streams.
    cmd = [
        "gst-launch-1.0",
        "v4l2src",
        f"device={device}",
        "!",
        f"video/x-raw,width={width},height={height},framerate={fps}/1",
        "!",
        "videoconvert",
        "!",
        "autovideosink",
    ]
    print("Starting GStreamer window. Press Ctrl+C to stop.")
    print(" ".join(cmd))
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def infer_display_from_xauthority(xauthority_path: str) -> Optional[str]:
    if not os.path.exists(xauthority_path):
        return None
    try:
        result = subprocess.run(
            ["xauth", "-f", xauthority_path, "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    displays = []
    for line in result.stdout.splitlines():
        # Examples: "hostname:1  MIT-MAGIC-COOKIE-1 ...", "hostname/unix:1 ..."
        m = re.search(r":(\d+)\s+MIT-MAGIC-COOKIE-1", line)
        if m:
            displays.append(int(m.group(1)))
    if not displays:
        return None
    return f":{min(displays)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple camera viewer")
    parser.add_argument("--device", help="Camera device path, e.g. /dev/video0")
    parser.add_argument("--width", type=int, default=640, help="Requested frame width")
    parser.add_argument("--height", type=int, default=480, help="Requested frame height")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS")
    parser.add_argument("--list", action="store_true", help="List available cameras and exit")
    parser.add_argument(
        "--window-backend",
        choices=["opencv", "gstreamer"],
        default="opencv",
        help="Backend for realtime window output",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI and save stream to file")
    parser.add_argument("--output", default="/tmp/camera_capture.avi", help="Output video path for headless mode")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames to capture in headless mode")
    args = parser.parse_args()

    devices = list_video_devices()
    if args.list:
        if not devices:
            print("No /dev/video* devices found.")
            return 1
        print("Available cameras:")
        for dev, name in devices:
            print(f"  {dev:12s}  {name}")
        return 0

    if not devices:
        print("No camera devices found.")
        return 1

    try:
        cap, opened_device = open_first_working_device(args.device)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print(f"Opened: {opened_device}")
    no_display = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    headless = args.headless or no_display
    if headless:
        if no_display and not args.headless:
            print("No GUI session detected. Switching to headless capture mode.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        fps = cap.get(cv2.CAP_PROP_FPS) or float(args.fps)
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"MJPG"),
            max(1.0, float(fps)),
            (width, height),
        )
        if not writer.isOpened():
            print(f"Warning: failed to open output file {args.output}. Capturing frames without saving.")
            writer = None

        print(f"Capturing up to {args.max_frames} frames...")
        frame_count = 0
        start = time.time()
        while frame_count < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed. Exiting.")
                break
            if writer is not None:
                writer.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Captured {frame_count} frames")

        elapsed = max(0.001, time.time() - start)
        print(f"Done: {frame_count} frames in {elapsed:.2f}s ({frame_count / elapsed:.1f} FPS)")
        if writer is not None:
            writer.release()
            print(f"Saved: {args.output}")
        cap.release()
        return 0

    # Help non-GUI terminals (SSH/IDE) connect to an existing desktop session.
    if not os.environ.get("XAUTHORITY"):
        os.environ["XAUTHORITY"] = os.path.expanduser("~/.Xauthority")
    if not os.environ.get("DISPLAY"):
        inferred_display = infer_display_from_xauthority(os.environ["XAUTHORITY"])
        if inferred_display:
            os.environ["DISPLAY"] = inferred_display
            print(f"DISPLAY was not set. Using {inferred_display} from Xauthority.")

    if args.window_backend == "gstreamer":
        cap.release()
        return run_gstreamer_window(opened_device, args.width, args.height, args.fps)

    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed. Exiting.")
                break

            cv2.imshow("Simple Camera Viewer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except cv2.error as exc:
        print(f"OpenCV window backend failed: {exc}")
        print("Falling back to GStreamer window backend...")
        cap.release()
        cv2.destroyAllWindows()
        return run_gstreamer_window(opened_device, args.width, args.height, args.fps)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
