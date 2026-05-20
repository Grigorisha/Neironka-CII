#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запись видео с веб-камеры в MP4 (H.264, yuv420p, CFR)."
    )
    parser.add_argument("--output", default="webcam_record.mp4", help="Путь к итоговому MP4.")
    parser.add_argument("--duration", type=int, default=15, help="Длительность записи в секундах.")
    parser.add_argument("--fps", type=float, default=25.0, help="Частота кадров (CFR).")
    parser.add_argument("--camera", type=int, default=0, help="Индекс веб-камеры.")
    parser.add_argument("--width", type=int, default=1280, help="Ширина кадра.")
    parser.add_argument("--height", type=int, default=720, help="Высота кадра.")
    return parser.parse_args()


def transcode_to_h264(input_path: Path, output_path: Path, fps: float) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_raw = output_path.with_name(output_path.stem + "_raw.mp4")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру.", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height

    writer = cv2.VideoWriter(
        str(temp_raw),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        print("Ошибка: не удалось создать файл для записи видео.", file=sys.stderr)
        cap.release()
        return 1

    print(f"Запись началась: {args.duration} сек, {args.fps} FPS.")
    print("Нажмите 'q' в окне предпросмотра для досрочной остановки.")

    start = time.time()
    while (time.time() - start) < args.duration:
        ok, frame = cap.read()
        if not ok:
            print("Предупреждение: кадр не получен, запись остановлена.")
            break
        writer.write(frame)
        cv2.imshow("Webcam recording (press q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if transcode_to_h264(temp_raw, output_path, args.fps):
        temp_raw.unlink(missing_ok=True)
        print(f"Готово: {output_path}")
        print("Формат: MP4 / H.264 / yuv420p / CFR")
        return 0

    fallback_path = output_path.with_name(output_path.stem + "_fallback.mp4")
    temp_raw.replace(fallback_path)
    print("ffmpeg не найден или конвертация не удалась.", file=sys.stderr)
    print(f"Сохранен fallback-файл: {fallback_path}", file=sys.stderr)
    print("Установите ffmpeg и перекодируйте в H.264/yuv420p.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
