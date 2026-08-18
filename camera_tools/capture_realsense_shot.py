#!/usr/bin/env python3
"""Одиночные снимки с RGB-камеры RealSense с точной меткой времени устройства.

Отличия от `record_realsense_raw.py` и `capture_realsense_rgb_photo.py`:

* Работа через SDK (`pyrealsense2`), а не через V4L2/OpenCV. Только SDK отдаёт
  метку времени из внутренних часов камеры и позволяет управлять выдержкой.
  Заодно снимается вопрос выбора ноды: поток `rs.stream.color` запрашивается
  явно, ИК-модуль подменить его не может.
* Частота по умолчанию — максимальная для выбранного разрешения (для D415 это
  30 Гц на 1920x1080). Частота ограничивает выдержку сверху и тем самым смаз.
* Сохраняется не видео, а отдельные кадры: пробел — снимок. Рядом с каждым
  кадром пишется JSON со всеми доступными метками времени, а в `index.csv`
  добавляется строка для последующей синхронизации с лидаром и координатами.

Смаз вызывается не частотой кадров, а выдержкой: при скорости v и масштабе
GSD (метров на пиксель) смаз равен v * t_exp / GSD пикселей. Поэтому по
умолчанию выдержка ограничивается сверху (`--ae-limit-us`), а при съёмке на
ходу выдержку лучше фиксировать вручную (`--exposure-us`).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - зависит от машины
    print(
        "Ошибка: не установлен pyrealsense2. Установка: pip install pyrealsense2",
        file=sys.stderr,
    )
    raise SystemExit(1)

CALIB_APPLY_DIR = Path(__file__).resolve().parent / "undistortion" / "apply"
if str(CALIB_APPLY_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_APPLY_DIR))

# Метки времени, которые пробуем прочитать из метаданных кадра.
# Единицы измерения приведены так, как их отдаёт librealsense.
METADATA_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("frame_timestamp", "us"),      # часы устройства
    ("sensor_timestamp", "us"),     # момент считывания сенсора
    ("backend_timestamp", "ms"),    # время ядра (UVC backend)
    ("time_of_arrival", "ms"),      # время прихода кадра на хост
    ("frame_counter", "count"),
    ("actual_exposure", "raw"),
    ("actual_fps", "raw"),
    ("gain_level", "raw"),
)

INDEX_COLUMNS = (
    "file",
    "unix_time_s",
    "timestamp_domain",
    "frame_timestamp_ms",
    "md_frame_timestamp_us",
    "md_sensor_timestamp_us",
    "md_time_of_arrival_ms",
    "frame_number",
    "md_frame_counter",
    "md_actual_exposure",
    "host_realtime_s",
    "host_monotonic_s",
    "width",
    "height",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Одиночные снимки с RGB-камеры RealSense. Пробел — снимок, Q — выход. "
            "Каждый кадр сохраняется без потерь вместе с меткой времени устройства."
        )
    )
    parser.add_argument("output_dir", nargs="?", help="Папка для снимков. Будет создана автоматически.")
    parser.add_argument("--serial", help="Серийный номер камеры, если подключено несколько.")
    parser.add_argument("--width", type=int, default=1920, help="Ширина кадра (по умолчанию максимум D415).")
    parser.add_argument("--height", type=int, default=1080, help="Высота кадра.")
    parser.add_argument(
        "--fps", type=int, default=0,
        help="Частота кадров. 0 (по умолчанию) — максимальная доступная для этого разрешения.",
    )
    parser.add_argument(
        "--exposure-us", type=int, default=0,
        help="Фиксированная выдержка в микросекундах. 0 — автоэкспозиция. Для съёмки на ходу 500–3000.",
    )
    parser.add_argument(
        "--gain", type=int, default=-1,
        help="Усиление при ручной выдержке (0–128). -1 — не трогать.",
    )
    parser.add_argument(
        "--ae-limit-us", type=int, default=8000,
        help="Верхний предел выдержки для автоэкспозиции, мкс. 0 — не ограничивать.",
    )
    parser.add_argument(
        "--warmup-frames", type=int, default=30,
        help="Сколько кадров пропустить на старте, чтобы сошлась автоэкспозиция.",
    )
    parser.add_argument("--burst", type=int, default=1, help="Сколько кадров подряд сохранять по одному нажатию.")
    parser.add_argument("--preview-width", type=int, default=960, help="Ширина окна превью (кадр не меняется).")
    parser.add_argument(
        "--once", action="store_true",
        help="Без окна: снять кадр(ы) сразу после прогрева и выйти. Для скриптов и SSH без X.",
    )
    parser.add_argument("--format", choices=("png", "tiff"), default="png", help="Формат снимка (оба без потерь).")
    parser.add_argument("--calib", help="Файл калибровки. Если задан, рядом сохраняется undistort-копия.")
    parser.add_argument("--list-profiles", action="store_true", help="Показать доступные режимы RGB-потока и выйти.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Устройство и профили
# --------------------------------------------------------------------------- #

def find_device(serial: Optional[str]) -> rs.device:
    devices = list(rs.context().query_devices())
    if not devices:
        raise RuntimeError(
            "Камера RealSense не найдена. Проверьте подключение (`rs-enumerate-devices`) "
            "и что камеру не занял другой процесс."
        )
    if serial:
        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == serial:
                return dev
        raise RuntimeError(f"Камера с серийным номером {serial} не найдена.")
    return devices[0]


def color_profiles(device: rs.device) -> List[Tuple[int, int, int, str]]:
    """Список (ширина, высота, fps, формат) для цветного сенсора."""
    profiles: List[Tuple[int, int, int, str]] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            video = profile.as_video_stream_profile()
            if video is None:
                continue
            profiles.append((video.width(), video.height(), profile.fps(), str(profile.format())))
    return sorted(set(profiles), key=lambda p: (-p[0] * p[1], -p[2]))


def resolve_fps(profiles: List[Tuple[int, int, int, str]], width: int, height: int, requested: int) -> int:
    available = sorted({fps for w, h, fps, _ in profiles if (w, h) == (width, height)}, reverse=True)
    if not available:
        sizes = sorted({(w, h) for w, h, _, _ in profiles}, key=lambda s: -s[0] * s[1])
        raise RuntimeError(
            f"Разрешение {width}x{height} камера не поддерживает. Доступны: "
            + ", ".join(f"{w}x{h}" for w, h in sizes)
        )
    if requested <= 0:
        return available[0]
    if requested not in available:
        raise RuntimeError(
            f"На {width}x{height} доступны только {available} Гц, запрошено {requested}."
        )
    return requested


def configure_color_sensor(device: rs.device, args: argparse.Namespace) -> rs.sensor:
    """Настраивает цветной сенсор: время, выдержка, усиление."""
    sensor = None
    for candidate in device.query_sensors():
        if any(p.stream_type() == rs.stream.color for p in candidate.get_stream_profiles()):
            sensor = candidate
            break
    if sensor is None:
        raise RuntimeError("У устройства нет цветного сенсора.")

    def try_set(option: rs.option, value: float, note: str) -> None:
        if not sensor.supports(option):
            print(f"  {note}: не поддерживается прошивкой — пропущено")
            return
        try:
            sensor.set_option(option, value)
        except Exception as exc:
            print(f"  {note}: не удалось установить ({exc})")

    # Метки времени в единой шкале с часами хоста. Без этого get_timestamp()
    # возвращает «сырые» часы устройства, не привязанные к календарному времени.
    try_set(rs.option.global_time_enabled, 1, "global_time_enabled")

    # Автоэкспозиции запрещаем ронять частоту кадров ради длинной выдержки.
    try_set(rs.option.auto_exposure_priority, 0, "auto_exposure_priority=0")

    if args.exposure_us > 0:
        try_set(rs.option.enable_auto_exposure, 0, "enable_auto_exposure=0")
        try_set(rs.option.exposure, float(args.exposure_us), f"exposure={args.exposure_us} мкс")
        if args.gain >= 0:
            try_set(rs.option.gain, float(args.gain), f"gain={args.gain}")
    else:
        try_set(rs.option.enable_auto_exposure, 1, "enable_auto_exposure=1")
        if args.ae_limit_us > 0:
            try_set(rs.option.auto_exposure_limit_toggle, 1, "auto_exposure_limit_toggle=1")
            try_set(rs.option.auto_exposure_limit, float(args.ae_limit_us),
                    f"auto_exposure_limit={args.ae_limit_us} мкс")
    return sensor


# --------------------------------------------------------------------------- #
# Метки времени
# --------------------------------------------------------------------------- #

def read_metadata(frame: rs.frame) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for name, unit in METADATA_FIELDS:
        value_key = getattr(rs.frame_metadata_value, name, None)
        if value_key is None:
            continue
        if not frame.supports_frame_metadata(value_key):
            continue
        try:
            data[name] = {"value": int(frame.get_frame_metadata(value_key)), "unit": unit}
        except Exception:
            continue
    return data


def collect_timestamps(frame: rs.frame, host_realtime_s: float, host_monotonic_s: float) -> Dict[str, Any]:
    domain = frame.get_frame_timestamp_domain()
    domain_name = str(domain)
    frame_ts_ms = float(frame.get_timestamp())
    metadata = read_metadata(frame)

    # get_timestamp() уже приведён к эпохе хоста, если librealsense смог
    # построить соответствие часов устройства и хоста (global_time).
    if domain == rs.timestamp_domain.global_time:
        unix_time_s: Optional[float] = frame_ts_ms / 1000.0
        time_source = "global_time: часы камеры, приведённые librealsense к времени хоста"
    elif domain == rs.timestamp_domain.system_time:
        unix_time_s = frame_ts_ms / 1000.0
        time_source = (
            "system_time: часы ХОСТА в момент приёма кадра. Часы камеры недоступны — "
            "скорее всего, ядро без патча метаданных UVC"
        )
    else:
        unix_time_s = None
        time_source = (
            "hardware_clock: сырые часы камеры, начало отсчёта произвольно. "
            "Для перевода в календарное время используйте host_realtime_s и смещение"
        )

    result: Dict[str, Any] = {
        "unix_time_s": unix_time_s,
        "unix_time_iso": (
            datetime.fromtimestamp(unix_time_s, tz=timezone.utc).isoformat() if unix_time_s else None
        ),
        "time_source": time_source,
        "timestamp_domain": domain_name,
        "frame_timestamp_ms": frame_ts_ms,
        "frame_number": int(frame.get_frame_number()),
        "host_realtime_s": host_realtime_s,
        "host_monotonic_s": host_monotonic_s,
        "host_minus_device_ms": (host_realtime_s * 1000.0 - frame_ts_ms) if unix_time_s else None,
        "metadata": metadata,
    }
    return result


def index_row(path: Path, stamps: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    md = stamps["metadata"]

    def md_value(key: str) -> Any:
        return md.get(key, {}).get("value")

    return {
        "file": path.name,
        "unix_time_s": f"{stamps['unix_time_s']:.6f}" if stamps["unix_time_s"] else "",
        "timestamp_domain": stamps["timestamp_domain"],
        "frame_timestamp_ms": f"{stamps['frame_timestamp_ms']:.3f}",
        "md_frame_timestamp_us": md_value("frame_timestamp"),
        "md_sensor_timestamp_us": md_value("sensor_timestamp"),
        "md_time_of_arrival_ms": md_value("time_of_arrival"),
        "frame_number": stamps["frame_number"],
        "md_frame_counter": md_value("frame_counter"),
        "md_actual_exposure": md_value("actual_exposure"),
        "host_realtime_s": f"{stamps['host_realtime_s']:.6f}",
        "host_monotonic_s": f"{stamps['host_monotonic_s']:.6f}",
        "width": width,
        "height": height,
    }


def append_index(index_path: Path, row: Dict[str, Any]) -> None:
    is_new = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------- #
# Сохранение
# --------------------------------------------------------------------------- #

class ShotWriter:
    def __init__(self, output_dir: Path, extension: str, calib: Any) -> None:
        self.output_dir = output_dir
        self.extension = extension
        self.calib = calib
        self.index_path = output_dir / "index.csv"
        self.count = 0

    def save(self, frame_bgr: np.ndarray, stamps: Dict[str, Any]) -> Path:
        now = datetime.now()
        base = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
        image_path = self.output_dir / f"shot_{base}.{self.extension}"
        if not cv2.imwrite(str(image_path), frame_bgr):
            raise RuntimeError(f"Не удалось сохранить {image_path}")

        payload = dict(stamps)
        payload["file"] = image_path.name
        payload["width"] = int(frame_bgr.shape[1])
        payload["height"] = int(frame_bgr.shape[0])
        image_path.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        append_index(self.index_path, index_row(image_path, stamps, frame_bgr.shape[1], frame_bgr.shape[0]))

        if self.calib is not None:
            undistorted = self.calib.undistort(frame_bgr)
            cv2.imwrite(str(self.output_dir / f"shot_{base}_undist.{self.extension}"), undistorted)

        self.count += 1
        return image_path


def load_calibration_or_none(path_str: Optional[str], width: int, height: int) -> Any:
    if not path_str:
        return None
    from calibration_io import load_calibration  # импорт здесь: нужен только с --calib

    calib_path = Path(path_str).expanduser().resolve()
    if not calib_path.exists():
        raise RuntimeError(f"Файл калибровки не найден: {calib_path}")
    calib = load_calibration(calib_path)
    if (calib.image_width, calib.image_height) != (width, height):
        raise RuntimeError(
            "Калибровка снята для другого разрешения: "
            f"кадр {width}x{height}, калибровка {calib.image_width}x{calib.image_height}"
        )
    return calib


# --------------------------------------------------------------------------- #

def report_metadata_support(stamps: Dict[str, Any]) -> None:
    print("\nМетки времени первого кадра:")
    print(f"  домен: {stamps['timestamp_domain']}")
    print(f"  источник: {stamps['time_source']}")
    available = sorted(stamps["metadata"].keys())
    print(f"  метаданные кадра: {', '.join(available) if available else 'недоступны'}")
    if "frame_timestamp" not in stamps["metadata"]:
        print(
            "  ВНИМАНИЕ: часы устройства недоступны. В Linux метаданные UVC требуют\n"
            "  патча ядра из librealsense (scripts/patch-realsense-ubuntu-lts.sh).\n"
            "  Без него метка времени берётся с часов хоста в момент приёма кадра —\n"
            "  это на десятки миллисекунд позже реальной съёмки и «плавает» под нагрузкой."
        )
    print()


def main() -> int:
    args = parse_args()

    try:
        device = find_device(args.serial)
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    profiles = color_profiles(device)

    if args.list_profiles:
        print(f"{device.get_info(rs.camera_info.name)} "
              f"s/n {device.get_info(rs.camera_info.serial_number)} "
              f"fw {device.get_info(rs.camera_info.firmware_version)}")
        print("Режимы RGB-потока (ширина x высота @ Гц, формат):")
        for width, height, fps, fmt in profiles:
            print(f"  {width}x{height} @ {fps:>3} Гц  {fmt}")
        return 0

    if not args.output_dir:
        print("Ошибка: не указана папка для снимков.", file=sys.stderr)
        return 1

    try:
        fps = resolve_fps(profiles, args.width, args.height, args.fps)
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        calib = load_calibration_or_none(args.calib, args.width, args.height)
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"Камера: {device.get_info(rs.camera_info.name)} "
          f"s/n {device.get_info(rs.camera_info.serial_number)}")
    print(f"Режим: {args.width}x{args.height} @ {fps} Гц")

    config = rs.config()
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, fps)

    pipeline = rs.pipeline()
    try:
        profile = pipeline.start(config)
    except Exception as exc:
        print(f"Ошибка: не удалось запустить поток ({exc})", file=sys.stderr)
        return 1

    # Настройки применяются после старта: часть опций (выдержка, предел
    # автоэкспозиции) драйвер сбрасывает при запуске потока.
    print("Настройка сенсора:")
    try:
        configure_color_sensor(profile.get_device(), args)
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        pipeline.stop()
        return 1

    writer = ShotWriter(output_dir, args.format, calib)
    window_name = "RealSense -- SPACE: snapshot, Q: quit"
    reported = False

    try:
        for _ in range(max(0, args.warmup_frames)):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            host_realtime_s = time.time()
            host_monotonic_s = time.monotonic()
            color = frames.get_color_frame()
            if not color:
                continue

            frame_bgr = np.asanyarray(color.get_data())
            stamps = collect_timestamps(color, host_realtime_s, host_monotonic_s)

            if not reported:
                report_metadata_support(stamps)
                reported = True

            if args.once:
                for _ in range(max(1, args.burst)):
                    path = writer.save(frame_bgr, stamps)
                    print(f"Снимок: {path}")
                    if args.burst > 1:
                        frames = pipeline.wait_for_frames()
                        color = frames.get_color_frame()
                        frame_bgr = np.asanyarray(color.get_data())
                        stamps = collect_timestamps(color, time.time(), time.monotonic())
                break

            scale = args.preview_width / frame_bgr.shape[1]
            display = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.putText(
                display, f"SPACE: shot ({writer.count})  Q: quit",
                (10, display.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                for shot in range(max(1, args.burst)):
                    if shot > 0:
                        frames = pipeline.wait_for_frames()
                        color = frames.get_color_frame()
                        frame_bgr = np.asanyarray(color.get_data())
                        stamps = collect_timestamps(color, time.time(), time.monotonic())
                    path = writer.save(frame_bgr, stamps)
                    ts = stamps["unix_time_iso"] or f"{stamps['frame_timestamp_ms']:.3f} ms (hw)"
                    print(f"Снимок {writer.count}: {path.name}  t={ts}")
            elif key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"Сохранено снимков: {writer.count}. Индекс: {writer.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
