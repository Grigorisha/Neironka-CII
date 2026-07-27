#!/usr/bin/env python3
"""Обработка одиночного кадра: на вход путь к изображению, на выход одна картинка.

Тонкая обёртка над FrameProcessor (frame_processor.py). Для встраивания в ROS
или другой долгоживущий процесс используйте класс напрямую — этот скрипт грузит
модели заново при каждом запуске (~7 секунд), см. docs/INTEGRATION_ROS.md.

  Группа 1 — что показывать (обязательна, ровно один флаг):
    --boxes-damage   рамки повреждений покрытия поверх оригинала (5 YOLO-моделей)
    --boxes-infra    рамки знаков, столбов, светофоров, люков поверх оригинала (OWL-ViT)
    --boxes-all      рамки обоих источников на одном кадре
    --masks          чёрный фон, цветные объекты; чем заполнять — группа 2

  Группа 2 — каким пайплайном (только вместе с --masks, ровно один флаг):
    --pipe1 / --pipe-seg-all       повреждения масками SAM  + инфраструктура масками SAM
    --pipe2 / --pipe-box-all       повреждения заливкой бокса + инфраструктура масками SAM
    --pipe3 / --pipe-seg-damage    только повреждения, масками SAM
    --pipe4 / --pipe-seg-infra     только инфраструктура, масками SAM
    --pipe5 / --pipe-box-damage    только повреждения, заливкой бокса
    --pipe6 / --pipe-box-infra     только инфраструктура, заливкой бокса

  Эквивалентная явная форма (выбор по сути двумерный):
    --damage {seg,box,off} --infra {seg,box,off}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CALIB_APPLY_DIR = Path(__file__).resolve().parent / "undistortion" / "apply"
if str(CALIB_APPLY_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_APPLY_DIR))

from frame_processor import PIPE_NUMBERS, PIPE_PRESETS, FrameProcessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обработка одного кадра детекторами дорожных дефектов и инфраструктуры.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Путь к входному изображению.")
    parser.add_argument("--output", required=True, help="Путь к выходному изображению (.png).")

    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--boxes-damage", dest="kind", action="store_const", const="boxes-damage",
                      help="Рамки повреждений покрытия поверх оригинала.")
    kind.add_argument("--boxes-infra", dest="kind", action="store_const", const="boxes-infra",
                      help="Рамки знаков/столбов/светофоров/люков поверх оригинала.")
    kind.add_argument("--boxes-all", dest="kind", action="store_const", const="boxes-all",
                      help="Рамки обоих источников поверх оригинала.")
    kind.add_argument("--masks", dest="kind", action="store_const", const="masks",
                      help="Маски: чёрный фон, цветные объекты.")

    pipe = parser.add_mutually_exclusive_group()
    for num, name in PIPE_NUMBERS.items():
        pipe.add_argument(f"--{num}", dest="pipe", action="store_const", const=name,
                          help=f"То же, что --pipe-{name}.")
    for name in PIPE_PRESETS:
        pipe.add_argument(f"--pipe-{name}", dest="pipe", action="store_const", const=name,
                          help=argparse.SUPPRESS)

    parser.add_argument("--damage", choices=["seg", "box", "off"],
                        help="Явная форма: как показывать повреждения покрытия.")
    parser.add_argument("--infra", choices=["seg", "box", "off"],
                        help="Явная форма: как показывать инфраструктуру.")

    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Порог уверенности YOLO (повреждения). По умолчанию 0.25.")
    parser.add_argument("--infra-threshold", type=float, default=0.15,
                        help="Порог уверенности OWL-ViT (инфраструктура). По умолчанию 0.15.")
    parser.add_argument("--weights-dir", default="",
                        help="Папка с .pt весами YOLO. Пусто = YOLO_WEIGHTS_DIR из local_config.py.")
    parser.add_argument("--device", default="", help="cuda | cpu. Пусто = автоопределение.")
    parser.add_argument("--calib", default="",
                        help="Файл калибровки OpenCV YAML — устранить дисторсию перед обработкой.")
    return parser.parse_args()


def load_calibration_or_die(path_str: str):
    from calibration_io import load_calibration

    calib_path = Path(path_str).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Файл калибровки не найден: {calib_path}")
    return load_calibration(calib_path)


def run() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Ошибка: входной файл не найден: {input_path}", file=sys.stderr)
        return 1
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Ошибка: не удалось прочитать изображение: {input_path}", file=sys.stderr)
        return 1

    if args.calib:
        try:
            calib = load_calibration_or_die(args.calib)
        except Exception as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            return 1
        h, w = frame.shape[:2]
        if (w, h) != (calib.image_width, calib.image_height):
            print(f"Ошибка: размер кадра {w}x{h} не совпадает с калибровкой "
                  f"{calib.image_width}x{calib.image_height}", file=sys.stderr)
            return 1
        frame = calib.undistort(frame)

    try:
        processor = FrameProcessor(
            kind=args.kind, pipe=args.pipe, damage=args.damage, infra=args.infra,
            weights_dir=args.weights_dir or None,
            threshold=args.threshold, infra_threshold=args.infra_threshold,
            device=args.device or None,
        )
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2

    result = processor.process(frame)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result.image):
        print(f"Ошибка: не удалось сохранить {output_path}", file=sys.stderr)
        return 1

    print(f"Режим: {args.kind} (повреждения={processor.damage_mode}, "
          f"инфраструктура={processor.infra_mode})")
    print(f"Найдено: повреждений {len(result.damage_items)}, "
          f"инфраструктуры {len(result.infra_items)}")
    for item in result.items:
        print(f"  [{item.source}] {item.label}: {item.score:.3f}")
    print(f"Сохранено: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
