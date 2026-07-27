# Калибровка камеры и устранение дисторсии (undistortion)

Модуль **офлайн-калибровки** камеры по шахматной доске и **исправления дисторсии** кадров.
Рассчитан на камеру **Intel RealSense**; ROS2-часть опциональна.

Это внутренняя (intrinsic) калибровка камеры — она вычисляет матрицу камеры и коэффициенты
дисторсии и применяет их для устранения искажений объектива. Не путать с внешней калибровкой
лидар↔камера (экстринсики), которая выполняется отдельным модулем системы.

## Структура

```
undistortion/
├── calibrate/
│   └── camera_calibration.py    # офлайн: фото → camera_calib.yml
├── apply/
│   ├── calibration_io.py        # загрузка YAML + remap (используется всем пайплайном)
│   ├── undistort_image.py       # один кадр без ROS
│   ├── camera_undistort.py      # ROS2-нода (опционально)
│   └── opencv_to_ros_camera_info.py
├── config/
│   └── camera_calib.yml         # результат калибровки (создаётся вами)
├── scripts/
│   ├── capture_calib_photos_interactive.py  # съёмка с preview + сохранение по кнопке
│   └── capture_calib_photos.sh              # обёртка для съёмки серии снимков
└── README.md
```

## Быстрый старт (RealSense, без ROS)

### 1. Снять калибровочные фото (live preview + снимок по кнопке)

Нужна печатная шахматная доска. Узнайте число **внутренних углов** (не клеток), например 9×6.

```bash
chmod +x scripts/capture_calib_photos.sh
./scripts/capture_calib_photos.sh /home/orin/workspace/detection/outputs/calib_photos 20
```

Управление в окне:

- `SPACE`/`s` - сохранить кадр
- `q`/`ESC` - завершить съемку

Или напрямую Python-скриптом:

```bash
python3 scripts/capture_calib_photos_interactive.py /home/orin/workspace/detection/outputs/calib_photos --max-shots 20
```

**Важно:** разрешение при съёмке и при использовании калибровки должно совпадать (например всегда 1280×720).

### 2. Запустить калибровку

```bash
python3 calibrate/camera_calibration.py \
  --images "/home/orin/workspace/detection/outputs/calib_photos/calib_*.jpg" \
  --pattern_size 9x6 \
  --square_size 0.025 \
  --min_images 12 \
  --show \
  --out config/camera_calib.yml
```

Параметры:

| Параметр | Описание |
|----------|----------|
| `--pattern_size` | Внутренние углы доски, `WxH` |
| `--square_size` | Размер клетки в **метрах** (измерьте линейкой) |
| `--min_images` | Минимум удачных кадров (рекомендуется 15–25) |

В выводе смотрите `RMS` и `Mean reprojection error` — чем меньше, тем лучше (ориентир: reprojection &lt; 0.5 px).

### 3. Проверить undistort на одном фото

```bash
python3 apply/undistort_image.py \
  --calib config/camera_calib.yml \
  --input /path/to/test.jpg \
  --output /path/to/test_undistorted.jpg
```

### 4. Использовать в своем коде detection

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".../undistortion/apply")))
from calibration_io import load_calibration

calib = load_calibration("config/camera_calib.yml")
frame_undistorted = calib.undistort(frame_bgr)
```

В текущих `camera_tools` калибровка подключается через обязательный аргумент `--calib`.

## ROS2 (если понадобится)

1. **Публикация camera_info** — сгенерировать из калибровки OpenCV:

   ```bash
   python3 apply/opencv_to_ros_camera_info.py \
     --calib config/camera_calib.yml \
     --out config/realsense_camera_info.yaml \
     --camera_name realsense_rgb
   ```

2. **Undistort в топике** (нужны `rclpy`, `cv_bridge`):

   ```bash
   python3 apply/camera_undistort.py --ros-args \
     -p calib_file:=/home/orin/workspace/detection/camera_tools/undistortion/config/camera_calib.yml \
     -p input_topic:=/camera/image_raw
   ```

RealSense SDK может отдавать заводскую калибровку по depth/RGB — для **2D-детекции на RGB** всё равно
полезна калибровка именно того потока и разрешения, которым вы пользуетесь (V4L2 `/dev/video*`).

## Зависимости

```bash
pip install opencv-python numpy pyyaml   # pyyaml только для opencv_to_ros_camera_info.py
# ROS2: rclpy, cv_bridge, sensor_msgs
```

## Ссылки

- [Nav2 camera calibration tutorial](https://docs.nav2.org/tutorials/docs/camera_calibration.html)
