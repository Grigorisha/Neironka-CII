# Camera Tools: использование

Этот документ описывает утилиты в `detection/camera_tools` для работы с камерой Intel RealSense.

**Калибровка камеры (шахматная доска, undistort):** см. `undistortion/README.md`.

Во всех рабочих скриптах `camera_tools` параметр `--calib` обязателен. Пример файла:

`/home/orin/workspace/detection/camera_tools/undistortion/config/camera_calib.yml`

## 1) Проверка доступных видеоустройств

```bash
python3 /home/orin/workspace/detection/camera_tools/simple_camera_viewer.py --list
```

Покажет все `/dev/video*`, которые видит система.

## 2) Просмотр видео в реальном времени

Рекомендуемый запуск для Jetson:

```bash
DISPLAY=:1 XAUTHORITY=/home/orin/.Xauthority \
python3 /home/orin/workspace/detection/camera_tools/simple_camera_viewer.py \
  --device /dev/video2 \
  --window-backend opencv \
  --calib /home/orin/workspace/detection/camera_tools/undistortion/config/camera_calib.yml
```

Если не знаете номер устройства, сначала выполните `--list`.

### Полезные параметры viewer

- `--device /dev/videoN` - выбрать устройство.
- `--window-backend opencv` - режим с применением калибровки (обязателен для undistort).
- `--width 1280 --height 720 --fps 30` - настройки потока.
- `--headless --output /tmp/camera_capture.avi --max-frames 300` - захват без окна в файл.
- `--calib /path/to/camera_calib.yml` - обязательный файл калибровки.

## 3) Снять одно фото именно с RGB-камеры RealSense

### Быстрая команда (рекомендуется)

```bash
/home/orin/workspace/detection/camera_tools/take_rgb_photo.sh \
  /home/orin/workspace/detection/outputs \
  /home/orin/workspace/detection/camera_tools/undistortion/config/camera_calib.yml
```

Аргумент - путь к директории, куда сохранить фото.  
Скрипт создаст папку, если ее нет.

### Прямой запуск Python-скрипта

```bash
python3 /home/orin/workspace/detection/camera_tools/capture_realsense_rgb_photo.py \
  /home/orin/workspace/detection/outputs \
  --calib /home/orin/workspace/detection/camera_tools/undistortion/config/camera_calib.yml
```

### Полезные параметры RGB-снимка

- `--device /dev/videoN` - явно выбрать устройство.
- `--width 1280 --height 720` - размер кадра.
- `--warmup-frames 10` - число прогревочных кадров перед сохранением.
- `--calib /path/to/camera_calib.yml` - обязательный файл калибровки.

## 4) Что делает RGB-снимок

Скрипт `capture_realsense_rgb_photo.py`:

1. Ищет ноды RealSense в `/sys/class/video4linux/video*`.
2. Пытается выбрать RGB-поток по форматам (`MJPG`, `YUYV`, и т.д.), а не depth.
3. Пробует несколько подходящих устройств, если первое занято.
4. Сохраняет JPG с именем вида `realsense_rgb_YYYYMMDD_HHMMSS.jpg`.

## 5) Частые проблемы

### Устройство занято

Симптом: `open failed` или `frame read failed`.

Решение:

- Остановить другие приложения, которые используют камеру (`realsense-viewer`, viewer-скрипты и т.д.).
- Повторить команду.

### Нет окна при запуске viewer

Симптом: ошибки вида `Could not open display`.

Решение:

- Запускать с корректными `DISPLAY` и `XAUTHORITY` (пример выше).
- Для вашей текущей конфигурации рабочий вариант: `DISPLAY=:1`.

## 6) Где находятся скрипты

- `simple_camera_viewer.py` - просмотр потока камеры.
- `capture_realsense_rgb_photo.py` - сохранить одно RGB фото.
- `take_rgb_photo.sh` - удобная обертка для команды фото.
- `record_webcam_mp4.py` - запись видео в MP4.
