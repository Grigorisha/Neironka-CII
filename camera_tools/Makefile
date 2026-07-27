PYTHON ?= /home/orin/workspace/.venvs/detection/bin/python

VIEWER_SCRIPT ?= simple_camera_viewer.py
CALIB_CAPTURE_SCRIPT ?= undistortion/scripts/capture_calib_photos_interactive.py
CALIB_SOLVER_SCRIPT ?= undistortion/calibrate/camera_calibration.py
RECORD_SCRIPT ?= record_realsense_raw.py

DEVICE ?= /dev/video4
WIDTH ?= 1280
HEIGHT ?= 720
FPS ?= 30
WINDOW_BACKEND ?= opencv
DISPLAY_VAR ?= :1
XAUTHORITY_VAR ?= /home/orin/.Xauthority

# Отдельные переменные для record-raw: пусто по умолчанию, чтобы полагаться на
# автоопределение RGB-ноды в самом скрипте, а не на DEVICE=/dev/video4 выше
# (это значение подобрано под другие тулзы/машину и не всегда указывает на RGB-сенсор).
RECORD_DEVICE ?=
RECORD_OUTPUT_DIR ?= ../outputs/recordings
RECORD_WIDTH ?= 1920
RECORD_HEIGHT ?= 1080
RECORD_FPS ?= 6

CALIB_FILE ?= undistortion/config/camera_calib.yml
CALIB_PHOTOS_DIR ?= ../outputs/calib_photos
CALIB_PREFIX ?= calib
CALIB_EXT ?= jpg
CALIB_MAX_SHOTS ?= 0
PATTERN_SIZE ?= 9x6
SQUARE_SIZE ?= 0.025
MIN_IMAGES ?= 12
SHOW_DETECTIONS ?= 0

.PHONY: help camera-list camera-on calib-test calib-capture calib-solve calib-solve-gui record-raw

help:
	@echo "Camera tools Make targets:"
	@echo "  make camera-list   - Показать доступные /dev/video*"
	@echo "  make camera-on     - Включить камеру (live preview)"
	@echo "  make calib-test    - Тест калибровки с переключением ON/OFF (клавиша c)"
	@echo "  make calib-capture - Съемка калибровочных фото (preview, save по SPACE/s)"
	@echo "  make calib-solve   - Расчет коэффициентов калибровки по снимкам (headless)"
	@echo "  make calib-solve-gui - Калибровка с показом найденных углов (нужен DISPLAY)"
	@echo "  make record-raw    - Превью + запись видео без сжатия (RGB RealSense, макс. разрешение)"
	@echo ""
	@echo "Клавиши:"
	@echo "  camera-on: q для выхода"
	@echo "  calib-test: c включить/выключить калибровку, q для выхода"
	@echo "  calib-capture: SPACE/s сохранить кадр, q/ESC выход"
	@echo "  record-raw: SPACE старт/стоп записи, q для выхода"
	@echo ""
	@echo "Основные параметры:"
	@echo "  DEVICE=$(DEVICE)"
	@echo "  WIDTH=$(WIDTH) HEIGHT=$(HEIGHT) FPS=$(FPS)"
	@echo "  CALIB_FILE=$(CALIB_FILE)"
	@echo "  CALIB_PHOTOS_DIR=$(CALIB_PHOTOS_DIR)"
	@echo "  PATTERN_SIZE=$(PATTERN_SIZE) SQUARE_SIZE=$(SQUARE_SIZE) MIN_IMAGES=$(MIN_IMAGES)"
	@echo "  SHOW_DETECTIONS=$(SHOW_DETECTIONS)"
	@echo ""
	@echo "Примеры:"
	@echo "  make camera-on DEVICE=/dev/video4"
	@echo "  make calib-test DEVICE=/dev/video4 CALIB_FILE=undistortion/config/camera_calib.yml"
	@echo "  make calib-capture CALIB_PHOTOS_DIR=../outputs/calib_photos CALIB_MAX_SHOTS=25"
	@echo "  make calib-solve CALIB_PHOTOS_DIR=../outputs/calib_photos PATTERN_SIZE=9x6 SQUARE_SIZE=0.025"
	@echo "  make calib-solve-gui DISPLAY_VAR=:1 XAUTHORITY_VAR=/home/orin/.Xauthority"

camera-list:
	$(PYTHON) "$(VIEWER_SCRIPT)" --list

camera-on:
	DISPLAY="$(DISPLAY_VAR)" XAUTHORITY="$(XAUTHORITY_VAR)" \
	$(PYTHON) "$(VIEWER_SCRIPT)" \
		$(if $(DEVICE),--device "$(DEVICE)",) \
		--window-backend "$(WINDOW_BACKEND)" \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)" \
		--fps "$(FPS)"

calib-test:
	DISPLAY="$(DISPLAY_VAR)" XAUTHORITY="$(XAUTHORITY_VAR)" \
	$(PYTHON) "$(VIEWER_SCRIPT)" \
		$(if $(DEVICE),--device "$(DEVICE)",) \
		--window-backend opencv \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)" \
		--fps "$(FPS)" \
		--calib "$(CALIB_FILE)"

calib-capture:
	@mkdir -p "$(CALIB_PHOTOS_DIR)"
	DISPLAY="$(DISPLAY_VAR)" XAUTHORITY="$(XAUTHORITY_VAR)" \
	$(PYTHON) "$(CALIB_CAPTURE_SCRIPT)" \
		"$(CALIB_PHOTOS_DIR)" \
		$(if $(DEVICE),--device "$(DEVICE)",) \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)" \
		--prefix "$(CALIB_PREFIX)" \
		--ext "$(CALIB_EXT)" \
		--max-shots "$(CALIB_MAX_SHOTS)"

calib-solve:
	@mkdir -p "$$(dirname "$(CALIB_FILE)")"
	$(PYTHON) "$(CALIB_SOLVER_SCRIPT)" \
		--images "$(CALIB_PHOTOS_DIR)/$(CALIB_PREFIX)_*.$(CALIB_EXT)" \
		--pattern_size "$(PATTERN_SIZE)" \
		--square_size "$(SQUARE_SIZE)" \
		--min_images "$(MIN_IMAGES)" \
		$(if $(filter 1 true yes on,$(SHOW_DETECTIONS)),--show,) \
		--out "$(CALIB_FILE)"

calib-solve-gui:
	@mkdir -p "$$(dirname "$(CALIB_FILE)")"
	DISPLAY="$(DISPLAY_VAR)" XAUTHORITY="$(XAUTHORITY_VAR)" \
	$(PYTHON) "$(CALIB_SOLVER_SCRIPT)" \
		--images "$(CALIB_PHOTOS_DIR)/$(CALIB_PREFIX)_*.$(CALIB_EXT)" \
		--pattern_size "$(PATTERN_SIZE)" \
		--square_size "$(SQUARE_SIZE)" \
		--min_images "$(MIN_IMAGES)" \
		--show \
		--out "$(CALIB_FILE)"

record-raw:
	@mkdir -p "$(RECORD_OUTPUT_DIR)"
	DISPLAY="$(DISPLAY_VAR)" XAUTHORITY="$(XAUTHORITY_VAR)" \
	$(PYTHON) "$(RECORD_SCRIPT)" \
		"$(RECORD_OUTPUT_DIR)" \
		$(if $(RECORD_DEVICE),--device "$(RECORD_DEVICE)",) \
		--width "$(RECORD_WIDTH)" \
		--height "$(RECORD_HEIGHT)" \
		--fps "$(RECORD_FPS)"
