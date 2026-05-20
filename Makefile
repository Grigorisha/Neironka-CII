PYTHON ?= /home/orin/workspace/.venvs/detection/bin/python

RECORD_SCRIPT ?= camera_tools/record_webcam_mp4.py
MASK_VIDEO_SCRIPT ?= camera_tools/run_mask_pipeline.py
MASK_IMAGE_SCRIPT ?= camera_tools/run_mask_image.py
VIEWER_SCRIPT ?= camera_tools/simple_camera_viewer.py
RGB_PHOTO_SCRIPT ?= camera_tools/capture_realsense_rgb_photo.py

OUTPUT ?= recordings/webcam_$(shell date +%Y%m%d_%H%M%S).mp4
INPUT_VIDEO ?= recordings/webcam_20260515_190833.mp4
MASK_OUTPUT ?= outputs/mask_$(shell date +%Y%m%d_%H%M%S).mp4
INPUT_IMAGE ?= outputs/test_input_frame2.png
OUTPUT_DIR ?= outputs/single_image_masks
OUTPUT_NAME ?= instance_mask.png
FRAME_OUTPUT ?= outputs/frame_from_video.png
FRAME_IDX ?= 120

DURATION ?= 15
FPS ?= 10
CAMERA ?= 0
WIDTH ?= 1280
HEIGHT ?= 720
THRESHOLD ?= 0.15
TEXTS ?=
STREAM_URL ?= udp://127.0.0.1:5000?pkt_size=1316
STREAM_FORMAT ?= udp
DEVICE ?=
WINDOW_BACKEND ?= gstreamer
DISPLAY_VAR ?= :1
XAUTHORITY_VAR ?= /home/orin/.Xauthority
RGB_OUTPUT_DIR ?= outputs/rgb_snapshots

.PHONY: help record-webcam mask-video mask-camera extract-frame mask-image mask-image-from-video camera-list camera-viewer camera-rgb-photo

help:
	@echo "Удобные команды:"
	@echo "  make record-webcam         - Запись MP4 с камеры в recordings/"
	@echo "  make mask-video            - Маска видеофайла в outputs/*.mp4"
	@echo "  make mask-camera           - Маска с камеры в UDP/RTSP поток"
	@echo "  make extract-frame         - Извлечь PNG-кадр из видео"
	@echo "  make mask-image            - Маска одного PNG (цветные instance)"
	@echo "  make mask-image-from-video - Извлечь кадр и сразу построить его маску"
	@echo "  make camera-list           - Показать /dev/video* устройства"
	@echo "  make camera-viewer         - Просмотр камеры в реальном времени"
	@echo "  make camera-rgb-photo      - Снять 1 фото с RGB камеры RealSense"
	@echo ""
	@echo "Параметры (можно переопределять):"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  INPUT_VIDEO=$(INPUT_VIDEO)"
	@echo "  INPUT_IMAGE=$(INPUT_IMAGE)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  THRESHOLD=$(THRESHOLD)"
	@echo "  DEVICE=$(DEVICE)"
	@echo "  RGB_OUTPUT_DIR=$(RGB_OUTPUT_DIR)"
	@echo ""
	@echo "Примеры:"
	@echo "  make mask-image INPUT_IMAGE=outputs/test_input_frame2.png OUTPUT_DIR=outputs/single_image_masks"
	@echo "  make mask-video INPUT_VIDEO=recordings/webcam_20260515_190833.mp4 MASK_OUTPUT=outputs/mask.mp4"
	@echo "  make extract-frame INPUT_VIDEO=recordings/webcam_20260515_190833.mp4 FRAME_IDX=120 FRAME_OUTPUT=outputs/frame.png"
	@echo "  make camera-viewer DEVICE=/dev/video2 WINDOW_BACKEND=gstreamer"
	@echo "  make camera-rgb-photo RGB_OUTPUT_DIR=outputs/rgb_snapshots"

record-webcam:
	@mkdir -p "$$(dirname "$(OUTPUT)")"
	$(PYTHON) "$(RECORD_SCRIPT)" \
		--output "$(OUTPUT)" \
		--duration "$(DURATION)" \
		--fps "$(FPS)" \
		--camera "$(CAMERA)" \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)"

mask-video:
	@mkdir -p "$$(dirname "$(MASK_OUTPUT)")"
	$(PYTHON) "$(MASK_VIDEO_SCRIPT)" \
		--mode video \
		--input "$(INPUT_VIDEO)" \
		--output "$(MASK_OUTPUT)" \
		--fps "$(FPS)" \
		--threshold "$(THRESHOLD)" \
		--texts "$(TEXTS)"

mask-camera:
	$(PYTHON) "$(MASK_VIDEO_SCRIPT)" \
		--mode camera \
		--camera "$(CAMERA)" \
		--fps "$(FPS)" \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)" \
		--threshold "$(THRESHOLD)" \
		--stream-url "$(STREAM_URL)" \
		--stream-format "$(STREAM_FORMAT)" \
		--texts "$(TEXTS)"

extract-frame:
	@mkdir -p "$$(dirname "$(FRAME_OUTPUT)")"
	$(PYTHON) -c "import cv2; cap=cv2.VideoCapture('$(INPUT_VIDEO)'); cap.set(cv2.CAP_PROP_POS_FRAMES,$(FRAME_IDX)); ok,frame=cap.read(); cap.release(); assert ok, 'Не удалось извлечь кадр'; cv2.imwrite('$(FRAME_OUTPUT)', frame)"
	@echo "Готово: $(FRAME_OUTPUT)"

mask-image:
	@mkdir -p "$(OUTPUT_DIR)"
	$(PYTHON) "$(MASK_IMAGE_SCRIPT)" \
		--input-image "$(INPUT_IMAGE)" \
		--output-dir "$(OUTPUT_DIR)" \
		--output-name "$(OUTPUT_NAME)" \
		--threshold "$(THRESHOLD)" \
		--texts "$(TEXTS)"

mask-image-from-video: extract-frame
	@$(MAKE) mask-image INPUT_IMAGE="$(FRAME_OUTPUT)"

camera-list:
	$(PYTHON) "$(VIEWER_SCRIPT)" --list

camera-viewer:
	DISPLAY="$(DISPLAY_VAR)" XAUTHORITY="$(XAUTHORITY_VAR)" \
	$(PYTHON) "$(VIEWER_SCRIPT)" \
		$(if $(DEVICE),--device "$(DEVICE)",) \
		--window-backend "$(WINDOW_BACKEND)" \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)" \
		--fps "$(FPS)"

camera-rgb-photo:
	@mkdir -p "$(RGB_OUTPUT_DIR)"
	$(PYTHON) "$(RGB_PHOTO_SCRIPT)" "$(RGB_OUTPUT_DIR)" \
		$(if $(DEVICE),--device "$(DEVICE)",) \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)"
