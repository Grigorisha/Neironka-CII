PYTHON ?= python3
SCRIPT ?= record_webcam_mp4.py
MASK_SCRIPT ?= run_mask_pipeline.py

OUTPUT ?= recordings/webcam_$(shell date +%Y%m%d_%H%M%S).mp4
DURATION ?= 15
FPS ?= 25
CAMERA ?= 0
WIDTH ?= 1280
HEIGHT ?= 720
INPUT ?= recordings/webcam_$(shell date +%Y%m%d_%H%M%S).mp4
MASK_OUTPUT ?= outputs/mask_$(shell date +%Y%m%d_%H%M%S).mp4
STREAM_URL ?= udp://127.0.0.1:5000?pkt_size=1316
STREAM_FORMAT ?= udp
THRESHOLD ?= 0.15
TEXTS ?=

.PHONY: help record mask-video mask-camera

help:
	@echo "Targets:"
	@echo "  make record        - Записать видео с веб-камеры в MP4 (H.264/yuv420p/CFR)"
	@echo "  make mask-video    - Обработать видеофайл и сохранить видео бинарной маски"
	@echo "  make mask-camera   - Обработать камеру и отправить маску в RTSP/UDP поток"
	@echo ""
	@echo "Пример с параметрами:"
	@echo "  make record DURATION=30 FPS=20"
	@echo "  make record OUTPUT=recordings/road.mp4"
	@echo "  make mask-video INPUT=recordings/road.mp4 MASK_OUTPUT=outputs/road_mask.mp4"
	@echo "  make mask-camera CAMERA=0 STREAM_URL=udp://127.0.0.1:5000?pkt_size=1316"

record:
	@mkdir -p "$$(dirname "$(OUTPUT)")"
	$(PYTHON) "$(SCRIPT)" \
		--output "$(OUTPUT)" \
		--duration "$(DURATION)" \
		--fps "$(FPS)" \
		--camera "$(CAMERA)" \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)"

mask-video:
	@mkdir -p "$$(dirname "$(MASK_OUTPUT)")"
	$(PYTHON) "$(MASK_SCRIPT)" \
		--mode video \
		--input "$(INPUT)" \
		--output "$(MASK_OUTPUT)" \
		--fps "$(FPS)" \
		--threshold "$(THRESHOLD)" \
		--texts "$(TEXTS)"

mask-camera:
	$(PYTHON) "$(MASK_SCRIPT)" \
		--mode camera \
		--camera "$(CAMERA)" \
		--fps "$(FPS)" \
		--width "$(WIDTH)" \
		--height "$(HEIGHT)" \
		--threshold "$(THRESHOLD)" \
		--stream-url "$(STREAM_URL)" \
		--stream-format "$(STREAM_FORMAT)" \
		--texts "$(TEXTS)"
