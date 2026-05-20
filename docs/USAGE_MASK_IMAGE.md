# Инструкция: обработка одного PNG в цветную instance-маску

Этот документ описывает, как запустить обработку **одного изображения** и получить на выходе PNG, где каждый сегментированный объект окрашен своим цветом.

## 1) Подготовка

Перейдите в проект:

```bash
cd /home/orin/workspace/detection
```

Используем готовое окружение:

```bash
source /home/orin/workspace/.venvs/detection/bin/activate
```

Проверьте, что есть ключевые файлы:

- `local_config.py`
- `weights/sam_vit_b_01ec64.pth`
- `camera_tools/run_mask_image.py`

## 2) Основная команда

```bash
python camera_tools/run_mask_image.py \
  --input-image /абсолютный/или/относительный/путь/input.png \
  --output-dir /путь/к/папке_для_результата
```

Что делает команда:

- читает входной PNG;
- запускает OWL-ViT + SAM;
- строит цветную instance-маску;
- сохраняет PNG в указанную директорию.

По умолчанию имя выходного файла: `instance_mask.png`.

## 3) Полезные параметры

Задать имя выходного файла:

```bash
python camera_tools/run_mask_image.py \
  --input-image outputs/test_input_frame2.png \
  --output-dir outputs/single_image_masks \
  --output-name mask_colored.png
```

Изменить порог детекции:

```bash
python camera_tools/run_mask_image.py \
  --input-image outputs/test_input_frame2.png \
  --output-dir outputs/single_image_masks \
  --threshold 0.05
```

Ограничить список классов (по умолчанию используется большой список из `mask_pipeline.py`):

```bash
python camera_tools/run_mask_image.py \
  --input-image outputs/test_input_frame2.png \
  --output-dir outputs/single_image_masks \
  --texts "car,truck,person,tree,pothole,road sign"
```

## 4) Полный рабочий пример

```bash
cd /home/orin/workspace/detection
source /home/orin/workspace/.venvs/detection/bin/activate

python camera_tools/run_mask_image.py \
  --input-image outputs/test_input_frame2.png \
  --output-dir outputs/single_image_masks \
  --output-name mask_colored2.png \
  --threshold 0.05
```

Ожидаемый результат:

- в `outputs/single_image_masks/` появится PNG;
- в консоли будет строка вида: `detections=N instances=M`;
- устройство обычно определяется как `Device: cuda` (если CUDA доступна).

## 5) Если нужно получить PNG из видео в recordings

Пример извлечения одного кадра из видео:

```bash
python -c "import cv2; cap=cv2.VideoCapture('recordings/webcam_20260515_191043.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,120); ok,frame=cap.read(); cap.release(); assert ok; cv2.imwrite('outputs/frame_from_video.png', frame)"
```

Далее обработка этого кадра:

```bash
python camera_tools/run_mask_image.py \
  --input-image outputs/frame_from_video.png \
  --output-dir outputs/single_image_masks
```

## 6) Типовые проблемы

- `FileNotFoundError` на входном изображении: проверьте путь в `--input-image`.
- Ошибка по checkpoint: убедитесь, что есть `weights/sam_vit_b_01ec64.pth`.
- Долгий первый запуск: `transformers` может докачивать модельные файлы.
- Предупреждение `torchvision/io image extension`: в текущем сценарии обычно не критично, обработка может работать нормально.
