# Работа с ROS-нодой детекции

Практическая инструкция: как запустить, настроить, проверить и отладить ноду
`ros/defect_detection_node.py`. Про встраивание в свой код — см.
[INTEGRATION_ROS.md](INTEGRATION_ROS.md), про устройство и порядок обработки
с диаграммами — [ARCHITECTURE.md](ARCHITECTURE.md).

Нода делает одно: берёт кадр из входного топика, прогоняет через детекторы,
публикует обработанное изображение в выходной топик.

---

## 1. Запуск

Две обязательные вещи: подгрузить окружение ROS и запускать **интерпретатором
из venv** (в нём лежат torch с CUDA и модели).

```bash
source /opt/ros2_humble/install/setup.bash
cd ~/workspace/detection
~/workspace/.venvs/detection/bin/python ros/defect_detection_node.py \
    --config ros/config/default.json
```

Так работает потому, что `rclpy` собран для Python 3.8 и venv тоже на Python 3.8:
после `source setup.bash` интерпретатор venv видит и ROS, и наши зависимости.

Признак успешного старта — три строки в логе:

```
[INFO] [defect_detection]: Загрузка моделей (kind=masks, pipe=pipe3, …)…
[INFO] [defect_detection]: Модели загружены за 7.3 с (повреждения=seg, инфраструктура=off)
[INFO] [defect_detection]: Вход: /camera/color/image_raw -> выход: /defect_detection/image
```

**Первые ~7 секунд нода не отвечает** — грузятся 5 YOLO-моделей и SAM. Это нормально.

Остановка — `Ctrl+C`, либо `pkill -f defect_detection_node.py`.

---

## 2. Настройка через конфиг

Всё задаётся JSON-файлом, код править не нужно. Скопируйте
`ros/config/default.json`, поправьте под задачу и передайте через `--config`.

| Ключ | По умолчанию | Смысл |
|---|---|---|
| `input_topic` | `/camera/color/image_raw` | откуда брать кадры |
| `output_topic` | `/defect_detection/image` | куда публиковать результат |
| `kind` | `"masks"` | `boxes-damage`, `boxes-infra`, `boxes-all`, `masks` |
| `pipe` | `"pipe3"` | `pipe1`…`pipe6`, только при `kind="masks"` |
| `damage` / `infra` | `null` | явная форма вместо `pipe`: `"seg"`, `"box"`, `"off"` |
| `threshold` | `0.25` | порог уверенности YOLO (повреждения) |
| `infra_threshold` | `0.15` | порог уверенности OWL-ViT (инфраструктура) |
| `weights_dir` | `null` | папка весов YOLO; `null` = из `local_config.py` |
| `device` | `null` | `"cuda"` / `"cpu"`; `null` = автоопределение |
| `input_reliability` | `"best_effort"` | QoS входа: `best_effort` или `reliable` |
| `output_reliability` | `"reliable"` | QoS выхода |
| `log_every` | `10` | раз во сколько кадров писать статистику |

Опечатка в имени ключа — ошибка при старте со списком допустимых, а не молчаливое
игнорирование.

### Что выбрать в `kind` и `pipe`

| Задача | Настройки |
|---|---|
| посмотреть, что находит детектор | `kind: "boxes-damage"` |
| точные маски повреждений (самое частое) | `kind: "masks"`, `pipe: "pipe3"` |
| то же, но втрое быстрее и грубее | `kind: "masks"`, `pipe: "pipe5"` |
| повреждения + знаки и столбы | `kind: "masks"`, `pipe: "pipe1"` |
| только знаки и столбы | `kind: "masks"`, `pipe: "pipe4"` |

Полная таблица режимов — в [SPEC_process_frame.md](SPEC_process_frame.md).

**Важно про QoS:** `input_reliability` должен совпадать с настройкой источника
кадров. Драйверы камер обычно публикуют `best_effort` — это и стоит по умолчанию.
Если источник публикует `reliable`, а нода слушает `best_effort`, кадры дойдут;
наоборот — нет. При подозрении сверьте `ros2 topic info <топик> --verbose`.

---

## 3. Источник кадров для проверки

ROS-драйвера RealSense на этой машине **нет**, поэтому входной топик сам по себе
пустой. Для проверки есть публикатор — шлёт в топик кадры из папки или видеофайла:

```bash
source /opt/ros2_humble/install/setup.bash
cd ~/workspace/detection

# из папки с картинками, зациклено
~/workspace/.venvs/detection/bin/python ros/publish_frames.py \
    --source outputs/pubframes --rate 6 --loop

# из видеофайла, первые 50 кадров
~/workspace/.venvs/detection/bin/python ros/publish_frames.py \
    --source outputs/recordings/some.avi --rate 6 --limit 50
```

Параметры: `--topic` (куда), `--rate` (Гц), `--loop`, `--limit`,
`--reliability` (должен совпадать с `input_reliability` ноды), `--frame-id`.

**Учтите:** на кадрах 1920×1080 публикатор реально выдаёт около **0.6 Гц**, а не
запрошенные 6 — упирается в декодирование PNG и передачу 6-мегабайтных сообщений.
Для проверки ноды этого достаточно (она всё равно медленнее), но как измеритель
пропускной способности публикатор не годится.

---

## 4. Проверка, что всё работает

```bash
source /opt/ros2_humble/install/setup.bash

ros2 node list                        # должна быть /defect_detection
ros2 topic list                       # входной и выходной топики
ros2 topic info /camera/color/image_raw   # Subscription count должен стать 1
ros2 topic hz /defect_detection/image     # частота выдачи результата
```

Просмотр результата (`rqt_image_view` не установлен, но есть эти два способа):

```bash
# отдельное окно с картинкой
ros2 run image_tools showimage --ros-args -r image:=/defect_detection/image

# либо rviz2 -> Add -> By topic -> /defect_detection/image
rviz2
```

Понадобится графическая сессия. По SSH без проброса X окно не откроется.

### Как читать лог

```
кадров обработано 6, пропущено 1 | последний 2.02 с (~0.5 FPS) | повреждений 4, инфраструктуры 0
```

- **пропущено** — кадры, вытесненные более свежими, пока шла обработка. Это
  штатное поведение, а не сбой: обработка медленнее съёмки, и нода намеренно
  не копит очередь (см. раздел 6).
- **последний** — время обработки последнего кадра.
- **повреждений / инфраструктуры** — сколько объектов нашлось. Ноль — валидный
  результат: на выход уйдёт чистый оригинал (для рамок) или чёрный кадр (для масок).

Первый обработанный кадр логируется всегда, дальше — раз в `log_every`.

---

## 5. Если что-то не так

| Симптом | Причина и что делать |
|---|---|
| `ModuleNotFoundError: No module named 'rclpy'` | Не выполнен `source /opt/ros2_humble/install/setup.bash` |
| `ModuleNotFoundError: No module named 'torch'` | Запущено системным `python3`. Нужен `~/workspace/.venvs/detection/bin/python` |
| `Веса модели … не найдены` | Нет `weights_yolo/*.pt` или неверный `YOLO_WEIGHTS_DIR` в `local_config.py` |
| `Неизвестные ключи в конфиге` | Опечатка в JSON, в сообщении перечислены допустимые ключи |
| Нода стартовала, но лога обработки нет | Кадры не доходят. Проверьте `ros2 topic info <вход>`: `Subscription count` должен быть 1, `Publisher count` ≥ 1. Частая причина — разные топики у источника и в конфиге либо несовпадение QoS |
| Не сразу видно топики в `ros2 topic list` | Обнаружение DDS занимает несколько секунд после старта. Подождите и повторите |
| `Поддерживаются только bgr8 и rgb8` | Источник публикует другую кодировку (например `mono8` или `16UC1` — это карта глубины, а не цветной кадр). Подайте цветной топик |
| Нода «висит» первые секунды | Грузятся модели, ~7 с. Норма |
| Первый кадр обработался вдвое дольше | Прогрев CUDA. Дальше выходит на обычное время |

---

## 6. Чего ожидать по скорости

Замеры на Jetson AGX Orin, кадры 1920×1080:

| Режим (`pipe`) | Время на кадр | FPS |
|---|---|---|
| `pipe5` — повреждения боксами | 0.42 с | 2.4 |
| `pipe3` — повреждения масками SAM | 1.09 с | 0.9 |
| `pipe1` — полный цикл | 1.27 с | 0.8 |

Под нагрузкой в ROS время выше — при реальном прогоне через топики выходило
1.3-2.2 с на кадр: добавляются приём и распаковка 6-мегабайтных сообщений.

**Обрабатывать каждый кадр в темпе съёмки невозможно.** Съёмка идёт на 6 FPS,
это 166 мс на кадр — даже самый быстрый режим в 2.5 раза медленнее, а полный
цикл в 8 раз. Поэтому нода обрабатывает не всё подряд: приёмный колбэк только
запоминает последний кадр, обработка идёт в отдельном потоке, а кадры, пришедшие
во время работы, вытесняются. Очередь не растёт, задержка не накапливается,
на выход всегда идёт свежий кадр. В логе это видно как «пропущено N».

Если нужен каждый кадр — писать видео и обрабатывать офлайн
(`camera_tools/build_defect_quad_video.py`), а не через ноду.

Подробности замеров и что можно ускорить — в
[INTEGRATION_ROS.md](INTEGRATION_ROS.md), раздел 7.

---

## 7. Пример: свой конфиг

```bash
cp ros/config/default.json ros/config/my.json
```

```json
{
  "input_topic": "/my_camera/image",
  "output_topic": "/road_defects/mask",
  "kind": "masks",
  "pipe": "pipe1",
  "threshold": 0.3,
  "infra_threshold": 0.12,
  "log_every": 5
}
```

```bash
~/workspace/.venvs/detection/bin/python ros/defect_detection_node.py --config ros/config/my.json
```

Указывать все ключи не нужно — чего нет в файле, берётся из значений по умолчанию.
