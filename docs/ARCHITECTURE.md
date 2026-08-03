# Архитектура: ROS-нода и обработка кадров нейросетями

Как устроен путь кадра от входного топика до выходного и что происходит внутри.
Про запуск и настройку — [USAGE_ROS_NODE.md](USAGE_ROS_NODE.md), про встраивание
в свой код — [INTEGRATION_ROS.md](INTEGRATION_ROS.md).

---

## 1. Граф ROS

```mermaid
flowchart LR
    SRC["Источник кадров<br/>камера или publish_frames.py"]
    NODE["/defect_detection<br/>нода"]
    SINK["Потребитель<br/>showimage, rviz2, ваш узел"]

    SRC -->|"sensor_msgs/Image<br/>/camera/color/image_raw<br/>QoS: best_effort"| NODE
    NODE -->|"sensor_msgs/Image<br/>/defect_detection/image<br/>QoS: reliable"| SINK
```

Топики и QoS задаются в JSON-конфиге. Вход по умолчанию `best_effort` — так
публикуют драйверы камер. Выход `reliable`, чтобы результат не терялся.

---

## 2. Два потока внутри ноды

Обработка кадра занимает 0.4–2.2 с, а кадры приходят каждые 166 мс. Если бы
колбэк обрабатывал кадр сам, он заблокировал бы исполнителя ROS на секунды.
Поэтому колбэк только запоминает последний кадр, а работа идёт в отдельном потоке.

```mermaid
sequenceDiagram
    participant CAM as Источник
    participant CB as Колбэк ROS
    participant SLOT as Ячейка _pending
    participant W as Рабочий поток
    participant PUB as Выходной топик

    CAM->>CB: кадр 1
    CB->>SLOT: запомнить
    SLOT->>W: событие _wake
    W->>SLOT: забрать и очистить
    activate W
    CAM->>CB: кадр 2
    CB->>SLOT: запомнить
    CAM->>CB: кадр 3
    CB->>SLOT: кадр 2 вытеснен<br/>«пропущено» +1
    W->>PUB: результат кадра 1
    deactivate W
    W->>SLOT: забрать кадр 3
    activate W
    W->>PUB: результат кадра 3
    deactivate W
```

Ячейка рассчитана ровно на один кадр. Очередь не растёт, задержка не
накапливается, в обработку всегда уходит самый свежий кадр.

---

## 3. Конвейер обработки кадра

```mermaid
flowchart TD
    MSG["sensor_msgs/Image"] --> CONV["imgmsg_to_bgr()<br/>учёт msg.step, bgr8 или rgb8"]
    CONV --> FRAME["numpy BGR (H, W, 3)"]

    FRAME --> DCHK{"damage ≠ off?"}
    FRAME --> ICHK{"infra ≠ off?"}

    subgraph D ["Ветка damage — повреждения покрытия"]
        DCHK -->|да| YOLO["RoadDefectYoloPipeline.detect()<br/>5 YOLO-моделей последовательно<br/>~0.42 с"]
        YOLO --> SUP["suppress_duplicates (IoU > 0.70)<br/>suppress_nested_boxes"]
        SUP --> DMODE{"режим damage"}
        DMODE -->|seg| DSAM["SAM.set_image ~0.64 с<br/>predict по каждому боксу<br/>точная маска"]
        DMODE -->|box| DBOX["маска = бокс,<br/>закрашенный целиком"]
    end

    subgraph I ["Ветка infra — знаки, столбы, светофоры, люки"]
        ICHK -->|да| OWL["OWL-ViT по INFRA_TEXTS<br/>17 текстовых описаний<br/>~0.15 с"]
        OWL --> IMODE{"режим infra"}
        IMODE -->|seg| ISAM["SAM.set_image ~0.64 с<br/>ВТОРОЙ раз: свой экземпляр SAM<br/>segment_boxes()"]
        IMODE -->|box| IBOX["маска = бокс,<br/>закрашенный целиком"]
    end

    DSAM --> ITEMS
    DBOX --> ITEMS
    ISAM --> ITEMS
    IBOX --> ITEMS

    ITEMS["список ProcessedItem<br/>label, score, box_xyxy,<br/>color_bgr, source, mask"]
    ITEMS --> KIND{"kind"}
    KIND -->|masks| RM["render_masks()<br/>чёрный фон,<br/>маски цветом класса"]
    KIND -->|"boxes-*"| RB["render_boxes()<br/>оригинал,<br/>рамки и подписи"]

    RM --> OUT["bgr_to_imgmsg()<br/>header входного сообщения<br/>переносится в выходное"]
    RB --> OUT
    OUT --> PUB["публикация в выходной топик"]
```

Ветки независимы: обе, одна или другая — в зависимости от режима.

---

## 4. Выбор режима

Выбор двумерный. Флаги `pipe1…pipe6` — именованные комбинации двух параметров.

```mermaid
flowchart LR
    K{"kind"}
    K -->|"boxes-damage"| B1["рамки повреждений<br/>поверх оригинала"]
    K -->|"boxes-infra"| B2["рамки инфраструктуры<br/>поверх оригинала"]
    K -->|"boxes-all"| B3["рамки обоих источников"]
    K -->|"masks"| P{"pipe"}

    P -->|pipe1| M1["damage: seg<br/>infra: seg"]
    P -->|pipe2| M2["damage: box<br/>infra: seg"]
    P -->|pipe3| M3["damage: seg<br/>infra: off"]
    P -->|pipe4| M4["damage: off<br/>infra: seg"]
    P -->|pipe5| M5["damage: box<br/>infra: off"]
    P -->|pipe6| M6["damage: off<br/>infra: box"]
```

|  | infra: seg | infra: box | infra: off |
|---|---|---|---|
| **damage: seg** | `pipe1` | — | `pipe3` |
| **damage: box** | `pipe2` | — | `pipe5` |
| **damage: off** | `pipe4` | `pipe6` | запрещено |

Модели загружаются **лениво под выбранный режим**: при `damage: off` пять
YOLO-моделей не поднимаются вообще, при `infra: off` не грузится OWL-ViT.
Экономятся и память, и время старта.

---

## 5. Жизненный цикл ноды

```mermaid
sequenceDiagram
    participant U as Оператор
    participant N as Нода
    participant M as Модели
    participant T as Топики

    U->>N: запуск с --config
    N->>N: читает JSON, проверяет ключи
    Note over N: неизвестный ключ —<br/>ошибка со списком допустимых
    N->>M: FrameProcessor(...)
    activate M
    Note over M: загрузка 5 YOLO + SAM<br/>и/или OWL-ViT — ~7 с<br/>нода не отвечает
    M-->>N: готово
    deactivate M
    N->>T: publisher и subscription
    N->>N: старт рабочего потока
    loop на каждый принятый кадр
        T->>N: кадр
        N->>N: обработка (первый вдвое дольше — прогрев CUDA)
        N->>T: результат
    end
    U->>N: Ctrl+C
    N->>N: остановка потока, destroy_node
```

---

## 6. Где уходит время

```mermaid
flowchart LR
    A["приём и распаковка<br/>сообщения 6 МБ<br/>0.1–0.9 с"] --> B["5 YOLO-моделей<br/>0.42 с"]
    B --> C["SAM: кодирование кадра<br/>0.64 с"]
    C --> D["OWL-ViT<br/>0.15 с"]
    D --> E["SAM: кодирование<br/>повторно<br/>0.64 с"]
    E --> F["отрисовка<br/>и публикация<br/>мало"]
```

| Этап | Время | Когда выполняется |
|---|---|---|
| 5 моделей YOLO | 0.42 с | `damage` ≠ off |
| SAM: кодирование кадра | 0.64 с | `damage` = seg |
| SAM: маска на каждый бокс | мало | за каждый найденный объект |
| OWL-ViT | 0.15 с | `infra` ≠ off |
| SAM: кодирование кадра повторно | 0.64 с | `infra` = seg |
| приём и распаковка сообщения | 0.1–0.9 с | всегда, зависит от нагрузки |

**Известная потеря.** У ветки damage и ветки infra свои экземпляры SAM, и каждый
независимо прогоняет один и тот же кадр через энкодер. В режимах `pipe1` и `pipe2`
это лишние 0.64 с. Общий экземпляр убрал бы их. Пока не сделано.

---

## 7. Итоговые числа

Замеры на Jetson AGX Orin, кадры 1920×1080:

| Режим | Время на кадр | FPS |
|---|---|---|
| `pipe5` — повреждения боксами | 0.42 с | 2.4 |
| `pipe3` — повреждения масками SAM | 1.09 с | 0.9 |
| `pipe1` — полный цикл | 1.27 с | 0.8 |

В реальном прогоне через ROS выходило 1.3–2.2 с на кадр: добавляются приём и
распаковка сообщений.

Съёмка идёт на 6 FPS — это 166 мс на кадр. Даже самый быстрый режим в 2.5 раза
медленнее, полный цикл в 8 раз. **Обработка каждого кадра в темпе съёмки
невозможна** — отсюда вытеснение кадров из раздела 2.
