#!/usr/bin/env bash
# Интерактивная съемка калибровочных фото: preview + сохранение по кнопке.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output_dir> [max_shots]"
  echo "  output_dir — папка для calib_001.jpg, calib_002.jpg, ..."
  echo "  max_shots  — лимит снимков (по умолчанию 0 = без лимита)"
  exit 1
fi

OUT_DIR="$1"
MAX_SHOTS="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE="${SCRIPT_DIR}/capture_calib_photos_interactive.py"
CALIB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${CAPTURE}" ]]; then
  echo "Not found: ${CAPTURE}"
  exit 1
fi

mkdir -p "${OUT_DIR}"
echo "Интерактивная съемка в ${OUT_DIR}"
echo "Управление: SPACE/s — сохранить кадр, q/ESC — выход"

python3 "${CAPTURE}" "${OUT_DIR}" --prefix calib --ext jpg --max-shots "${MAX_SHOTS}" --width 1280 --height 720

echo "Готово. Калибровка:"
echo "  python3 ${CALIB_DIR}/calibrate/camera_calibration.py \\"
echo "    --images '${OUT_DIR}/calib_*.jpg' --show \\"
echo "    --out ${CALIB_DIR}/config/camera_calib.yml"
