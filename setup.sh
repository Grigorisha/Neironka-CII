#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
CFG_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -U pip wheel
pip install -r "$CFG_DIR/requirements.txt"
python -m ipykernel install --user --name rosavtodor-sam-owlvit --display-name "SAM + OWL-ViT (РосАвтоДор)"

if [[ ! -f "$CFG_DIR/local_config.py" ]]; then
  cp "$CFG_DIR/local_config.example.py" "$CFG_DIR/local_config.py"
  echo "Создан $CFG_DIR/local_config.py — отредактируйте пути."
fi

mkdir -p "$CFG_DIR/data/images" "$CFG_DIR/weights"
echo ""
echo "Готово. В Cursor:"
echo "  1. Откройте папку: $ROOT"
echo "  2. Установите рекомендуемые расширения (Python, Jupyter)"
echo "  3. Откройте ноутбук и выберите ядро: SAM + OWL-ViT (РосАвтоДор)"
echo ""
echo "Скачайте веса SAM в: $CFG_DIR/weights/"
echo "  wget -O \"$CFG_DIR/weights/sam_vit_b_01ec64.pth\" \\"
echo "    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
