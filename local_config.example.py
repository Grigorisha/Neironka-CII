"""Скопируйте в local_config.py и укажите свои пути."""

from pathlib import Path

# Папка с кадрами (jpg/png)
images_path = str(Path(__file__).resolve().parent / "data" / "images") + "/"

# Веса SAM: https://github.com/facebookresearch/segment-anything#model-checkpoints
# vit_b (~375 MB): sam_vit_b_01ec64.pth
SAM_CHECKPOINT = str(Path(__file__).resolve().parent / "weights" / "sam_vit_b_01ec64.pth")

MODEL_TYPE = "vit_b"  # vit_b | vit_l | vit_h
