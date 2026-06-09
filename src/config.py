"""
config.py — 모든 실험 설정을 한 곳에서 관리.

이 파일만 고치면 실험 전체가 바뀝니다. 특히:
- DATA_ROOT: Kaggle chest_xray 폴더 경로 (★ 본인 환경에 맞게 수정)
- MODELS: 비교할 backbone들. DINOv3 승인 전에는 dinov3를 주석 처리하고,
  승인 후 주석만 풀면 됩니다.
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 경로 설정  (★ DATA_ROOT를 본인 환경에 맞게 수정하세요)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = Path(os.environ.get("XRAY_DATA_ROOT", PROJECT_ROOT / "data" / "chest_xray"))  # train/val/test folders
CACHE_DIR    = Path(os.environ.get("XRAY_CACHE_DIR", PROJECT_ROOT / "cache_518"))
RESULTS_DIR  = Path(os.environ.get("XRAY_RESULTS_DIR", PROJECT_ROOT / "results_518_ci"))
FIG_DIR      = RESULTS_DIR / "figures"

for d in (CACHE_DIR, RESULTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 데이터 설정
# ─────────────────────────────────────────────────────────────
IMG_SIZE   = 518
# ImageFolder는 클래스를 알파벳순으로 매핑: NORMAL=0, PNEUMONIA=1
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 2
BATCH_SIZE  = int(os.environ.get("XRAY_BATCH_SIZE", 16))        # 518x518 may require a small batch
NUM_WORKERS = int(os.environ.get("XRAY_NUM_WORKERS", 4))

# ImageNet 정규화 통계 (3개 backbone 모두 자연이미지 기반이므로 공통 사용)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ─────────────────────────────────────────────────────────────
# 추출할 layer (ViT-Base는 block 0~11, 총 12개)
#   사람이 읽는 라벨(L3..)과 실제 0-base 인덱스를 매핑
# ─────────────────────────────────────────────────────────────
LAYERS = {          # label : 0-base block index
    "L3":  2,
    "L6":  5,
    "L9":  8,
    "L12": 11,
}

# ─────────────────────────────────────────────────────────────
# 비교할 모델들
#   key   : 캐시 파일명·결과에 쓰일 짧은 식별자
#   loader: models.py 가 인식하는 타입
#   id    : timm 또는 huggingface 모델 문자열
# ─────────────────────────────────────────────────────────────
MODELS = {
    "imagenet": {
        "loader": "timm",
        "id": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "pretty": "ImageNet ViT-B/16 (Supervised)",
    },
    "dinov2": {
        "loader": "timm",
        "id": "vit_base_patch14_dinov2.lvd142m",
        "pretty": "DINOv2 ViT-B/14 (SSL)",
    },
    # ── DINOv3: 라이선스 승인 후 아래 주석을 풀어주세요 ──
    "dinov3": {
        "loader": "hf",
        "id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "pretty": "DINOv3 ViT-B/16 (SSL)",
    },
    
}

# ─────────────────────────────────────────────────────────────
# 학습/실험 설정
# ─────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2]            # 3 seeds
LABEL_FRACTION = 1.0         # 100% label 고정 (label efficiency는 future work)

# Linear probing (PyTorch head) 설정
PROBE_LR      = 1e-3
PROBE_EPOCHS  = 50
PROBE_WD      = 1e-4
PROBE_BATCH   = 256          # feature는 가벼우므로 큰 배치 OK
