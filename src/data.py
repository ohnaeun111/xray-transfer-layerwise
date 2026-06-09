"""
data.py — 데이터 로딩과 전처리.

핵심:
- grayscale X-ray를 3-channel로 복제 (자연이미지 backbone이 3ch을 기대)
- Resize to IMG_SIZE + ImageNet normalize
- ImageFolder가 NORMAL=0, PNEUMONIA=1 로 자동 매핑
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from config import (
    DATA_ROOT, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
)


def build_transform():
    """X-ray 전용 transform. Grayscale → 3채널 복제가 핵심."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # 흑백 X-ray를 3채널로
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataset(split: str):
    """split in {'train','val','test'}."""
    tf = build_transform()
    ds = datasets.ImageFolder(str(DATA_ROOT / split), transform=tf)
    # ImageFolder의 클래스 순서가 우리 가정과 일치하는지 확인
    assert ds.classes == CLASS_NAMES, (
        f"클래스 순서 불일치: {ds.classes} (기대값 {CLASS_NAMES}). "
        f"폴더명을 확인하세요."
    )
    return ds


def get_loader(split: str, shuffle: bool = False):
    ds = get_dataset(split)
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=True,
    )


def describe():
    """데이터 구성을 출력 — Day 1 sanity check용."""
    for split in ("train", "val", "test"):
        try:
            ds = get_dataset(split)
        except FileNotFoundError:
            print(f"[{split}] 폴더 없음 — 경로 확인 필요: {DATA_ROOT/split}")
            continue
        labels = [y for _, y in ds.samples]
        n_normal = labels.count(0)
        n_pneu = labels.count(1)
        print(f"[{split:5s}] total={len(ds):5d}  "
              f"NORMAL={n_normal:5d}  PNEUMONIA={n_pneu:5d}")


if __name__ == "__main__":
    # python data.py 로 실행하면 데이터 통계 + 배치 shape 확인
    print(f"DATA_ROOT = {DATA_ROOT}")
    describe()
    loader = get_loader("test")
    x, y = next(iter(loader))
    print(f"배치 shape: x={tuple(x.shape)}  y={tuple(y.shape)}")
    print(f"x 범위: [{x.min():.3f}, {x.max():.3f}]  (정규화 후이므로 음수 정상)")
