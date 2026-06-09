
import argparse
import torch
from tqdm import tqdm

from config import CACHE_DIR, MODELS, LAYERS
from data import get_loader
from models import load_backbone


def extract_split(wrapper, split):
    loader = get_loader(split, shuffle=False)
    labels_all = []
    feats_all = {label: [] for label in LAYERS}

    for x, y in tqdm(loader, desc=f"  {split}", leave=False):
        x = x.to(next(wrapper.parameters()).device, non_blocking=True)
        feats = wrapper.extract(x)          # {label: [B,D] on cpu}
        for label in LAYERS:
            feats_all[label].append(feats[label])
        labels_all.append(y)

    feats_cat = {label: torch.cat(v, dim=0) for label, v in feats_all.items()}
    labels_cat = torch.cat(labels_all, dim=0)
    return feats_cat, labels_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help="추출할 모델 key (기본: config의 전체)")
    ap.add_argument("--splits", nargs="*", default=["train", "test"],
                    help="추출할 split (기본: train test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    for model_key in args.models:
        print(f"\n=== 모델 로드: {model_key} ({MODELS[model_key]['pretty']}) ===")
        wrapper = load_backbone(model_key, args.device)

        for split in args.splits:
            out_path = CACHE_DIR / f"{model_key}_{split}.pt"
            if out_path.exists():
                print(f"  이미 존재, 건너뜀: {out_path.name}")
                continue
            feats, labels = extract_split(wrapper, split)
            torch.save({"features": feats, "labels": labels}, out_path)
            shapes = {k: tuple(v.shape) for k, v in feats.items()}
            print(f"  저장: {out_path.name}  labels={tuple(labels.shape)}  {shapes}")

        # 메모리 정리
        del wrapper
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print("\n완료.")


if __name__ == "__main__":
    main()
