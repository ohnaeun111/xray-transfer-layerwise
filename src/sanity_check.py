import argparse
import numpy as np
import torch

from config import MODELS, LAYERS, IMG_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=list(MODELS.keys())[0])
    ap.add_argument("--n", type=int, default=300, help="train subset 크기")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"=== Sanity Check: {args.model} on {args.device} ===\n")

    # ── 1. 데이터 ──
    print("[1] 데이터 로딩...")
    from data import get_dataset, describe
    describe()
    train_ds = get_dataset("train")
    test_ds = get_dataset("test")

    # ── 2. 모델 + 더미 추출 ──
    print("\n[2] 모델 로드 + 중간 layer shape 확인...")
    from models import load_backbone
    wrapper = load_backbone(args.model, args.device)
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=args.device)
    feats = wrapper.extract(dummy)
    for label in sorted(feats, key=lambda k: LAYERS[k]):
        print(f"    {label}: {tuple(feats[label].shape)}")

    # ── 3. 작은 subset feature 추출 (L12만) ──
    print(f"\n[3] subset feature 추출 (train {args.n}개, test 전체)...")
    from torch.utils.data import DataLoader, Subset
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train_ds), size=min(args.n, len(train_ds)), replace=False)
    tr_loader = DataLoader(Subset(train_ds, idx), batch_size=64)
    te_loader = DataLoader(test_ds, batch_size=64)

    def grab(loader):
        Xs, ys = [], []
        for x, y in loader:
            f = wrapper.extract(x.to(args.device))["L12"]
            Xs.append(f.numpy()); ys.append(y.numpy())
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr, ytr = grab(tr_loader)
    Xte, yte = grab(te_loader)
    print(f"    Xtr={Xtr.shape}  Xte={Xte.shape}")

    # ── 4. linear probing ──
    print("\n[4] linear probing (sklearn LogisticRegression)...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    prob = clf.predict_proba(sc.transform(Xte))[:, 1]
    auroc = roc_auc_score(yte, prob)

    # ── 5. 판정 ──
    print(f"\n[5] Test AUROC = {auroc:.4f}")
    if auroc >= 0.85:
        print("    ✅ 통과 — 파이프라인 정상. 본 실험으로 진행 가능.")
    else:
        print("    ⚠️  낮음 — 아래를 점검하세요:")
        print("       - Grayscale→3채널 변환이 적용됐는지 (data.py)")
        print("       - ImageNet mean/std normalize 적용 여부")
        print("       - 클래스 매핑(NORMAL=0/PNEUMONIA=1)이 맞는지")
        print("       - 모델이 frozen·eval 상태인지")
        print("       - subset(--n)이 너무 작지 않은지 (>=300 권장)")


if __name__ == "__main__":
    main()
