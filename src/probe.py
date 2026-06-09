"""
probe.py — [Step 4] Linear probing.

캐시된 feature를 읽어 (model × layer × seed) 마다 독립적인
linear classifier를 학습하고 평가 지표를 계산한다.

핵심 설계:
  - 각 layer feature로 "독립적인" classifier를 따로 학습한다.
    (그래야 layer 간 공정한 비교가 됨)
  - classifier는 PyTorch Linear(D→2) + AdamW + class-weighted CE.
    (proposal과 일관. sklearn LogisticRegression도 옵션으로 둠)
  - feature는 학습 전 표준화(StandardScaler, train 통계로 fit).

출력: results/metrics.csv  (model, layer, seed, AUROC, F1, Sens, Spec, Acc)

실행:
    python probe.py
    python probe.py --classifier sklearn   # sklearn 백엔드 사용
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    confusion_matrix, accuracy_score,
)

from config import (
    CACHE_DIR, RESULTS_DIR, MODELS, LAYERS, SEEDS, NUM_CLASSES,
    PROBE_LR, PROBE_EPOCHS, PROBE_WD, PROBE_BATCH,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(model_key, split):
    path = CACHE_DIR / f"{model_key}_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 extract_features.py를 실행하세요."
        )
    blob = torch.load(path, map_location="cpu")
    return blob["features"], blob["labels"].numpy()


def bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=1000, seed=0):
    """Test set에 대한 95% bootstrap CI for AUROC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def compute_metrics(y_true, y_prob):
    """y_prob: positive(Pneumonia=1) 클래스 확률."""
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0   # recall of positive
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy_score(y_true, y_pred),
    }


# ── PyTorch linear head ─────────────────────────────────────
def train_torch_probe(Xtr, ytr, Xte, seed, device):
    set_seed(seed)
    in_dim = Xtr.shape[1]
    clf = nn.Linear(in_dim, NUM_CLASSES).to(device)

    # class-weighted CE (1:2.7 불균형 보정)
    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    opt = torch.optim.AdamW(clf.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROBE_EPOCHS)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    n = len(Xtr_t)

    clf.train()
    for _ in range(PROBE_EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, PROBE_BATCH):
            idx = perm[i:i + PROBE_BATCH]
            opt.zero_grad()
            loss = crit(clf(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()

    clf.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        prob = torch.softmax(clf(Xte_t), dim=1)[:, 1].cpu().numpy()
    return prob


# ── sklearn 백엔드 (대안) ────────────────────────────────────
def train_sklearn_probe(Xtr, ytr, Xte, seed):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=seed,
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["torch", "sklearn"], default="torch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS_DIR / "metrics.csv"))
    args = ap.parse_args()

    rows = []
    for model_key in MODELS:
        feats_tr, ytr = load_cache(model_key, "train")
        feats_te, yte = load_cache(model_key, "test")

        for label in sorted(LAYERS, key=lambda k: LAYERS[k]):
            Xtr = feats_tr[label].numpy()
            Xte = feats_te[label].numpy()

            # train 통계로 표준화
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xte_s = scaler.transform(Xte)

            for seed in SEEDS:
                if args.classifier == "torch":
                    prob = train_torch_probe(Xtr_s, ytr, Xte_s, seed, args.device)
                else:
                    prob = train_sklearn_probe(Xtr_s, ytr, Xte_s, seed)

                m = compute_metrics(yte, prob)
                ci_lo, ci_hi = bootstrap_auroc_ci(yte, prob, seed=seed)
                m.update({
                    "model": model_key, "layer": label, "seed": seed,
                    "CI_low": round(ci_lo, 4), "CI_high": round(ci_hi, 4),
                })
                rows.append(m)
                print(f"{model_key:10s} {label:4s} seed{seed}  "
                      f"AUROC={m['AUROC']:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
                      f"F1={m['F1']:.4f}  Sens={m['Sensitivity']:.3f}  Spec={m['Specificity']:.3f}")

    df = pd.DataFrame(rows)[
        ["model", "layer", "seed", "AUROC", "CI_low", "CI_high", "F1",
         "Sensitivity", "Specificity", "Accuracy"]
    ]
    df.to_csv(args.out, index=False)
    print(f"\n저장: {args.out}  ({len(df)} 행)")

    # 요약: model×layer 평균 AUROC + 평균 95% bootstrap CI
    print("\n=== 요약 (AUROC mean, 95% bootstrap CI averaged over seeds) ===")
    g = df.groupby(["model", "layer"])
    summary = g["AUROC"].mean().reset_index()
    summary["CI_low"]  = g["CI_low"].mean().values
    summary["CI_high"] = g["CI_high"].mean().values
    summary["AUROC"] = summary["AUROC"].round(4)
    summary["CI_low"]  = summary["CI_low"].round(4)
    summary["CI_high"] = summary["CI_high"].round(4)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
