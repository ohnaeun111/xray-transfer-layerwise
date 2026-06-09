"""
analyze.py — [Step 5] 분석 및 시각화.

probe.py가 만든 metrics.csv 와 캐시된 feature를 이용해
proposal의 핵심 figure들을 생성한다.

생성물 (results/figures/):
  fig1_layerwise_curve.png   — 메인: layer별 AUROC 곡선 (모델별 라인)
  fig3_cka_heatmap.png       — 모델 간 best-layer feature CKA 유사도
  fig4_tsne.png              — best vs worst layer feature t-SNE

(Figure 2 optimal-layer 표, Figure 5 attention map,
 failure case는 별도 함수/노트북에서 다룬다. 아래 stub 참고.)

실행:
    python analyze.py
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import (
    CACHE_DIR, RESULTS_DIR, FIG_DIR, MODELS, LAYERS,
)

LAYER_ORDER = sorted(LAYERS, key=lambda k: LAYERS[k])
LAYER_X = [LAYERS[k] + 1 for k in LAYER_ORDER]   # 1-base depth for x축


# ─────────────────────────────────────────────────────────────
# Figure 1: layer-wise AUROC curve  (메인 결과)
# ─────────────────────────────────────────────────────────────
def fig1_layerwise_curve(df):
    has_ci = "CI_low" in df.columns and "CI_high" in df.columns
    # model×layer 집계
    agg = df.groupby(["model", "layer"]).agg(
        AUROC=("AUROC", "mean"),
        **( {"CI_low": ("CI_low", "mean"), "CI_high": ("CI_high", "mean")} if has_ci
            else {"AUROC_std": ("AUROC", "std")} )
    ).reset_index()

    plt.figure(figsize=(7, 5))
    for model_key in MODELS:
        sub = (agg[agg.model == model_key]
               .set_index("layer").loc[LAYER_ORDER])
        mu = sub["AUROC"].values
        if has_ci:
            yerr = [mu - sub["CI_low"].values, sub["CI_high"].values - mu]
            ylabel = "AUROC (mean, 95% bootstrap CI)"
        else:
            yerr = sub["AUROC_std"].values
            ylabel = "AUROC (mean ± std over seeds)"
        plt.errorbar(LAYER_X, mu, yerr=yerr,
                     marker="o", capsize=4, label=MODELS[model_key]["pretty"])
    plt.xticks(LAYER_X, LAYER_ORDER)
    plt.xlabel("Layer depth")
    plt.ylabel(ylabel)
    plt.title("Layer-wise transferability on Chest X-ray Pneumonia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig1_layerwise_curve.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"저장: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 2: optimal layer 표 (텍스트로도 출력)
# ─────────────────────────────────────────────────────────────
def fig2_optimal_layer_table(df):
    g = df.groupby(["model", "layer"])["AUROC"].mean().reset_index()
    rows = []
    for model_key in MODELS:
        sub = g[g.model == model_key]
        best = sub.loc[sub["AUROC"].idxmax()]
        rows.append({
            "model": MODELS[model_key]["pretty"],
            "best_layer": best["layer"],
            "best_AUROC": round(float(best["AUROC"]), 4),
        })
    table = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "optimal_layer.csv"
    table.to_csv(out_csv, index=False)
    print(f"저장: {out_csv}")
    print(table.to_string(index=False))

    # PNG 테이블 저장
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.5 * len(table)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table.values,
        colLabels=["Model", "Best Layer", "Best AUROC"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#DCE6F1")
    plt.title("Optimal Layer per Model", pad=12, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_png = FIG_DIR / "fig2_optimal_layer_table.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"저장: {out_png}")
    return table


# ─────────────────────────────────────────────────────────────
# CKA (linear) — 두 feature 행렬의 표현 유사도
# ─────────────────────────────────────────────────────────────
def linear_cka(X, Y):
    """X:[N,d1], Y:[N,d2] -> scalar in [0,1]."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # HSIC 기반 linear CKA
    xtx = X.T @ X
    yty = Y.T @ Y
    xty = X.T @ Y
    hsic = np.sum(xty ** 2)
    denom = np.sqrt(np.sum(xtx ** 2) * np.sum(yty ** 2))
    return float(hsic / denom) if denom > 0 else 0.0


def fig3_cka_heatmap(split="test"):
    """모델 간 best-layer feature CKA. 여기선 각 모델의 L12로 단순화."""
    keys = list(MODELS.keys())
    mats = {}
    for k in keys:
        blob = torch.load(CACHE_DIR / f"{k}_{split}.pt", map_location="cpu")
        mats[k] = blob["features"]["L12"].numpy()

    n = len(keys)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = linear_cka(mats[keys[i]], mats[keys[j]])

    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(M, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, label="Linear CKA")
    plt.xticks(range(n), keys, rotation=45, ha="right")
    plt.yticks(range(n), keys)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                     color="white" if M[i, j] < 0.6 else "black", fontsize=9)
    plt.title("Cross-model representation similarity (L12, CKA)")
    plt.tight_layout()
    out = FIG_DIR / "fig3_cka_heatmap.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"저장: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 4: t-SNE of features (한 모델의 best/worst layer)
# ─────────────────────────────────────────────────────────────
def fig4_tsne(df=None, split="test"):
    from sklearn.manifold import TSNE

    # 모델별 best layer를 metrics.csv에서 결정
    best_layer = {}
    if df is not None:
        g = df.groupby(["model", "layer"])["AUROC"].mean().reset_index()
        for key in MODELS:
            sub = g[g.model == key]
            best_layer[key] = sub.loc[sub["AUROC"].idxmax(), "layer"]
    else:
        best_layer = {key: LAYER_ORDER[-1] for key in MODELS}

    keys = list(MODELS.keys())
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 4.5 * len(keys)))
    if len(keys) == 1:
        axes = [axes]

    for row, key in enumerate(keys):
        blob = torch.load(CACHE_DIR / f"{key}_{split}.pt", map_location="cpu")
        labels = blob["labels"].numpy()
        for col, layer in enumerate([LAYER_ORDER[0], best_layer[key]]):
            ax = axes[row][col]
            X = blob["features"][layer].numpy()
            emb = TSNE(n_components=2, init="pca", perplexity=30,
                       random_state=0).fit_transform(X)
            for cls, name in enumerate(["NORMAL", "PNEUMONIA"]):
                m = labels == cls
                ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.5, label=name)
            tag = "worst" if col == 0 else "best"
            ax.set_title(f"{MODELS[key]['pretty']} — {layer} ({tag})")
            ax.legend(markerscale=2)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("t-SNE of frozen CLS features (worst vs best layer per model)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig4_tsne.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"저장: {out}")


def main():
    df = pd.read_csv(RESULTS_DIR / "metrics.csv")
    fig1_layerwise_curve(df)
    fig2_optimal_layer_table(df)
    fig3_cka_heatmap()
    fig4_tsne(df)
    print("\n분석 완료. results/figures/ 확인.")


if __name__ == "__main__":
    main()
