# Layer-wise Transferability of ImageNet, DINOv2, and DINOv3 to Chest X-ray Pneumonia Classification

This repository contains the code used for the ICASSP 2026 project report:

**Layer-wise Transferability of ImageNet, DINOv2, and DINOv3 to Chest X-ray Pneumonia Classification**

The project evaluates how frozen natural-image-pretrained Vision Transformer (ViT) representations transfer to a pediatric chest X-ray pneumonia classification task. The focus is not only final performance, but also how transferability changes across depth (L3, L6, L9, L12) and how supervised and self-supervised representations differ.

## Main experimental setting

- **Task:** Binary classification, Normal vs. Pneumonia
- **Dataset:** Chest X-Ray Images (Pneumonia), Kermany et al., Cell 2018; Kaggle, CC BY 4.0
- **Input size:** 518 x 518
- **Backbones:**
  - ImageNet ViT-B/16, supervised
  - DINOv2 ViT-B/14, self-supervised
  - DINOv3 ViT-B/16, self-supervised
- **Protocol:** Frozen backbone + independent linear probe for each layer
- **Layers:** L3, L6, L9, L12
- **Evaluation:** AUROC, F1, sensitivity, specificity, accuracy, and AUROC bootstrap confidence intervals

## Repository structure

```text
xray-transfer-layerwise/
├── src/
│   ├── config.py            # Experiment paths and hyperparameters
│   ├── data.py              # Dataset loading and preprocessing
│   ├── models.py            # ViT/DINO backbone wrappers and layer extraction
│   ├── sanity_check.py      # Small end-to-end pipeline check
│   ├── extract_features.py  # Frozen feature extraction and caching
│   ├── probe.py             # Linear probing experiments
│   └── analyze.py           # Figures, CKA, and t-SNE analysis
└── requirements.txt
```

## Data preparation

Download the Chest X-Ray Images (Pneumonia) dataset and arrange it as follows:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

By default, the code looks for the dataset at `data/chest_xray` inside the repository. You can also set an environment variable:

```bash
export XRAY_DATA_ROOT=/path/to/chest_xray
```

On Windows PowerShell:

```powershell
$env:XRAY_DATA_ROOT="C:\path\to\chest_xray"
```

## Installation

```bash
pip install -r requirements.txt
```

Install the PyTorch version that matches your CUDA environment from the official PyTorch installation page.

DINOv3 may require HuggingFace access approval and login:

```bash
huggingface-cli login
```

If the stable `transformers` package does not yet support the selected DINOv3 model, install the development version:

```bash
pip install git+https://github.com/huggingface/transformers
```

## How to reproduce the experiments

Run the scripts from the repository root.

### 1. Sanity check

```bash
cd src
python data.py
python models.py
python sanity_check.py --model dinov2
```

### 2. Extract frozen features

```bash
python extract_features.py
```

This creates cached feature files under `cache_518/`. These files are not tracked by Git.

### 3. Train linear probes

```bash
python probe.py
```

This creates `results_518_ci/metrics.csv`.

### 4. Generate analysis figures

```bash
python analyze.py
```

This creates the layer-wise AUROC curve, optimal-layer table, CKA heatmap, and t-SNE figure under `results_518_ci/figures/`.

### 5. Run the full pipeline

From the repository root:

```bash
bash run_all.sh
```

## Reported PPT results

The small files in `results_from_ppt/` contain the numerical values used in the report:

- `l12_auroc_ci.csv`: L12 AUROC with 95% confidence intervals
- `layerwise_auroc_518.csv`: L3/L6/L9/L12 AUROC values at 518 px
- `cka_values.csv`: Pairwise CKA similarities

These files are included only to make the report values transparent. The main experiments should be reproduced by running the pipeline above.

## Code availability statement

Use the following statement in the final report after replacing the placeholder URL with the actual GitHub repository URL:

> Code Availability: The source code used in this project is publicly available at GitHub: https://github.com/your-repository-link. The repository includes the implementation, preprocessing scripts, linear-probing experiments, analysis scripts, and instructions required to reproduce the main results.

## Notes and caveats

- The comparison uses CLS-token representations only.
- DINOv2 uses a ViT-B/14 backbone, while ImageNet and DINOv3 use ViT-B/16. This patch-size difference should be mentioned as a caveat when interpreting the results.
- The project intentionally excludes fine-tuning to isolate representation quality.
- The dataset is not redistributed in this repository. Users should download it from the original source and follow its license terms.
