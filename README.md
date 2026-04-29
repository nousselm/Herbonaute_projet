# 🌿 Herbarium Classification with CrossViT

Deep Learning project focused on **binary classification of herbarium images**, combining **Vision Transformers (CrossViT)**, **segmentation-guided learning**, and **interpretability analysis**.

## 📌 Overview

This project investigates how to effectively leverage **segmented and non-segmented botanical images** to improve both:

* Classification performance
* Model interpretability

The work explores multiple strategies to integrate segmentation into a Vision Transformer pipeline, moving from naive multi-input approaches to **explicit spatial guidance and loss constraints**.

---

## 🎯 Objectives

The project is structured around five main objectives:

* **O1 — Input configurations comparison**
  Compare:

  * Non-segmented images
  * Segmented images
  * Multi-branch CrossViT (both modalities)

* **O2 — Symmetric CrossViT architecture**
  Remove scale bias by enforcing:

  * Same resolution
  * Same patch size
  * Aligned augmentations

* **O3 — Segmentation-guided patch weighting**
  Use segmentation masks to **weight transformer patches**

* **O4 — Interpretability via attention analysis**
  Analyze attention maps and compare them to segmentation masks using IoU

* **O5 — IoU-based loss integration**
  Add a spatial alignment constraint directly into training

---

## 🧠 Methodology

### 🖼️ Data

* ~2000 herbarium images (segmented + non-segmented)
* Binary classification task (presence/absence of a morphological trait)
* Final dataset after filtering:

  * Train: ~790 images
  * Validation: ~190 images
* Data **not included** due to confidentiality constraints

---

### ⚙️ Preprocessing

* Resize to **224×224**
* Data augmentation:

  * Random crop
  * Flips
  * Affine transforms
  * Color jitter
* Normalization using ImageNet statistics

---

### 🏗️ Model

* **CrossViT (Vision Transformer multi-branch)**
* Custom wrapper to handle:

  * Single input
  * Dual modality input (seg / non-seg)

---

### 🔍 Key Innovation (O3)

Instead of feeding segmentation directly:

→ Convert segmentation into a **patch-level weighting signal**

For each patch:

[
r_p = \frac{\text{plant pixels}}{\text{total pixels}}
]

Then compute weight ( w_p ) using:

* Linear
* Power
* Saturated
* Centered weighting functions

Applied to patch embeddings before aggregation.

---

### 📊 Metrics

* Accuracy
* F1-score
* IoU (for interpretability)

---

## 📈 Results Summary

### O1 — Baseline findings

* Best model: **non-segmented images only**
* F1-score ≈ **0.89**
* Segmentation alone → unstable

### O2 — Symmetric architecture

* No performance gain
* Slight drop vs baseline
* Shows segmentation is not naturally exploited

### O3 — Patch weighting

* Improves performance:

  * F1 ≈ **0.85–0.86**
* Better spatial focus

### O4 — Interpretability

* Attention aligns partially with plant regions
* IoU ≈ **0.45–0.52**

### O5 — IoU loss

* Improves spatial alignment:

  * IoU ≈ **0.53**
* Trade-off:

  * Slight instability in training
  * Similar classification performance

---

## 🧪 Key Insights

* Segmentation **as input ≠ useful by default**
* Segmentation **as spatial prior = effective**
* There is a **trade-off**:

  * Performance vs Interpretability
* Explicit constraints (IoU loss) improve alignment

---

## 📂 Repository Structure (example)

```bash
├── data/                # Not included (private dataset)
├── models/              # CrossViT implementations
├── notebooks/           # Experiments & preprocessing
├── scripts/             # Data processing scripts
├── results/             # Plots, metrics
├── Projet_Deep_Learning.ipynb
└── README.md
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (example)
python train.py --config config.yaml
```

Or open the notebook:

```bash
Projet_Deep_Learning.ipynb
```

---

## 🔒 Data Availability

The dataset is **not included** in this repository because it belongs to a research lab.

However, the code is fully reproducible with a similar dataset of:

* Paired segmented / non-segmented images
* Binary labels

---

## 📚 Report

Full project report available here:
📄 *Rapport Projet DL - Groupe 3* 

---

## 👥 Authors

* Tasnim ATIG
* Noussayba EL MARRAKCHI
* Ahmed MAAOUIA
* Abdoulaye MBAYE

---

## 🔮 Future Work

* Improve segmentation quality
* Explore attention supervision methods
* Test larger datasets
* Apply to multi-class classification

