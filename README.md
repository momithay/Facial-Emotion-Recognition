# Facial-Emotion-Recognition


A multi-task deep learning system that jointly predicts **high-level emotion** (positive / negative / surprise) and **15 Facial Action Coding System (FACS) Action Units** from a single facial image, using transfer learning on top of MobileNetV2.

> Built as an applied computer vision project exploring multi-task learning, transfer learning, and class-imbalance handling on a real affective-computing dataset (CK+).

---

## Table of Contents
- [Overview](#overview)
- [Why Multi-Task Learning?](#why-multi-task-learning)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

Facial expression recognition is usually framed as a single classification problem ("what emotion is this?"), but the underlying signal is really a combination of localized muscle movements — **Action Units (AUs)** — defined by the Facial Action Coding System. This project treats emotion recognition as a **multi-task learning problem**: a shared convolutional backbone learns visual features that feed two task-specific heads —

1. **Emotion head** — 3-class softmax classifier (`positive`, `negative`, `surprise`)
2. **FACS head** — 15-unit multi-label sigmoid classifier over individual Action Units (AU1, AU2, AU4, AU6, AU7, AU11, AU12, AU14, AU15, AU17, AU23, AU24, AU25, AU26, AU27)

Training both heads jointly encourages the shared backbone to learn representations grounded in the actual muscular basis of expression, rather than only the coarse emotion label.

## Why Multi-Task Learning?

- **Richer supervision signal**: AU labels are a more fine-grained, physiologically grounded signal than a single emotion class, which can act as a regularizer for the shared trunk.
- **Interpretability**: predicting AUs alongside emotion gives a human-readable explanation for *why* the model predicted a given emotion (e.g., AU12 "lip corner puller" contributing to "positive").
- **Data efficiency**: sharing a backbone across two related tasks makes better use of a relatively small, imbalanced dataset than training two separate networks.

## Dataset

- **Source**: [CK+ (Extended Cohn-Kanade)](https://www.jeffcohn.net/Resources/) facial expression dataset — image sequences labeled with emotion and AU annotations.
- **Labels used**: a derived `data_labels.csv` mapping each image `filepath` to a `high_level_emotion` label and 15 binary AU columns.
- **Class imbalance**: the raw dataset is heavily skewed toward certain emotions. This was addressed via:
  - Stratified **train / validation / test** splits (64% / 16% / 20%)
  - **Upsampling** of minority emotion classes on the training set using `sklearn.utils.resample`
  - Aggressive **data augmentation** (see below)

## Pipeline

1. **Recursive image discovery** — a custom directory walker (`list_folders_and_images`) traverses the nested CK+ folder structure and collects all `.png` image paths.
2. **Image loading & resizing** — each image is loaded with PIL, resized to `224×224`, and converted to a NumPy array; duplicate filepaths are dropped to keep the dataframe consistent.
3. **Grayscale → RGB normalization** — CK+ contains a mix of grayscale and RGB images. A conversion step stacks grayscale channels 3× so every image is a uniform 3-channel input compatible with MobileNetV2's ImageNet-pretrained weights.
4. **Label engineering** — emotion labels are mapped to integers (`positive: 0, negative: 1, surprise: 2`); AU columns are kept as a 15-dim multi-hot vector.
5. **Data augmentation** (via `ImageDataGenerator`) on the training split only:
   - Rotation (±15°), horizontal & vertical flips
   - Zoom (±20%), brightness jitter (80–120%)
   - Channel shift, pixel rescaling to `[0, 1]`
6. **Class-balance correction** — minority-class upsampling on the training set prior to model fitting.

## Model Architecture

```
Input (224x224x3)
      │
MobileNetV2 (ImageNet weights, frozen base)
      │
GlobalAveragePooling2D
      │
Dense(64, relu) → BatchNormalization → Dropout(0.5)
      │
      ├── Dense(3, softmax)   → emotion_output   (sparse_categorical_crossentropy)
      └── Dense(15, sigmoid)  → facs_output       (binary_crossentropy)
```

- **Backbone**: MobileNetV2 pretrained on ImageNet, used as a frozen feature extractor — chosen for its efficiency/accuracy trade-off, well suited for lightweight or on-device inference.
- **Multi-task heads**: a shared dense projection feeds two independent output layers, trained jointly with a combined loss (categorical cross-entropy for emotion + binary cross-entropy for AUs).
- **Regularization**: `BatchNormalization` and `Dropout(0.5)` were added iteratively after the first baseline model showed signs of overfitting.

## Training Strategy

- **Baseline model** → simple dense head, no regularization, 10 epochs.
- **Hyperparameter search** — a grid search over:
  - Learning rate: `{1e-4, 1e-3}`
  - Dense layer units: `{32, 64}`
  - Dropout rate: `{0.3, 0.5}`

  with `EarlyStopping` (patience=5, restoring best weights) to prevent overfitting during the search.
- **Final model** — best-performing configuration retrained with `BatchNormalization` + `Dropout(0.5)` and the Adam optimizer.
- **Inference utility** — a standalone prediction script loads a single image, applies the same preprocessing pipeline, and returns both the predicted emotion label and the binarized AU vector (threshold = 0.5).

## Results

Evaluated on a held-out test split:

| Task            | Precision | Recall | ROC AUC |
|-----------------|:---------:|:------:|:-------:|
| Emotion (macro) | 0.614     | 0.497  | 0.713   |
| FACS AU (micro) | 0.399     | 0.377  | 0.661   |

**Overall emotion classification accuracy: 66.96%**

These results reflect the challenge of a small, imbalanced dataset with a frozen backbone; see [Limitations & Future Work](#limitations--future-work) for how this would be pushed further.

## Key Engineering Decisions

- **Transfer learning over training from scratch** — with a limited-size dataset, a frozen ImageNet backbone gave far more stable convergence than training a CNN end-to-end.
- **Multi-task over single-task** — sharing a trunk across emotion + AU prediction was a deliberate choice to squeeze more signal out of a small labeled set and produce more explainable predictions.
- **Systematic hyperparameter search over ad-hoc tuning** — a `ParameterGrid` sweep with early stopping was used to make the tuning process reproducible and comparable across runs.
- **Explicit imbalance handling** — both data-level (upsampling, augmentation) and evaluation-level (macro/micro precision-recall, ROC AUC rather than raw accuracy) treatment of class imbalance, since accuracy alone is a misleading metric on this dataset.

## Tech Stack

`Python` · `TensorFlow / Keras` · `MobileNetV2` · `scikit-learn` · `pandas` / `NumPy` · `PIL` / `OpenCV` · `Matplotlib` · Google Colab (GPU runtime)

## Repository Structure

```
.
├── notebooks/
│   └── facial_emotion_facs_multitask.ipynb   # main notebook: data prep, training, evaluation
├── data/                                     # CK+ images + data_labels.csv (not committed — see note below)
├── README.md
└── requirements.txt
```

> **Note on data**: the CK+ dataset is distributed under its own research-use license and is not redistributed in this repository. See the [Dataset](#dataset) section for the source and instructions to request access, then place the images and `data_labels.csv` under `data/`.

## How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the CK+ dataset (see [Dataset](#dataset)) and place it under `data/`.
3. Open `notebooks/facial_emotion_facs_multitask.ipynb` and run cells top-to-bottom (originally developed in Google Colab with data mounted from Drive — update the working-directory cell if running locally).
4. The final trained model can be used for single-image inference via the prediction cell at the end of the notebook.

## Limitations & Future Work

- **Fine-tune the backbone**: unfreezing the top MobileNetV2 blocks with a low learning rate could recover more emotion-specific features.
- **Larger / more diverse data**: CK+ is a lab-collected, posed-expression dataset; results would need validation on in-the-wild datasets (e.g., AffectNet, RAF-DB) before any production use.
- **Loss weighting**: the combined loss currently sums emotion and AU losses unweighted — a tuned weighting (or uncertainty-based multi-task weighting) could better balance the two objectives.
- **AU-level metrics**: reporting per-AU precision/recall (rather than only micro-averaged) would better reveal which action units are hardest to detect.
