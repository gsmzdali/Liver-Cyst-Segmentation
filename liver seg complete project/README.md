# Liver Cyst Segmentation

### Comparing classical contour-based image processing, U-Net and YOLOv8-Seg for medical image segmentation

**MSc Advanced Computer Science dissertation · University of Sheffield · 2025**  
**Researcher:** Ali Gasimzade  
**Supervisor:** Dr Charles Grellois  
**Dissertation mark:** **75/100 (distinction-level)**

## Overview

This repository showcases the implementation and experimental findings of my MSc dissertation, **_Liver Cyst Segmentation: A Comparison of Contour-Based and Deep Learning Methods_**. I investigated three complementary approaches to segmenting liver cysts and other lesions: a transparent, training-free image-processing pipeline; **U-Net** for semantic segmentation; and **YOLOv8-Seg** for instance segmentation.

The project combines **Python, OpenCV, PyTorch-based deep learning, Ultralytics YOLO, Jupyter notebooks and a Flask demonstration interface**. The research was undertaken in a context of NHS-linked medical imaging research, with reported interest from the NHS laboratory in potential use of the work. It investigates approaches that could contribute to future computer-assisted imaging workflows while recognising the additional steps required for clinical translation.

> ## Responsible Use and Clinical Research Potential

This project was developed as part of an MSc dissertation at the University of Sheffield, investigating AI-driven liver lesion segmentation with potential applications in clinical imaging workflows, including NHS healthcare settings.

The research explores how deep learning and classical image processing could support medical image analysis and contribute to future computer-assisted diagnostic tools. While the experimental results demonstrate promising segmentation performance, further clinical evaluation, external validation and appropriate regulatory approvals would be required before deployment in patient care.

The project incorporates research involving NHS-derived MRI data under applicable research governance arrangements. Restricted clinical data are not publicly distributed. Third-party datasets and pretrained models remain subject to their respective licences and permissions.

## Research contributions

- **Designed and investigated three segmentation approaches** spanning interpretable image processing, semantic segmentation and instance segmentation.
- **Implemented a contour-based workflow** using intensity processing, CLAHE, thresholding, morphological operations and OpenCV contour extraction, with a Python GUI.
- **Developed and evaluated a U-Net pipeline** for pixel-wise lesion segmentation using annotated CT images.
- **Trained and assessed YOLOv8-Seg** for lesion localisation and per-instance mask prediction.
- **Built a Flask-based demonstration interface** connecting model code with a user-facing workflow.
- **Analysed model behaviour quantitatively**, using Dice, IoU, precision–recall and mask mAP, and discussed limitations including annotation scarcity, class imbalance and CT–MRI domain differences.

## Research question and motivation

Medical image segmentation is challenging when lesion boundaries are faint, contrast varies across scans, lesions differ in size and appearance, or expert annotations are limited. The dissertation asks how **classical contour extraction and deep learning** can address these challenges, and what trade-offs they introduce in interpretability, accuracy, data requirements and potential clinical applicability.

## Methods and implementations

### 1. Classical contour-based processing (MRI)

The classical pipeline investigates **intensity normalisation, contrast enhancement with CLAHE, thresholding, morphological filtering and contour extraction**. Unlike a learned model, it can be applied without training labels and offers a transparent processing sequence. Its limitations include sensitivity to contrast and parameter choices, particularly around ambiguous lesion boundaries.

**Code:** [`cyst_gui/cyst_gui.py`](cyst_gui/cyst_gui.py)

### 2. U-Net semantic segmentation (CT)

A convolutional **encoder–decoder with skip connections** learns pixel-level segmentation masks from annotated CT images. The accompanying notebook documents preprocessing, training and evaluation; the Flask project contains related model code.

**Code:** [U-Net training notebook](liver_cyst_seg%20UNET%20training.ipynb) · [`webapp/unet_model.py`](webapp/unet_model.py)

### 3. YOLOv8-Seg instance segmentation (CT)

The YOLOv8-Seg experiments investigate **lesion localisation and instance-level mask generation**, with saved training configurations, checkpoint files and aggregate evaluation plots. The project examines mask precision–recall and average precision in addition to segmentation quality.

**Code:** [YOLOv8-Seg training notebook](liver_cyst_seg%20YOLO%20training.ipynb) · [Training configuration](liver%20cyst%20seg%20model%20yolo%20results/args.yaml)

### Demonstration application

A **Flask web application** provides an experimental interface for the segmentation workflow. The source is included for inspection, but its U-Net inference path requires a checkpoint omitted from the public GitHub package because of file-size constraints.

**Code:** [`webapp/app.py`](webapp/app.py) · [`webapp/templates/`](webapp/templates/)

## Datasets and evaluation design

| Data source | Research role | Public repository status |
|---|---|---|
| **NHS liver-cyst MRI subset** | Classical contour-based investigation | Restricted data not distributed |
| **Liver Verify CT dataset** | Annotated CT data for deep-learning experiments | Dataset images and masks not redistributed |
| **LiTS, MSD and 3D-IRCADb** | Background and related benchmark context | Refer to original providers |

**Comparison caveat:** The classical approach was explored on MRI, while U-Net and YOLOv8-Seg were assessed on annotated CT data. These are **different experimental settings**, not a controlled three-model comparison on an identical dataset. The classical MRI work is therefore not assigned directly comparable Dice/mAP figures.

## Experimental results

The following are **figures reported in the MSc dissertation**; they are not claims of clinical performance or independently reproduced results from this public package.

| Model and measure | Reported result | Evaluation context |
|---|---:|---|
| U-Net — mean Dice | **≈ 0.831** | Validation result described in the results chapter |
| U-Net — mean Dice | **≈ 0.827** | Held-out test result described in the dissertation |
| YOLOv8-Seg — mask mAP@0.5 | **≈ 0.90** | Reported mask evaluation |
| YOLOv8-Seg — mask mAP@0.5:0.95 | **≈ 0.57** | Reported mask evaluation |

**Metric interpretation:** Dice measures overlap between predicted and reference masks; IoU is another overlap measure; mask mAP evaluates instance-segmentation performance over precision–recall operating points and IoU thresholds. **Dice and mAP are not interchangeable**, and metrics from different splits should not be treated as directly equivalent.

**Dissertation reporting note:** The abstract also reports an approximately **0.78 U-Net mean Dice** under a different summary of results. The validation and held-out test values above are attributed to the results chapter; the underlying split and reporting context should be checked in the full dissertation before using any figure as a single definitive benchmark.

### Evaluation visualisations

**U-Net**

- [Mean Dice learning curve](Liver%20cyst%20seg%20unet%20data%20and%20results/results/UNET-curve_mdice.png)
- [Mean IoU learning curve](Liver%20cyst%20seg%20unet%20data%20and%20results/results/UNET-curve_miou.png)
- [Per-class Dice at epoch 30](Liver%20cyst%20seg%20unet%20data%20and%20results/results/UNET-per_class_dice_epoch_30.png)
- [Training log (CSV)](Liver%20cyst%20seg%20unet%20data%20and%20results/results/UNET-training_log.csv)

**YOLOv8-Seg**

- [Mask precision–recall curve](liver%20cyst%20seg%20model%20yolo%20results/YOLO-MaskPR_curve.png)
- [Training metrics](liver%20cyst%20seg%20model%20yolo%20results/YOLO-results.png)
- [Normalised confusion matrix](liver%20cyst%20seg%20model%20yolo%20results/YOLO-confusion_matrix_normalized.png)
- [Training metrics (CSV)](liver%20cyst%20seg%20model%20yolo%20results/YOLO-results.csv)

## Repository structure

```text
Liver-Cyst-Segmentation/
├── README.md
├── .gitignore
├── liver_cyst_seg UNET training.ipynb
├── liver_cyst_seg YOLO training.ipynb
├── best.pt
├── cyst_gui/
│   └── cyst_gui.py
├── webapp/
│   ├── app.py
│   ├── unet_model.py
│   ├── best.pt
│   ├── templates/
│   └── static/style.css
├── Liver cyst seg unet data and results/
│   └── results/                # Aggregate U-Net metrics and plots
└── liver cyst seg model yolo results/
    ├── args.yaml
    ├── weights/                # YOLO checkpoints
    └── YOLO-*                  # Aggregate YOLO metrics and plots
```

The displayed tree is a **readable overview**, not a full inventory. The public GitHub package retains the original Python source and notebook **code cells**. Saved notebook outputs and execution counts were cleared to avoid inadvertently publishing medical images; no notebook code cells were rewritten. The full original research workspace remains separate.

### What's intentionally not included

- Restricted NHS MRI images and other patient-image outputs.
- CT training/validation/test images and masks, pending confirmation of redistribution rights.
- Python virtual environments, IDE settings and generated caches.
- Two large U-Net `.pth` checkpoints (each above GitHub's standard 100 MB per-file limit).
- Dissertation documents and medical-image mosaics pending public-sharing review.

The retained YOLO `.pt` checkpoints and aggregate evaluation artefacts allow inspection of the experimental work, but **the public repository is not a fully self-contained reproduction of every experiment**.

## Running the code

This is a **research-code portfolio**, not a tested one-command deployment package. The original source has not been refactored for portability.

1. Create a new Python virtual environment.
2. Install the libraries required by the relevant component, as indicated by its imports (for example Python, PyTorch, Ultralytics, OpenCV, NumPy, Jupyter and Flask). Exact historical package versions are not guaranteed by this repository.
3. Obtain the relevant dataset from its original, authorised source and comply with its licence or research agreement.
4. Adjust local dataset and checkpoint paths for your environment; paths in the original notebooks may reflect the research machine.
5. Run the desired notebook, contour GUI or Flask app. U-Net inference requires a compatible checkpoint that is **not included** in this public copy.

## Limitations and future directions

The dissertation identifies several challenges: **limited MRI ground-truth annotations, heterogeneous lesion appearance, class imbalance, CT–MRI domain shift and the need for independent clinical validation**. These are important considerations when interpreting model performance and transferring experimental methods into real clinical settings.

Potential next steps include **uncertainty-aware segmentation, active learning, domain adaptation, more data-efficient training, external evaluation and clinician-annotated MRI benchmarking**, subject to the necessary data access and ethics approvals.

## Dissertation and attribution

**Gasimzade, A. (2025). _Liver Cyst Segmentation: A Comparison of Contour-Based and Deep Learning Methods._** MSc Advanced Computer Science dissertation, University of Sheffield. **Supervisor:** Dr Charles Grellois. **Mark:** 75/100.

The full dissertation may be made available to prospective academic supervisors **where the applicable permissions allow**. Third-party datasets, pretrained models and research materials remain subject to their original licences and governance arrangements.

## Responsible Use and Clinical Research Potential

This project was developed as part of an MSc dissertation at the University of Sheffield, investigating AI-driven liver lesion segmentation with potential applications in clinical imaging workflows, including NHS healthcare settings. **The NHS laboratory involved in the research expressed interest in potential use of the work**, providing a practical motivation for exploring methods that could support medical-image analysis. This interest should not be interpreted as evidence that the software has been adopted or deployed by the NHS.

The research explores how deep learning and classical image processing could support medical image analysis and contribute to future computer-assisted diagnostic tools. While the experimental results demonstrate promising segmentation performance, further clinical evaluation, external validation and appropriate regulatory approvals would be required before deployment in patient care.

The project incorporates research involving NHS-derived MRI data under applicable research governance arrangements. Restricted clinical data are not publicly distributed. Third-party datasets and pretrained models remain subject to their respective licences and permissions.
