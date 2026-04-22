# Karyotype Generation Pipeline

This repository implements a multi-stage deep learning pipeline for semi-automated karyotype generation and structural chromosomal anomaly detection.

## Overview

The system follows a clinically inspired workflow:

1. Overlapped metaphase filtering
2. Single chromosome detection (YOLOv8)
3. Homologous pair classification
4. Karyotype assembly
5. Structural anomaly detection (Autoencoder)

## Project Structure

```
karyotype-pipeline/
├── notebooks/
│   ├── 01_setup_env.ipynb              # Environment setup & path validation
│   ├── 02_prepare_data.ipynb           # VOC→YOLO conversion + ResNet ImageFolder
│   ├── 03_train_models.ipynb           # Train YOLOv8 (multi/single) + ResNet50
│   ├── 04_pipeline_A.ipynb             # Pipeline A: YOLO 24-class end-to-end
│   ├── 05_pipeline_B.ipynb             # Pipeline B: YOLO single + ResNet50 end-to-end
│   └── 06_evaluation.ipynb             # End-to-end evaluation & clinical metrics
├── src/
│   ├── pipeline_a.py                   # Pipeline A orchestrator
│   ├── pipeline_b.py                   # Pipeline B orchestrator
│   ├── overlap_filter.py               # Overlapped metaphase gate
│   ├── yolo_detector.py                # YOLOv8 detection wrapper
│   ├── resnet_classifier.py            # ResNet50 classification wrapper
│   ├── karyotype_assembler.py          # Karyotype grid assembly
│   └── anomaly_detector.py             # Autoencoder structural anomaly detection
├── configs/
│   └── config.example.json             # Configuration template
├── models/
│   └── README.md                       # Instructions for placing model weights
├── data/
│   └── examples/                       # Example data structure
├── requirements.txt
└── .gitignore
```

## Pipelines

### Pipeline A — Direct Multi-class Detection
`overlap filter → YOLO 24 classes → count validation → karyotype assembly → autoencoder`

Uses a single YOLOv8 model trained to directly classify all 24 chromosome pairs in one pass.

### Pipeline B — Detect-then-Classify
`overlap filter → YOLO single-class → ResNet50 classification → count validation → assembly → autoencoder`

Uses a two-stage approach: first a single-class detector crops individual chromosomes, then ResNet50 classifies each crop into one of 24 homologous pairs.

## Models

Pretrained weights are not included due to size constraints. Please place them in the `models/` directory following the naming convention in `models/README.md`.

Required model files:
| File | Description |
|------|-------------|
| `overlap_modelo_entrenado2.pt` | YOLOv8 overlapped metaphase detector |
| `best-24_chromosomes.pt` | YOLOv8 24-class chromosome detector (Pipeline A) |
| `best_single_chromosomes.pt` | YOLOv8 single-class chromosome detector (Pipeline B) |
| `ResNet50_cromosomas_model.keras` | ResNet50 pair classifier (Pipeline B) |
| `autoencoder_model.keras` | Convolutional autoencoder for structural anomaly detection |

## Data Structure

Input data must follow this structure in Google Drive:

```
54816/
├── 24_chromosomes_object/
│   ├── JEPG/          # Metaphase images (~5,000)
│   └── annotations/   # Pascal VOC XML annotations
├── single_chromosomes_object/
│   ├── JEPG/          # Single chromosome images
│   └── annotations/   # Pascal VOC XML annotations
├── normal.csv
├── number_abnormalities.csv
└── structural_abnormalities.csv
```

## Reproducibility

This project was developed and tested using **Google Colab** with GPU acceleration (T4/A100). All notebooks are designed to:
- Auto-install required dependencies on first run
- Persist all outputs to Google Drive to survive session interruptions
- Resume interrupted training via `last.pt` checkpoints

### Running the full pipeline

1. Open `notebooks/01_setup_env.ipynb` in Google Colab and run all cells
2. Run `notebooks/02_prepare_data.ipynb` to prepare datasets (one-time operation)
3. Run `notebooks/03_train_models.ipynb` to train YOLOv8 and ResNet50 — **or** place pretrained weights in `models/`
4. Run `notebooks/04_pipeline_A.ipynb` or `notebooks/05_pipeline_B.ipynb` to generate karyotypes
5. Run `notebooks/06_evaluation.ipynb` for quantitative evaluation

## Requirements

See `requirements.txt`. Core dependencies:
- `ultralytics >= 8.0`
- `torch` / `torchvision`
- `tensorflow >= 2.13`
- `opencv-python`
- `numpy`, `pandas`, `tqdm`, `matplotlib`

## Author

Carolina Rosas Alatriste

*This work was developed as part of a graduate thesis and submitted for publication in Scientific Reports.*
