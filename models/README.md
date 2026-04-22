# Model Weights

Pretrained weights are **not included** in this repository due to file size constraints.

## Required files

Place the following files in this directory (`models/`) before running the pipelines:

| Filename | Pipeline | Description | Architecture |
|----------|----------|-------------|--------------|
| `overlap_modelo_entrenado2.pt` | A & B | Overlapped metaphase detector | YOLOv8s |
| `best-24_chromosomes.pt` | A only | 24-class chromosome pair detector | YOLOv8 |
| `best_single_chromosomes.pt` | B only | Single-class chromosome detector | YOLOv8 |
| `ResNet50_cromosomas_model.keras` | B only | 24-class pair classifier | ResNet50 (Keras/TF) |
| `autoencoder_model.keras` | A & B | Structural anomaly detector | Conv Autoencoder (Keras/TF) |

## Notes

- All `.pt` files are Ultralytics YOLOv8 weights (standard `best.pt` format).
- `.keras` files require TensorFlow ≥ 2.13.
- Update the paths in `configs/config.json` (copied from `config.example.json`) to reflect
  wherever you store your weights (e.g. Google Drive).
