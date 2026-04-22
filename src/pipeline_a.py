"""
pipeline_a.py
-------------
Pipeline A — Direct multi-class detection.

Workflow
--------
  overlap filter → YOLO 24-class → count validation
  → karyotype assembly → autoencoder anomaly detection

Usage (standalone)
------------------
  python src/pipeline_a.py --config configs/config.json \\
                            --images /path/to/metaphase/images \\
                            --out    /path/to/output/dir
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from overlap_filter import OverlapFilter, list_images
from yolo_detector import YOLODetector
from karyotype_assembler import KaryotypeAssembler, numerical_alert
from anomaly_detector import AnomalyDetector


def run_pipeline_a(
    images_dir: str | Path,
    out_dir: str | Path,
    overlap_model: str | Path,
    yolo24_model: str | Path,
    ae_model: Optional[str | Path] = None,
    overlap_conf: float = 0.25,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.70,
    ae_threshold: Optional[float] = None,
    ae_percentile: float = 95.0,
    device: int | str = 0,
    karyotype_order: Optional[list[str]] = None,
    cell_size: tuple[int, int] = (220, 220),
    cols: int = 4,
    pad: int = 10,
) -> list[dict]:
    """
    Run Pipeline A end-to-end on a directory of metaphase images.

    Parameters
    ----------
    images_dir : directory containing metaphase images (JPEG/PNG).
    out_dir    : output directory; sub-folders created automatically.
    overlap_model : path to overlapped-gate YOLOv8 weights.
    yolo24_model  : path to 24-class chromosome YOLOv8 weights.
    ae_model      : (optional) path to autoencoder Keras weights.
    overlap_conf  : confidence threshold for overlap gate.
    yolo_conf     : confidence threshold for chromosome detection.
    yolo_iou      : IoU threshold for NMS.
    ae_threshold  : fixed AE threshold; if None and ae_model given, fitted at p95.
    ae_percentile : percentile for AE threshold fitting.
    device        : torch device (0=first GPU, "cpu"=CPU).
    karyotype_order : chromosome label order (default KARYOTYPE_ORDER).
    cell_size, cols, pad : karyotype grid parameters.

    Returns
    -------
    results : list of per-image result dicts with keys:
        image_name, kept, counts, numerical_alerts,
        karyotype_path, ae_scores (if AE provided).
    """
    out_dir = Path(out_dir)
    karyotype_dir = out_dir / "karyotypes_A"
    karyotype_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Overlap gate ───────────────────────────────────────────────
    print("[Pipeline A] Stage 1/4 — Overlap gate …")
    gate = OverlapFilter(
        model_path=overlap_model,
        conf_thr=overlap_conf,
        device=device,
    )
    all_images = list_images(images_dir)
    kept_paths, _ = gate.filter_images(all_images)
    print(f"  Kept {len(kept_paths)} / {len(all_images)} images after gate.")

    # ── 2. YOLO 24-class detection ────────────────────────────────────
    print("[Pipeline A] Stage 2/4 — YOLO 24-class detection …")
    detector = YOLODetector(
        model_path=yolo24_model,
        conf_thr=yolo_conf,
        iou_thr=yolo_iou,
        device=device,
    )
    assembler = KaryotypeAssembler(
        order=karyotype_order,
        cell_size=cell_size,
        cols=cols,
        pad=pad,
    )

    # ── 3. Optional: load AE ──────────────────────────────────────────
    ae: Optional[AnomalyDetector] = None
    if ae_model and Path(ae_model).exists():
        print("[Pipeline A] Loading autoencoder …")
        ae = AnomalyDetector(
            model_path=ae_model,
            threshold=ae_threshold,
        )

    # ── 4. Process each image ─────────────────────────────────────────
    print("[Pipeline A] Stage 3-4/4 — Detect → assemble → score …")
    results: list[dict] = []
    all_crops_for_ae_fit: list[np.ndarray] = []

    for img_path in tqdm(kept_paths, desc="Pipeline A"):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        det_and_crops = detector.predict_and_crop(img_path)
        counts: dict[str, int] = {}
        crops_by_label: dict[str, list[np.ndarray]] = {}

        for det, crop in det_and_crops:
            lbl = det.label
            counts[lbl] = counts.get(lbl, 0) + 1
            crops_by_label.setdefault(lbl, []).append(crop)
            all_crops_for_ae_fit.append(crop)

        # Numerical count validation
        alerts = numerical_alert(counts)

        # Karyotype assembly
        karyo_path = karyotype_dir / f"{img_path.stem}_karyo_A.png"
        assembler.build(crops_by_label, output_path=karyo_path)

        entry: dict = {
            "image_name": img_path.name,
            "kept": True,
            "counts": counts,
            "numerical_alerts": alerts,
            "karyotype_path": str(karyo_path),
            "ae_scores": None,
        }

        # AE scoring (if threshold already set)
        if ae is not None and ae.threshold is not None:
            ae_scores = ae.score_karyotype(crops_by_label)
            entry["ae_scores"] = ae_scores

        results.append(entry)

    # Fit AE threshold on collected crops if not pre-set
    if ae is not None and ae.threshold is None and all_crops_for_ae_fit:
        print(f"[Pipeline A] Fitting AE threshold at p{ae_percentile} …")
        ae.fit_threshold(all_crops_for_ae_fit, percentile=ae_percentile)
        # Re-score with fitted threshold
        for entry in results:
            img_path = Path(entry["image_name"])
            # Re-detect to get crops (they are not stored to save memory)
            det_and_crops = detector.predict_and_crop(
                Path(images_dir) / img_path.name
            )
            crops_by_label = {}
            for det, crop in det_and_crops:
                crops_by_label.setdefault(det.label, []).append(crop)
            entry["ae_scores"] = ae.score_karyotype(crops_by_label)

    # Persist results
    results_path = out_dir / "pipeline_a_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Pipeline A] Done. Results saved to {results_path}")

    return results


# ── CLI entry point ────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="Run Pipeline A")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--images", required=True, help="Directory with metaphase images")
    parser.add_argument("--out",    required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"

    run_pipeline_a(
        images_dir=args.images,
        out_dir=args.out,
        overlap_model=cfg["models"]["overlap_model"],
        yolo24_model=cfg["models"]["yolo24_model"],
        ae_model=cfg["models"].get("autoencoder_model"),
        overlap_conf=cfg["inference"]["overlap_conf_thr"],
        yolo_conf=cfg["inference"]["yolo_conf_thr"],
        yolo_iou=cfg["inference"]["yolo_iou_thr"],
        ae_threshold=None,
        ae_percentile=95.0,
        device=device,
        karyotype_order=cfg["karyotype"]["order"],
        cell_size=tuple(cfg["karyotype"]["cell_size"]),
        cols=cfg["karyotype"]["cols"],
        pad=cfg["karyotype"]["pad"],
    )


if __name__ == "__main__":
    _cli()
