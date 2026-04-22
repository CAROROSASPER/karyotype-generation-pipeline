"""
image_enhancer.py
-----------------
Pre-assembly image enhancement pipeline (paper Section 4, Algorithms 1–3).

Implements three algorithms described in the paper:

  Algorithm 1 — Crop Largest Object
    CLAHE → Gaussian blur → adaptive thresholding → morphological ops
    → find largest contour → mask → replace background with white → crop

  Algorithm 2 — Enhance Quality
    3×3 sharpening convolution (center weight 9, neighbors −1)

  Algorithm 3 — Process Images
    Apply Algorithm 1 + 2 to all images in a directory tree, preserving
    sub-folder structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Algorithm 1: Crop Largest Object ─────────────────────────────────

def crop_largest_object(image: np.ndarray, margin: int = 20) -> np.ndarray:
    """
    Isolate the largest chromosome object per image.

    Steps (paper Algorithm 1):
      1. Convert to grayscale.
      2. Enhance local contrast with CLAHE.
      3. Reduce noise with Gaussian blur.
      4. Segment foreground via adaptive thresholding.
      5. Clean mask with morphological opening + closing.
      6. Find all contours; if none, return image unchanged.
      7. Select the largest contour.
      8. Create binary mask → apply to image (bitwise AND).
      9. Replace black background with white.
      10. Compute bounding rect, expand by margin, crop.

    Parameters
    ----------
    image  : RGB or BGR ndarray (H, W, 3).
    margin : pixel margin added around the bounding rect (default 20).

    Returns
    -------
    Cropped RGB image with white background.
    """
    # Ensure we work in BGR (cv2 convention)
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()

    # Step 1: grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Step 3: Gaussian blur
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Step 4: Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )

    # Step 5: Morphological opening + closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 6: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Step 7: Largest contour
    c_max = max(contours, key=cv2.contourArea)

    # Step 8: Mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c_max], -1, 255, thickness=cv2.FILLED)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)

    # Step 9: Replace black background with white
    white_bg = np.ones_like(bgr) * 255
    inv_mask = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    result = cv2.add(masked, bg_part)

    # Step 10: Bounding rect + margin + crop
    x, y, w, h = cv2.boundingRect(c_max)
    H_img, W_img = result.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(W_img, x + w + margin)
    y2 = min(H_img, y + h + margin)
    cropped = result[y1:y2, x1:x2]

    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


# ── Algorithm 2: Enhance Quality ─────────────────────────────────────

_SHARPEN_KERNEL = np.array(
    [[-1, -1, -1, -1, -1],
     [-1,  2,  2,  2, -1],
     [-1,  2,  9,  2, -1],
     [-1,  2,  2,  2, -1],
     [-1, -1, -1, -1, -1]],
    dtype=np.float32,
)
# Normalize so weights sum to a positive value
_SHARPEN_KERNEL /= _SHARPEN_KERNEL.sum() if _SHARPEN_KERNEL.sum() > 0 else 1


def enhance_quality(image: np.ndarray) -> np.ndarray:
    """
    Sharpen an image using a 2D convolution kernel (paper Algorithm 2).

    A 3×3 kernel with high center weight (≈9) and negative neighbors (≈−1)
    enhances edges and fine chromosomal band structures.

    Parameters
    ----------
    image : RGB ndarray (H, W, 3) or grayscale (H, W).

    Returns
    -------
    Sharpened image (same dtype as input, clipped to valid range).
    """
    # Use a simple 3×3 unsharp-mask style kernel as described in the paper
    kernel_3x3 = np.array(
        [[-1, -1, -1],
         [-1,  9, -1],
         [-1, -1, -1]],
        dtype=np.float32,
    )
    sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel_3x3)
    return sharpened


# ── Algorithm 3: Process Images ───────────────────────────────────────

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def process_images(
    src_dir: str | Path,
    dst_dir: str | Path,
    margin: int = 20,
    verbose: bool = True,
) -> int:
    """
    Apply crop_largest_object + enhance_quality to every image in *src_dir*,
    preserving sub-folder structure in *dst_dir* (paper Algorithm 3).

    Parameters
    ----------
    src_dir : source root directory.
    dst_dir : destination root directory.
    margin  : pixel margin for crop_largest_object.
    verbose : print progress.

    Returns
    -------
    Number of images processed.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for img_path in src_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        # Algorithm 1
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cropped = crop_largest_object(img_rgb, margin=margin)

        # Algorithm 2
        sharpened = enhance_quality(cropped)

        # Preserve relative path
        rel = img_path.relative_to(src_dir)
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR))

        processed += 1
        if verbose and processed % 100 == 0:
            print(f"  Processed {processed} images…")

    if verbose:
        print(f"[DONE] {processed} images saved to {dst_dir}")
    return processed
