"""
karyotype_assembler.py
----------------------
Assembles chromosome crops into a karyotype grid image.

Also validates chromosome counts per homologous pair and raises
numerical anomaly alerts when counts deviate from the expected diploid
complement (2 per pair, with X/Y sex-chromosome logic).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


KARYOTYPE_ORDER = [
    "A1", "A2", "A3",
    "B4", "B5",
    "C6", "C7", "C8", "C9", "C10", "C11", "C12",
    "D13", "D14", "D15",
    "E16", "E17", "E18",
    "F19", "F20",
    "G21", "G22",
    "X", "Y",
]


def expected_counts(observed: dict[str, int]) -> dict[str, int]:
    """
    Return the expected diploid count per label given the observed X/Y distribution.

    Rules
    -----
    - Autosomes (1–22): expected 2 each.
    - If Y is observed (count > 0): expect X=1, Y=1  → 46,XY
    - If Y is not observed          : expect X=2, Y=0  → 46,XX
    """
    exp = {k: 2 for k in KARYOTYPE_ORDER if k not in ("X", "Y")}
    if observed.get("Y", 0) > 0:
        exp["X"] = 1
        exp["Y"] = 1
    else:
        exp["X"] = 2
        exp["Y"] = 0
    return exp


def numerical_alert(counts: dict[str, int]) -> dict[str, dict]:
    """
    Compare observed counts against the expected diploid complement.

    Returns
    -------
    alerts : dict
        {label: {"observed": int, "expected": int}} for every label
        where observed ≠ expected.  Empty dict means no anomaly.
    """
    exp = expected_counts(counts)
    alerts: dict[str, dict] = {}
    for label, e in exp.items():
        o = int(counts.get(label, 0))
        if o != int(e):
            alerts[label] = {"observed": o, "expected": int(e)}
    return alerts


class KaryotypeAssembler:
    """
    Builds a karyotype grid image from a dict of chromosome crops.

    Parameters
    ----------
    order : list[str] | None
        Ordered list of chromosome labels. Defaults to KARYOTYPE_ORDER.
    cell_size : tuple[int, int]
        (width, height) of each karyotype cell in pixels (default: (220, 220)).
    cols : int
        Number of columns in the grid (default: 4).
    pad : int
        Pixel padding between cells (default: 10).
    """

    def __init__(
        self,
        order: Optional[list[str]] = None,
        cell_size: tuple[int, int] = (220, 220),
        cols: int = 4,
        pad: int = 10,
    ):
        self.order = order if order else KARYOTYPE_ORDER
        self.cell_w, self.cell_h = cell_size
        self.cols = cols
        self.pad = pad

    def _fit_to_cell(self, img_rgb: np.ndarray) -> np.ndarray:
        """Scale *img_rgb* to fit inside a cell, centred on white background."""
        cell = np.ones((self.cell_h, self.cell_w, 3), dtype=np.uint8) * 255
        h, w = img_rgb.shape[:2]
        if h == 0 or w == 0:
            return cell
        scale = min(self.cell_h / h, self.cell_w / w)
        nh = max(1, int(h * scale))
        nw = max(1, int(w * scale))
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        oy = (self.cell_h - nh) // 2
        ox = (self.cell_w - nw) // 2
        cell[oy : oy + nh, ox : ox + nw] = resized
        return cell

    def _make_pair_cell(self, imgs: list[np.ndarray]) -> np.ndarray:
        """
        Stack up to 2 chromosome images vertically inside one cell.
        If more than 2 are present, only the first two are used
        (excess chromosomes are still reported in the count alert).
        """
        cell = np.ones((self.cell_h, self.cell_w, 3), dtype=np.uint8) * 255
        if not imgs:
            return cell

        half_h = (self.cell_h - 10) // 2

        def fit_half(img: np.ndarray, target_h: int) -> np.ndarray:
            h, w = img.shape[:2]
            s = min(target_h / h, self.cell_w / w)
            nh = max(1, int(h * s))
            nw = max(1, int(w * s))
            return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        if len(imgs) == 1:
            return self._fit_to_cell(imgs[0])

        top = fit_half(imgs[0], half_h)
        bot = fit_half(imgs[1], half_h)

        for row_img, y0 in [(top, 0), (bot, half_h + 10)]:
            ox = (self.cell_w - row_img.shape[1]) // 2
            oy = y0
            cell[oy : oy + row_img.shape[0], ox : ox + row_img.shape[1]] = row_img

        return cell

    def build(
        self,
        crops_by_label: dict[str, list[np.ndarray]],
        output_path: Optional[str | Path] = None,
    ) -> np.ndarray:
        """
        Build the karyotype canvas.

        Parameters
        ----------
        crops_by_label : dict[str, list[np.ndarray]]
            {label: [rgb_crop_1, rgb_crop_2, ...]}
        output_path : str | Path | None
            If provided, the image is saved to this path.

        Returns
        -------
        canvas : np.ndarray  (RGB, uint8)
        """
        n = len(self.order)
        rows = math.ceil(n / self.cols)
        H = rows * self.cell_h + (rows + 1) * self.pad
        W = self.cols * self.cell_w + (self.cols + 1) * self.pad
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

        for idx, label in enumerate(self.order):
            r = idx // self.cols
            c = idx % self.cols
            y0 = self.pad + r * (self.cell_h + self.pad)
            x0 = self.pad + c * (self.cell_w + self.pad)

            imgs = crops_by_label.get(label, [])
            cell = self._make_pair_cell(imgs)
            canvas[y0 : y0 + self.cell_h, x0 : x0 + self.cell_w] = cell

        if output_path is not None:
            bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr)

        return canvas
