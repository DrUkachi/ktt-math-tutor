"""Count objects in a rendered scene (e.g. 'how many goats?').

Two backends:

- ``OwlVitCounter`` — open-vocabulary detector (`google/owlvit-base-patch32`),
  loaded lazily, falls back if the weights file is absent.
- ``BlobCounter`` — OpenCV connected-components baseline, zero-MB footprint.

For Tier 3 the brief allows either; we ship the blob counter as default and
keep OwlVit as a stretch upgrade.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class ObjectCounter(Protocol):
    def count(self, image_path: str | Path, label: str) -> int: ...


class BlobCounter:
    """Connected-components baseline. Works on rendered icon scenes where
    objects are well-separated and on a uniform background."""

    def __init__(self, min_area: int = 200, threshold: int = 200):
        self.min_area = min_area
        self.threshold = threshold

    def count(self, image_path: str | Path, label: str) -> int:
        # ``label`` is ignored — this baseline counts foreground blobs.
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("opencv-python required for BlobCounter") from e
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_path)
        _, mask = cv2.threshold(
            img, self.threshold, 255, cv2.THRESH_BINARY_INV
        )
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        # Skip background (label 0); count blobs above min_area.
        return int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= self.min_area))


class OwlVitCounter:
    """Open-vocabulary detector. Lazy-imports transformers."""

    def __init__(self, model_id: str = "google/owlvit-base-patch32",
                 score_thresh: float = 0.2):
        self.model_id = model_id
        self.score_thresh = score_thresh
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
        except ImportError as e:
            raise RuntimeError(
                "transformers required for OwlVitCounter; "
                "use BlobCounter to stay within footprint."
            ) from e
        self._processor = OwlViTProcessor.from_pretrained(self.model_id)
        self._model = OwlViTForObjectDetection.from_pretrained(self.model_id)

    def count(self, image_path: str | Path, label: str) -> int:
        self._load()
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(text=[[label]], images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=self.score_thresh, target_sizes=target_sizes
        )[0]
        return int(len(results["scores"]))
