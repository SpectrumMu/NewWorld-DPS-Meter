"""Primitive contour-based damage number detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np

from .config import PipelineConfig
from .preprocessing import FramePacket


@dataclass
class DamageCandidate:
    bbox: tuple[int, int, int, int]  # x, y, w, h in ROI coordinates
    area: float
    mean_intensity: float
    contour: np.ndarray

    @property
    def score(self) -> float:
        # Use intensity weighted area as a crude measure of "damage" magnitude
        return self.area * (self.mean_intensity / 255.0)


@dataclass
class DetectionResult:
    frame: FramePacket
    candidates: List[DamageCandidate]

    @property
    def aggregate_score(self) -> float:
        return float(sum(c.score for c in self.candidates))


def detect_damage(packet: FramePacket, cfg: PipelineConfig) -> DetectionResult:
    mask = packet.mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[DamageCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.detector.min_area or area > cfg.detector.max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        roi = mask[y : y + h, x : x + w]
        mean_intensity = float(np.mean(roi)) if roi.size else 0.0
        candidates.append(
            DamageCandidate(
                bbox=(x, y, w, h),
                area=float(area),
                mean_intensity=mean_intensity,
                contour=contour,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return DetectionResult(frame=packet, candidates=candidates)
