"""Damage aggregation helpers."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .config import PipelineConfig
from .detectors import DetectionResult
from .preprocessing import VideoMetadata


@dataclass
class DamageSample:
    frame_index: int
    timestamp: float
    raw_intensity: float
    damage_estimate: float
    instantaneous_dps: float
    smoothed_dps: float


@dataclass
class DamageSummary:
    samples: List[DamageSample]
    total_damage: float
    mean_dps: float


def compute_timeseries(
    detections: Iterable[DetectionResult],
    cfg: PipelineConfig,
    fps: float,
) -> DamageSummary:
    halflife = max(0.01, cfg.metrics.decay_halflife_seconds)
    scale = cfg.metrics.intensity_to_damage_scale

    samples: List[DamageSample] = []
    total_damage = 0.0
    ema = 0.0
    last_timestamp = None

    decay_base = math.log(0.5) / halflife

    for result in detections:
        damage = result.aggregate_score * scale
        total_damage += damage

        timestamp = result.frame.timestamp
        if last_timestamp is None:
            dt = 1.0 / fps
        else:
            dt = max(1.0 / fps, timestamp - last_timestamp)
        last_timestamp = timestamp

        instantaneous_dps = damage / dt if dt > 0 else 0.0

        decay = math.exp(decay_base * dt)
        ema = ema * decay + instantaneous_dps * (1 - decay)

        samples.append(
            DamageSample(
                frame_index=result.frame.index,
                timestamp=timestamp,
                raw_intensity=result.aggregate_score,
                damage_estimate=damage,
                instantaneous_dps=instantaneous_dps,
                smoothed_dps=ema,
            )
        )

    total_time = samples[-1].timestamp if samples else 0.0
    mean_dps = total_damage / total_time if total_time > 0 else 0.0
    return DamageSummary(samples=samples, total_damage=total_damage, mean_dps=mean_dps)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_reports(
    summary: DamageSummary,
    metadata: VideoMetadata,
    cfg: PipelineConfig,
) -> None:
    csv_path = Path(cfg.output.csv_path)
    json_path = Path(cfg.output.json_path)
    _ensure_parent(csv_path)
    _ensure_parent(json_path)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame",
            "timestamp",
            "raw_intensity",
            "damage_estimate",
            "instantaneous_dps",
            "smoothed_dps",
        ])
        for sample in summary.samples:
            writer.writerow(
                [
                    sample.frame_index,
                    f"{sample.timestamp:.4f}",
                    f"{sample.raw_intensity:.3f}",
                    f"{sample.damage_estimate:.3f}",
                    f"{sample.instantaneous_dps:.3f}",
                    f"{sample.smoothed_dps:.3f}",
                ]
            )

    report = {
        "video": {
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
            "duration": metadata.duration,
        },
        "config": {
            "roi": {
                "x": cfg.roi.x,
                "y": cfg.roi.y,
                "width": cfg.roi.width,
                "height": cfg.roi.height,
            },
            "frame_stride": cfg.video.frame_stride,
            "intensity_to_damage_scale": cfg.metrics.intensity_to_damage_scale,
            "decay_halflife_seconds": cfg.metrics.decay_halflife_seconds,
        },
        "results": {
            "frames_processed": len(summary.samples),
            "total_damage": summary.total_damage,
            "mean_dps": summary.mean_dps,
            "final_smoothed_dps": summary.samples[-1].smoothed_dps if summary.samples else 0.0,
        },
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
