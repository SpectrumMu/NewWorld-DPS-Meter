"""Configuration helpers for the vision-based DPS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ROI:
    """Represents a rectangular region-of-interest within a frame."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def as_slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y2), slice(self.x, self.x2)


@dataclass
class VideoConfig:
    resize_width: Optional[int] = None
    frame_stride: int = 1


@dataclass
class ProcessingConfig:
    roi: ROI
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: int = 8
    blur_kernel: int = 3
    adaptive_threshold_block_size: int = 35
    adaptive_threshold_C: int = -5


@dataclass
class DetectorConfig:
    hsv_lower: tuple[int, int, int] = (0, 0, 200)
    hsv_upper: tuple[int, int, int] = (180, 60, 255)
    min_area: int = 15
    max_area: int = 5000
    dilation_iterations: int = 1


@dataclass
class MetricsConfig:
    decay_halflife_seconds: float = 2.0
    frame_rate_override: Optional[float] = None
    intensity_to_damage_scale: float = 1.0


@dataclass
class OutputConfig:
    save_debug_frames: bool = False
    debug_dir: Path = Path("reports/debug")
    csv_path: Path = Path("reports/sample-run.csv")
    json_path: Path = Path("reports/sample-run.json")


@dataclass
class PipelineConfig:
    video: VideoConfig
    processing: ProcessingConfig
    detector: DetectorConfig
    metrics: MetricsConfig
    output: OutputConfig

    @property
    def roi(self) -> ROI:
        return self.processing.roi


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML configuration file into strongly typed config objects."""

    cfg_path = Path(path)
    data = _load_yaml(cfg_path)

    video_cfg = data.get("video", {})
    processing_cfg = data.get("processing", {})
    detector_cfg = data.get("detector", {})
    metrics_cfg = data.get("metrics", {})
    output_cfg = data.get("output", {})

    roi_cfg = processing_cfg.get("roi")
    if not roi_cfg:
        raise ValueError("processing.roi must be provided in the config")

    roi = ROI(
        x=int(roi_cfg.get("x", 0)),
        y=int(roi_cfg.get("y", 0)),
        width=int(roi_cfg.get("width", 0)),
        height=int(roi_cfg.get("height", 0)),
    )
    if roi.width <= 0 or roi.height <= 0:
        raise ValueError("processing.roi width/height must be positive")

    video = VideoConfig(
        resize_width=video_cfg.get("resize_width"),
        frame_stride=int(video_cfg.get("frame_stride", 1)),
    )

    processing = ProcessingConfig(
        roi=roi,
        clahe_clip_limit=float(processing_cfg.get("clahe_clip_limit", 3.0)),
        clahe_tile_grid_size=int(processing_cfg.get("clahe_tile_grid_size", 8)),
        blur_kernel=int(processing_cfg.get("blur_kernel", 3)),
        adaptive_threshold_block_size=int(
            processing_cfg.get("adaptive_threshold_block_size", 35)
        ),
        adaptive_threshold_C=int(processing_cfg.get("adaptive_threshold_C", -5)),
    )

    detector = DetectorConfig(
        hsv_lower=tuple(int(v) for v in detector_cfg.get("hsv_lower", [0, 0, 200])),
        hsv_upper=tuple(int(v) for v in detector_cfg.get("hsv_upper", [180, 60, 255])),
        min_area=int(detector_cfg.get("min_area", 15)),
        max_area=int(detector_cfg.get("max_area", 5000)),
        dilation_iterations=int(detector_cfg.get("dilation_iterations", 1)),
    )

    frame_rate_override = metrics_cfg.get("frame_rate_override")
    if frame_rate_override is not None:
        frame_rate_override = float(frame_rate_override)

    metrics = MetricsConfig(
        decay_halflife_seconds=float(
            metrics_cfg.get("decay_halflife_seconds", 2.0)
        ),
        frame_rate_override=frame_rate_override,
        intensity_to_damage_scale=float(metrics_cfg.get("intensity_to_damage_scale", 1.0)),
    )

    output = OutputConfig(
        save_debug_frames=bool(output_cfg.get("save_debug_frames", False)),
        debug_dir=Path(output_cfg.get("debug_dir", "reports/debug")),
        csv_path=Path(output_cfg.get("csv_path", "reports/sample-run.csv")),
        json_path=Path(output_cfg.get("json_path", "reports/sample-run.json")),
    )

    return PipelineConfig(
        video=video,
        processing=processing,
        detector=detector,
        metrics=metrics,
        output=output,
    )
