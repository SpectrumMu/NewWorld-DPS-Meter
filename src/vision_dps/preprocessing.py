"""Video loading and pre-processing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

from .config import PipelineConfig


@dataclass
class VideoMetadata:
    fps: float
    frame_count: Optional[int]
    width: int
    height: int

    @property
    def duration(self) -> Optional[float]:
        if self.frame_count is None or self.fps <= 0:
            return None
        return self.frame_count / self.fps


@dataclass
class FramePacket:
    index: int
    timestamp: float
    raw_frame: np.ndarray
    roi_frame: np.ndarray
    mask: np.ndarray


def _resize_if_needed(frame: np.ndarray, resize_width: Optional[int]) -> np.ndarray:
    if resize_width is None:
        return frame
    h, w = frame.shape[:2]
    if w == 0 or w == resize_width:
        return frame
    scale = resize_width / float(w)
    new_size = (resize_width, int(round(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _clahe(gray: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    tile_grid_size = max(1, tile_grid_size)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray)


def _adaptive_threshold(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    block_size = max(3, block_size | 1)  # ensure odd and >=3
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )


def compute_video_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return VideoMetadata(fps=fps, frame_count=frame_count, width=width, height=height)


def iter_frames(
    video_path: str | Path,
    cfg: PipelineConfig,
) -> tuple[VideoMetadata, Generator[FramePacket, None, None]]:
    """Yield processed frame packets according to the pipeline configuration."""

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    metadata = compute_video_metadata(cap)
    fps = cfg.metrics.frame_rate_override or metadata.fps or 30.0
    stride = max(1, cfg.video.frame_stride)

    def _generator() -> Generator[FramePacket, None, None]:
        frame_index = -1
        try:
            while True:
                ret, frame = cap.read()
                frame_index += 1
                if not ret:
                    break
                if frame_index % stride != 0:
                    continue

                resized = _resize_if_needed(frame, cfg.video.resize_width)
                roi = cfg.processing.roi.as_slices()
                roi_frame = resized[roi]

                hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
                hsv_lower = np.array(cfg.detector.hsv_lower, dtype=np.uint8)
                hsv_upper = np.array(cfg.detector.hsv_upper, dtype=np.uint8)
                hsv_mask = cv2.inRange(hsv_roi, hsv_lower, hsv_upper)

                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                clahe = _clahe(
                    gray,
                    clip_limit=cfg.processing.clahe_clip_limit,
                    tile_grid_size=cfg.processing.clahe_tile_grid_size,
                )

                if cfg.processing.blur_kernel > 1:
                    blur_kernel = cfg.processing.blur_kernel | 1  # make odd
                    clahe = cv2.GaussianBlur(clahe, (blur_kernel, blur_kernel), 0)

                thresh = _adaptive_threshold(
                    clahe,
                    cfg.processing.adaptive_threshold_block_size,
                    cfg.processing.adaptive_threshold_C,
                )

                mask = cv2.bitwise_and(thresh, hsv_mask)
                if cfg.detector.dilation_iterations > 0:
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=cfg.detector.dilation_iterations)

                timestamp = frame_index / fps
                yield FramePacket(
                    index=frame_index,
                    timestamp=timestamp,
                    raw_frame=resized,
                    roi_frame=roi_frame,
                    mask=mask,
                )
        finally:
            cap.release()

    return metadata, _generator()
