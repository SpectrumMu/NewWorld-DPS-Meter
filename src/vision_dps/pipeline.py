"""Entry point for the DPS estimation baseline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import cv2
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import PipelineConfig, load_config
from .detectors import DetectionResult, detect_damage
from .metrics import DamageSummary, compute_timeseries, write_reports
from .preprocessing import FramePacket, iter_frames

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _apply_output_override(cfg: PipelineConfig, output_dir: Optional[Path]) -> PipelineConfig:
    if output_dir is None:
        return cfg
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.output
    cfg.output = replace(
        output,
        debug_dir=output_dir / Path(output.debug_dir).name,
        csv_path=output_dir / Path(output.csv_path).name,
        json_path=output_dir / Path(output.json_path).name,
    )
    return cfg


def _write_debug_frame(detection: DetectionResult, cfg: PipelineConfig) -> None:
    cfg.output.debug_dir.mkdir(parents=True, exist_ok=True)
    frame_bgr = detection.frame.roi_frame.copy()
    for candidate in detection.candidates:
        x, y, w, h = candidate.bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            frame_bgr,
            f"{candidate.score:.1f}",
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    debug_path = cfg.output.debug_dir / f"frame-{detection.frame.index:06d}.png"
    cv2.imwrite(str(debug_path), frame_bgr)


def _collect_detections(
    packets: Iterable[FramePacket],
    cfg: PipelineConfig,
    show_debug: bool,
) -> list[DetectionResult]:
    detections: list[DetectionResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Processing frames", start=False)
        for packet in packets:
            if not progress.tasks[task].started:
                progress.start_task(task)
            detection = detect_damage(packet, cfg)
            detections.append(detection)

            if cfg.output.save_debug_frames or show_debug:
                _write_debug_frame(detection, cfg)

            progress.advance(task)

    return detections


def run_pipeline(
    video_path: Path,
    config_path: Path,
    output_dir: Optional[Path] = None,
    force_debug: bool = False,
) -> DamageSummary:
    cfg = load_config(config_path)
    cfg = _apply_output_override(cfg, output_dir)
    if force_debug:
        cfg.output = replace(cfg.output, save_debug_frames=True)

    metadata, frame_iter = iter_frames(video_path, cfg)
    console.log(
        "Loaded video",
        {
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "size": (metadata.width, metadata.height),
        },
    )

    detections = _collect_detections(frame_iter, cfg, show_debug=force_debug)
    summary = compute_timeseries(detections, cfg, fps=cfg.metrics.frame_rate_override or metadata.fps or 30.0)
    write_reports(summary, metadata, cfg)

    console.log(
        "Run complete",
        {
            "frames_processed": len(summary.samples),
            "total_damage": summary.total_damage,
            "mean_dps": summary.mean_dps,
            "final_smooth_dps": summary.samples[-1].smoothed_dps if summary.samples else 0.0,
        },
    )
    return summary


@app.command("run")
def cli_run(
    video: Path = typer.Option(..., exists=True, readable=True, help="Input video file"),
    config: Path = typer.Option(
        Path("configs/sample_config.yaml"),
        exists=True,
        readable=True,
        help="YAML config file",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Override output directory (CSV/JSON/debug frames)",
    ),
    debug: bool = typer.Option(False, help="Force writing debug frames"),
) -> None:
    """Execute the baseline DPS pipeline."""

    run_pipeline(video, config, output_dir=output_dir, force_debug=debug)


if __name__ == "__main__":
    app()
