# Vision-Based DPS Meter (Baseline)

- 这个项目是本人Vibe Coding出来的石山，基于OCR的视觉DPS计量器基线实现，适用于《New World: Aeternum》
- 运行本项目需要有一定的python基础，包管理使用`uv`，用于适配各种操作系统
- 项目处于一个非常原始的阶段
   - 目前只能识别频率较低的伤害数字
   - 项目只支持离线识别，即需要先录制视频再处理，由于实时识别需要额外的性能优化和屏幕捕获支持，暂未实现
   - 具体运行步骤请参考下文

- 测试
   - Sch竞速视频坦克打BOSS：标定伤害39622，识别伤害36682
   - 西洋剑打稻草人：标定伤害345930，识别伤害83258


This project provides a starting point for building a vision-based damage-per-second (DPS) meter using computer vision. It currently focuses on parsing a single combat recording and extracting basic DPS metrics by detecting damage numbers within each frame.

## Project layout

```
.
├── sample-video.mov          # Example combat recording to experiment with
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── src/
    └── vision_dps/
        ├── __init__.py
        ├── config.py         # Centralised configuration for ROI/thresholds
        ├── pipeline.py       # High level orchestration of the DPS pipeline
        ├── preprocessing.py  # Frame loading & pre-processing utilities
        ├── detectors.py      # Damage number candidate detection
        └── metrics.py        # DPS aggregation helpers
```

## Getting started

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip wheel
   pip install -r requirements.txt
   ```

2. **Run the baseline pipeline**
   ```bash
   python -m vision_dps.pipeline --video sample-video.mov --config configs/sample_config.yaml --output-dir reports
   ```

   The baseline configuration focuses on simple colour/contrast based detection inside a region-of-interest (ROI). You can duplicate and tweak the YAML config to match new games or different capture setups.

3. **Inspect the results**
   The command outputs both a JSON report and a short CSV file in `reports/`. The JSON summarises the run, while the CSV lists per-frame or per-event damage estimates.

## uv workflow

Set up an isolated environment and install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

The CLI can then be invoked through uv to ensure you pick up the virtual environment:

```bash
uv run python -m vision_dps.pipeline --video sample-video.mov --config configs/sample_config.yaml --output-dir reports
```

## Testing

The repository ships with a small pytest suite to sanity-check configuration loading. After installing dependencies run:

```bash
uv run pytest
```


## Next steps

- Replace the colour threshold detector with a lightweight OCR model (e.g. `easyocr`, `tesseract`).
- Add UI overlays that highlight detected numbers to visually verify the detections.
- Incorporate temporal smoothing and crit detection to produce more reliable DPS metrics.
- Extend the pipeline into a real-time screen capture mode using `mss` or `d3dshot` for PC games.

## Troubleshooting

- **Dependencies fail to install**: Ensure you are running Python 3.10+ with pip upgraded to the latest version.
- **No detections**: Adjust the HSV ranges, ROI coordinates, and minimum contour area in the YAML config. Use the debug visualisation option (`--show-debug`) to see what the pipeline is looking at.


## Configuration primer

- **ROI**: The bounding box (in the resized frame) that contains the floating damage numbers. Tune this first.
- **HSV thresholds**: `detector.hsv_lower` and `detector.hsv_upper` are used to isolate bright/white numbers. Adjust to match your game's colour scheme.
- **Adaptive threshold**: Helps separate glyphs from the background after contrast enhancement.
- **Intensity scale**: `metrics.intensity_to_damage_scale` converts the aggregate contour intensity into a damage value. Calibrate it by comparing the CSV output against the on-screen damage log.
- **Halflife**: Controls how quickly the smoothed DPS reacts to spikes. Lower values respond faster but are noisier.

Enable debug frame dumps with `--debug` or `output.save_debug_frames: true` to inspect detections frame-by-frame.
