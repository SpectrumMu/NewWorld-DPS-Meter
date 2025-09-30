from pathlib import Path

import pytest

from vision_dps.config import load_config


@pytest.fixture()
def sample_config_path() -> Path:
    return Path("configs/sample_config.yaml")


def test_sample_config_loads(sample_config_path: Path) -> None:
    cfg = load_config(sample_config_path)

    assert cfg.roi.width > 0
    assert cfg.roi.height > 0
    assert cfg.video.frame_stride >= 1
    assert cfg.metrics.decay_halflife_seconds > 0


def test_intensity_scale_positive(sample_config_path: Path) -> None:
    cfg = load_config(sample_config_path)

    assert cfg.metrics.intensity_to_damage_scale >= 0


def test_output_paths_are_configurable(sample_config_path: Path, tmp_path: Path) -> None:
    cfg = load_config(sample_config_path)

    cfg.output.csv_path = tmp_path / "run.csv"
    cfg.output.json_path = tmp_path / "run.json"

    assert cfg.output.csv_path.parent == tmp_path
    assert cfg.output.json_path.parent == tmp_path
