import pytest

from config import AppConfig, parse_args, validate_config


def test_validate_config_accepts_defaults():
    cfg = AppConfig(sofa="dummy.sofa")
    validate_config(cfg)


def test_validate_config_rejects_invalid_block():
    cfg = AppConfig(sofa="dummy.sofa", block=0)
    with pytest.raises(ValueError, match="--block"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_kinect_bridge_port():
    cfg = AppConfig(sofa="dummy.sofa", kinect_bridge_port=70000)
    with pytest.raises(ValueError, match="--kinect-bridge-port"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_cli_output():
    cfg = AppConfig(sofa="dummy.sofa", cli_output="bad")
    with pytest.raises(ValueError, match="--cli-output"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_display_provider():
    cfg = AppConfig(sofa="dummy.sofa", display_provider="bad")
    with pytest.raises(ValueError, match="--display-provider"):
        validate_config(cfg)


def test_validate_config_accepts_open3d_display_provider():
    cfg = AppConfig(sofa="dummy.sofa", display_provider="open3d")
    validate_config(cfg)


def test_validate_config_rejects_negative_display_hz():
    cfg = AppConfig(sofa="dummy.sofa", display_hz=-1.0)
    with pytest.raises(ValueError, match="--display-hz"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_direction_provider():
    cfg = AppConfig(sofa="dummy.sofa", direction_provider="bad")
    with pytest.raises(ValueError, match="--direction-provider"):
        validate_config(cfg)


def test_validate_config_rejects_negative_direction_deadband():
    cfg = AppConfig(sofa="dummy.sofa", direction_deadband_deg=-0.1)
    with pytest.raises(ValueError, match="--direction-deadband-deg"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_acoustic_frame_provider():
    cfg = AppConfig(sofa="dummy.sofa", acoustic_frame_provider="bad")
    with pytest.raises(ValueError, match="--acoustic-frame-provider"):
        validate_config(cfg)


def test_parse_args_reads_yaml_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "sofa: assets/test.sofa",
                "pose_provider: toycv",
                "display_provider: open3d",
                "display_hz: 60",
                "camera_mirror: false",
                "acoustic_frame_provider: flip-front",
            ]
        ),
        encoding="utf-8",
    )
    cfg = parse_args(["--config", str(cfg_path)])
    assert cfg.sofa == "assets/test.sofa"
    assert cfg.pose_provider == "toycv"
    assert cfg.display_provider == "open3d"
    assert cfg.display_hz == 60.0
    assert cfg.camera_mirror is False
    assert cfg.acoustic_frame_provider == "flip-front"


def test_parse_args_cli_overrides_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "sofa: assets/test.sofa",
                "display_provider: open3d",
                "display_hz: 60",
            ]
        ),
        encoding="utf-8",
    )
    cfg = parse_args(
        [
            "--config",
            str(cfg_path),
            "--display-provider",
            "tui",
            "--display-hz",
            "12",
        ]
    )
    assert cfg.display_provider == "tui"
    assert cfg.display_hz == 12.0


def test_parse_args_rejects_unknown_yaml_key(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "sofa: assets/test.sofa",
                "bad_key: 1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        parse_args(["--config", str(cfg_path)])
