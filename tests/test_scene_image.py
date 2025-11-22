"""MLX Stable Diffusion scene renderer 호환성 테스트."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from trpg import scene_image


def test_build_command_variants_prioritize_positional_prompt() -> None:
    renderer = scene_image.MLXStableDiffusionSceneRenderer(
        command="python -m mlx_examples.stable_diffusion.txt2image",
        model="sdxl",
        negative_prompt="blurry",
        steps=20,
        guidance_scale=6.5,
    )

    variants = renderer._build_command_variants(Path("/tmp/out.png"), "hello world")

    assert variants, "커맨드 변형이 최소 하나는 생성되어야 합니다."
    first = variants[0]

    assert first[-1] == "hello world"
    assert "--prompt" not in first  # 위치 인자를 우선 사용
    assert "--cfg" in first  # guidance 파라미터는 --cfg로 전달
    assert "--guidance-scale" not in first
    assert "--model" in first
    assert "--negative_prompt" in first


def test_run_mlx_retries_with_prompt_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = scene_image.MLXStableDiffusionSceneRenderer(
        command="python -m mlx_examples.stable_diffusion.txt2image",
        variations=1,
    )

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text):  # type: ignore[override]
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                returncode=2,
                cmd=cmd,
                output="",
                stderr="error: unrecognized arguments: --cfg",
            )

        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.write_bytes(b"data")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scene_image.subprocess, "run", fake_run)

    data_urls = renderer._run_mlx("a scenic view")

    assert data_urls[0].startswith("data:image/png;base64,")
    assert len(calls) == 2  # 첫 번째 시도 실패 후 재시도
    assert any("--prompt" in flag for flag in calls[1])


def test_run_mlx_collects_multiple_variations(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = scene_image.MLXStableDiffusionSceneRenderer(
        command="python -m mlx_examples.stable_diffusion.txt2image",
        variations=3,
    )

    def fake_run(cmd, check, capture_output, text):  # type: ignore[override]
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.write_bytes(b"data")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scene_image.subprocess, "run", fake_run)

    data_urls = renderer._run_mlx("a scenic view")

    assert len(data_urls) == 3
    assert all(url.startswith("data:image/png;base64,") for url in data_urls)
