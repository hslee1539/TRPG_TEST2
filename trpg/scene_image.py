"""Scene image generation helpers powered by MLX Stable Diffusion.

이 모듈은 MLX 기반 Stable Diffusion 실행기를 래핑해 TRPG 장면을
시각화합니다. MLX 관련 패키지가 설치되지 않았거나 실행 환경이 준비되지
않아도 안전하게 실패하도록 설계되어 있습니다.
"""

from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_MLX_SD_COMMAND = f"{sys.executable} -m mlx_examples.stable_diffusion.generate"
DEFAULT_MLX_SD_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


@dataclass
class SceneImageResult:
    """결과 이미지 데이터와 사용한 프롬프트 정보를 담습니다."""

    prompt: str
    data_url: Optional[str]
    error: Optional[str] = None


class MLXStableDiffusionSceneRenderer:
    """MLX Stable Diffusion을 호출해 장면 이미지를 생성합니다.

    실제 MLX 실행 파일(예: ``python -m mlx_examples.stable_diffusion.generate``)
    을 서브프로세스로 실행하는 구조이므로, MLX 패키지가 설치되어 있지
    않으면 자동으로 우회하거나 오류 메시지를 반환합니다.
    """

    def __init__(
        self,
        *,
        command: Optional[str] = None,
        model: str = DEFAULT_MLX_SD_MODEL,
        negative_prompt: Optional[str] = None,
        steps: int = 28,
        guidance_scale: float = 7.5,
        enabled: bool = True,
    ) -> None:
        self.command = command or os.getenv("TRPG_MLX_SD_COMMAND", DEFAULT_MLX_SD_COMMAND)
        self.model = os.getenv("TRPG_MLX_SD_MODEL", model)
        self.negative_prompt = os.getenv("TRPG_MLX_SD_NEGATIVE_PROMPT", negative_prompt or "")
        self.steps = int(os.getenv("TRPG_MLX_SD_STEPS", str(steps)))
        self.guidance_scale = float(
            os.getenv("TRPG_MLX_SD_GUIDANCE", str(guidance_scale))
        )
        self.enabled = enabled

    def render(self, facts: Sequence[str]) -> Optional[SceneImageResult]:
        """현재까지의 사실을 기반으로 Stable Diffusion 프롬프트를 만들고 실행합니다."""

        if not self.enabled:
            return None

        prompt = self._build_prompt(facts)
        try:
            data_url = self._run_mlx(prompt)
        except Exception as exc:  # pragma: no cover - 방어적 코드 경로
            message = f"MLX Stable Diffusion 실행 오류: {exc}"
            return SceneImageResult(prompt=prompt, data_url=None, error=message)

        return SceneImageResult(prompt=prompt, data_url=data_url)

    def _build_prompt(self, facts: Sequence[str]) -> str:
        if not facts:
            return (
                "Soft cinematic key art of a brave adventurer exploring an unknown realm, "
                "moody lighting, dynamic composition"
            )

        description = " \n".join(facts)
        summary = textwrap.shorten(description, width=320, placeholder=" …")
        return (
            "Illustrate the current tabletop RPG scene in a painterly, detailed style. "
            "Show characters and environment faithfully. Narrative facts: "
            f"{summary}"
        )

    def _run_mlx(self, prompt: str) -> str:
        """MLX Stable Diffusion CLI를 실행하고 data URL 형태로 반환합니다."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scene.png"
            command = list(shlex.split(self.command))
            command.extend(["--prompt", prompt, "--output", str(output_path)])
            if self.model:
                command.extend(["--model", self.model])
            if self.negative_prompt:
                command.extend(["--negative-prompt", self.negative_prompt])
            command.extend(["--steps", str(self.steps), "--guidance-scale", str(self.guidance_scale)])

            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:  # pragma: no cover - 실행기 미설치 시
                raise RuntimeError(
                    "mlx Stable Diffusion 실행 파일을 찾을 수 없습니다. "
                    "MLX 예제를 설치했는지 확인하거나 TRPG_MLX_SD_COMMAND를 "
                    "사용해 실행 경로를 지정하세요."
                ) from exc
            except subprocess.CalledProcessError as exc:  # pragma: no cover - 실행 실패 시
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                raise RuntimeError(
                    "Stable Diffusion 실행에 실패했습니다: "
                    f"{stdout.strip()} {stderr.strip()}"
                ) from exc

            if not output_path.exists():  # pragma: no cover - 실행기 출력 누락 시
                raise RuntimeError("Stable Diffusion이 이미지를 생성하지 못했습니다.")

            image_bytes = output_path.read_bytes()
            encoded = base64.b64encode(image_bytes).decode("ascii")
            return f"data:image/png;base64,{encoded}"
