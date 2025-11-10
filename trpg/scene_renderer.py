"""Scene rendering helpers that combine ASCII art with generated images."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - tests exercise fallback path
    Image = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class SceneImage:
    """Container storing an encoded scene image."""

    mime_type: str
    data: str  # Base64 encoded image bytes


@dataclass
class SceneSnapshot:
    """Rich representation of the current scene."""

    ascii_art: str
    prompt: str
    image: Optional[SceneImage] = None

    def to_payload(self) -> dict:
        payload = {"ascii_art": self.ascii_art, "prompt": self.prompt}
        if self.image:
            payload["image"] = {
                "mime_type": self.image.mime_type,
                "data": self.image.data,
            }
        return payload


class ImageBackend(Protocol):
    """Protocol for backends capable of generating a scene image."""

    def generate(self, prompt: str) -> SceneImage:  # pragma: no cover - interface definition
        ...


class TinyMLXImageBackend:
    """Image backend that favours MLX powered, lightweight image generators.

    The implementation deliberately keeps the "model" tiny so that it can run in
    constrained environments while still exercising MLX primitives. When the
    real ``mlx`` package is available a deterministic procedural texture is
    computed using MLX array operations. The resulting image reacts to the
    textual prompt ensuring that different story beats produce distinct
    visuals.
    """

    def __init__(self, *, image_size: int = 256) -> None:
        self.image_size = image_size

    @staticmethod
    def _load_mlx():
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("MLX is not installed. Install `mlx` to enable MLX rendering.") from exc
        return mx

    def generate(self, prompt: str) -> SceneImage:
        mx = self._load_mlx()
        # Hash the prompt to derive deterministic but varied seeds.
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        seed_scalar = int.from_bytes(digest[:4], "little")
        frequency = 0.015 + (seed_scalar % 500) / 100_000.0
        colour_shift = (seed_scalar % 255) / 255.0

        # Build a coordinate grid using MLX primitives.
        side = self.image_size
        grid = mx.arange(side * side * 3, dtype=mx.float32)
        grid = mx.reshape(grid, (side, side, 3))
        pattern = mx.sin(grid * frequency + colour_shift)
        pattern = (pattern + 1.0) * 0.5  # normalise to 0..1

        try:
            numpy_pattern = pattern.asnumpy()
        except AttributeError:  # pragma: no cover - defensive fallback for older MLX
            import numpy as np

            numpy_pattern = np.array(pattern)

        image_bytes = _encode_png(numpy_pattern)
        return SceneImage(mime_type="image/png", data=image_bytes)


class ProceduralImageBackend:
    """Fallback backend that uses NumPy to approximate the MLX result."""

    def __init__(self, *, image_size: int = 256) -> None:
        self.image_size = image_size

    def generate(self, prompt: str) -> SceneImage:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("NumPy is required for procedural rendering.") from exc

        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[4:12], "little"))

        base = np.linspace(0, 1, self.image_size, dtype=np.float32)
        x_grad = np.tile(base, (self.image_size, 1))
        y_grad = np.tile(base.reshape(-1, 1), (1, self.image_size))

        texture = np.stack(
            [
                (x_grad + digest[0] / 255.0) % 1.0,
                (y_grad + digest[1] / 255.0) % 1.0,
                rng.random((self.image_size, self.image_size), dtype=np.float32),
            ],
            axis=-1,
        )
        image_bytes = _encode_png(texture)
        return SceneImage(mime_type="image/png", data=image_bytes)


class SceneRenderer:
    """Combine ASCII rendering with optional MLX powered images."""

    def __init__(
        self,
        *,
        backends: Optional[Sequence[ImageBackend]] = None,
    ) -> None:
        if backends is None:
            backends = (TinyMLXImageBackend(), ProceduralImageBackend())
        self._backends = list(backends)

    @staticmethod
    def _build_prompt(facts: Sequence[str]) -> str:
        interesting_facts: List[str] = []
        for fact in facts[-6:]:
            trimmed = fact.split(":", 1)[-1].strip()
            if trimmed:
                interesting_facts.append(trimmed)
        if not interesting_facts:
            return "Illustrate a calm tabletop RPG scene with soft lighting."
        joined = "; ".join(interesting_facts)
        return (
            "Illustrate a dynamic tabletop RPG scene capturing: "
            f"{joined}. Use cinematic lighting and coherent character designs."
        )

    def render(self, ascii_art: str, facts: Sequence[str]) -> SceneSnapshot:
        prompt = self._build_prompt(facts)
        image: Optional[SceneImage] = None
        for backend in self._backends:
            try:
                image = backend.generate(prompt)
                break
            except RuntimeError as exc:
                logger.debug("Scene image backend %s failed: %s", backend.__class__.__name__, exc)
                continue
        return SceneSnapshot(ascii_art=ascii_art, prompt=prompt, image=image)


def _encode_png(array: Iterable[Iterable[Iterable[float]]]) -> str:
    """Encode a float array as a PNG image and return a base64 string."""

    if Image is None:
        raise RuntimeError("Pillow is required to encode PNG images.")

    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("NumPy is required to encode images.") from exc

    np_array = np.array(array, dtype=np.float32)
    np_array = np.clip(np_array, 0.0, 1.0)
    np_array = (np_array * 255).astype(np.uint8)

    with io.BytesIO() as buffer:
        image = Image.fromarray(np_array, mode="RGB")
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded
