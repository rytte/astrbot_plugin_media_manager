"""Detect animation from actual frames without decoding the entire animation."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from astrbot.api import logger


def detect_animation(source: str | Path | bytes) -> bool | None:
    """Return True/False for confirmed animation/static images, None on failure.

    Multipage documents are not animations. An APNG poster does not count as
    an animation frame, so a second actual animation frame must be present.
    """
    try:
        with Image.open(
            io.BytesIO(source) if isinstance(source, bytes) else source
        ) as image:
            if image.format not in {"GIF", "WEBP", "PNG"}:
                return False
            try:
                image.seek(2 if image.info.get("default_image") else 1)
            except EOFError:
                return False
            return True
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.warning("Media 管理无法确认图片动静态：%s", type(exc).__name__)
        return None
