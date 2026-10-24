from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import ImageContent, TextContent
from PIL import Image as PillowImage

from astrbot.api.provider import ProviderRequest
from astrbot_plugin_media_manager.context_images import (
    CONVERSATION_ID_EXTRA,
    ContextImageError,
)

from test_animation_preview import animation_bytes
from test_context_images import Event, build_manager


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["img", "gimg"])
@pytest.mark.parametrize("count", [2, 12])
async def test_animation_preview_keeps_original_and_scopes_both_reference_types(
    tmp_path, prefix, count
):
    manager, storage, _ = await build_manager(tmp_path)
    path = tmp_path / "original.gif"
    payload = animation_bytes(count=count)
    path.write_bytes(payload)
    event = Event(path)
    event.set_extra(CONVERSATION_ID_EXTRA, "c1")
    if prefix == "img":
        await manager.prepare_request(
            event,
            ProviderRequest(
                conversation=SimpleNamespace(cid="c1"), image_urls=[str(path)]
            ),
        )
        image_ref = (await storage.list_context_images())[0]["image_ref"]
    else:
        record = await manager.group_cache.put(
            str(path), event.platform_id, event.unified_msg_origin, "c1"
        )
        image_ref = record["image_ref"]
    result = await manager.inspect_animation(event, image_ref)
    assert len(result.content) == 2
    text, image = result.content
    assert isinstance(text, TextContent)
    assert isinstance(image, ImageContent)
    assert image.mimeType == "image/png"
    assert f"共 {count} 帧，采样 {min(9, count)} 帧" in text.text
    assert "从左到右、从上到下" in text.text
    assert "同一张动图" in text.text
    assert ("其余格留空" in text.text) == (count < 9)
    with PillowImage.open(io.BytesIO(base64.b64decode(image.data))) as sheet:
        assert sheet.n_frames == 1
        assert sheet.size == (1152, 1248)
    original = await manager.inspect(event, image_ref)
    assert base64.b64decode(original.content[1].data) == payload
    assert path.read_bytes() == payload
    for wrong in (
        Event(platform_id="platform-b"),
        Event(unified_msg_origin="platform-a:FriendMessage:other"),
        Event(),
    ):
        wrong.set_extra(
            CONVERSATION_ID_EXTRA,
            "c2"
            if wrong.platform_id == event.platform_id
            and wrong.unified_msg_origin == event.unified_msg_origin
            else "c1",
        )
        with pytest.raises(ContextImageError, match="不属于当前会话"):
            await manager.inspect_animation(wrong, image_ref)


@pytest.mark.asyncio
async def test_static_corrupt_and_expired_references_return_safe_errors(tmp_path):
    manager, storage, _ = await build_manager(tmp_path)
    path = tmp_path / "static.gif"
    path.write_bytes(animation_bytes(count=1))
    event = Event(path)
    await manager.prepare_request(
        event,
        ProviderRequest(conversation=SimpleNamespace(cid="c1"), image_urls=[str(path)]),
    )
    record = (await storage.list_context_images())[0]
    with pytest.raises(ContextImageError, match="inspect") as error:
        await manager.inspect_animation(event, record["image_ref"])
    assert error.value.code == "animation_preview_unavailable"
    stored_path = Path(record["path"])
    stored_path.write_bytes(b"corrupt")
    with pytest.raises(ContextImageError, match="无法解码"):
        await manager.inspect_animation(event, record["image_ref"])
    stored_path.unlink()
    with pytest.raises(ContextImageError, match="不可用"):
        await manager.inspect_animation(event, record["image_ref"])
