from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import (
    ImageURLPart,
    Message,
    dump_messages_with_checkpoints,
)
from astrbot_plugin_media_manager.group_image_cache import GroupImageCache
from astrbot_plugin_media_manager.image_metadata import detect_animation
from astrbot_plugin_media_manager.storage import Storage
from test_animation_preview import animation_bytes
from test_context_images import Event, build_manager, save_image


@pytest.mark.parametrize("as_path", [False, True])
@pytest.mark.parametrize(
    "image_format,count,poster,expected",
    [
        ("GIF", 1, False, False),
        ("GIF", 2, False, True),
        ("WEBP", 1, False, False),
        ("WEBP", 2, False, True),
        ("PNG", 1, False, False),
        ("PNG", 2, False, True),
        ("PNG", 2, True, False),  # A poster and one animation frame are static.
        ("PNG", 3, True, True),
        ("TIFF", 2, False, False),  # Pages are not animation frames.
        ("JPEG", 1, False, False),
    ],
)
def test_detection_uses_actual_animation_frames(
    tmp_path, as_path, image_format, count, poster, expected
):
    if image_format == "JPEG":
        stream = io.BytesIO()
        Image.new("RGB", (3, 4)).save(stream, format="JPEG")
        payload = stream.getvalue()
    else:
        payload = animation_bytes(image_format=image_format, count=count, poster=poster)
    source = payload
    if as_path:
        source = tmp_path / "misleading-extension.jpg"
        source.write_bytes(payload)
    assert detect_animation(source) is expected


def test_detection_failures_remain_unknown(tmp_path):
    assert detect_animation(b"not an image") is None
    assert detect_animation(tmp_path / "missing.png") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_format,count,poster,expected",
    [
        ("GIF", 1, False, False),
        ("GIF", 2, False, True),
        ("WEBP", 2, False, True),
        ("PNG", 2, True, False),
        ("PNG", 3, True, True),
    ],
)
async def test_archive_and_cache_persist_kind_and_advertise_animation_only_when_confirmed(
    tmp_path,
    image_format,
    count,
    poster,
    expected,
):
    manager, storage, _ = await build_manager(tmp_path)
    path = tmp_path / "original.bin"
    payload = animation_bytes(image_format=image_format, count=count, poster=poster)
    path.write_bytes(payload)
    event = Event(path)
    request = ProviderRequest(
        conversation=SimpleNamespace(cid="c1"), image_urls=[str(path)]
    )
    await manager.prepare_request(event, request)
    archived = (await storage.list_context_images())[0]
    assert archived["is_animated"] == expected
    assert Path(archived["path"]).read_bytes() == payload
    cached = await manager.group_cache.put(
        str(path), event.platform_id, event.unified_msg_origin, "c1"
    )
    assert cached["is_animated"] is expected
    with sqlite3.connect(manager.group_cache.root / "index.sqlite3") as db:
        assert db.execute("SELECT is_animated FROM images").fetchone()[0] == expected
    for record in (archived, cached):
        text = manager._reference_text(record)
        assert ("；动图；" if expected else "；静图；") in text
        assert 'view="original"' in text
        assert ('view="animation"' in text) == expected
    current = Message.model_validate(await request.assemble_context())
    manager.bind_current_request_images(event, SimpleNamespace(messages=[current]))
    assert any(
        isinstance(part, ImageURLPart) and not part._no_save for part in current.content
    )
    await manager.finalize_images(event, SimpleNamespace(messages=[current]))
    history = str(dump_messages_with_checkpoints([current]))
    assert "data:image" not in history
    assert ("；动图；" if expected else "；静图；") in history
    manager.disabled_views = {"animation"}
    assert 'view="animation"' not in manager._reference_text(archived)


@pytest.mark.asyncio
async def test_unknown_detection_does_not_block_archival_or_advertise_animation(
    tmp_path,
):
    manager, storage, _ = await build_manager(tmp_path)
    path = tmp_path / "image.png"
    save_image(path, "PNG", (3, 4))
    with (
        patch(
            "astrbot_plugin_media_manager.context_images.detect_animation",
            return_value=None,
        ),
        patch(
            "astrbot_plugin_media_manager.group_image_cache.detect_animation",
            return_value=None,
        ),
    ):
        event = Event(path)
        request = ProviderRequest(
            conversation=SimpleNamespace(cid="c1"), image_urls=[str(path)]
        )
        await manager.prepare_request(event, request)
        archived = (await storage.list_context_images())[0]
        cached = await manager.group_cache.put(str(path), "p", "g", "c")
    for record in (archived, cached):
        assert record["is_animated"] is None
        text = manager._reference_text(record)
        assert "；动静态未知；" in text
        assert 'view="original"' in text
        assert 'view="animation"' not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("animated", [False, True])
async def test_legacy_archive_schema_is_upgraded_and_backfilled_without_changing_records(
    tmp_path, animated
):
    manager, storage, _ = await build_manager(tmp_path)
    path = tmp_path / "image.gif"
    path.write_bytes(animation_bytes(count=2 if animated else 1))
    before = (await manager._archive_sources(Event(path), "c1", [str(path)]))[str(path)]
    with sqlite3.connect(storage.database_path) as db:
        db.execute("ALTER TABLE context_images DROP COLUMN is_animated")
    restored = Storage(storage.database_path)
    await restored.initialize()
    await restored.backfill_image_animation(manager.root)
    after = (await restored.list_context_images())[0]
    assert after["is_animated"] == animated
    for key in (
        "image_ref",
        "platform_id",
        "unified_msg_origin",
        "conversation_id",
        "created_at",
        "last_accessed_at",
    ):
        assert after[key] == before[key]
    with patch("astrbot_plugin_media_manager.storage.detect_animation") as detect:
        await restored.backfill_image_animation(manager.root)
        detect.assert_not_called()  # Known metadata is reused after reload.


@pytest.mark.asyncio
async def test_backfill_does_not_probe_paths_outside_archive_root(tmp_path):
    manager, storage, _ = await build_manager(tmp_path)
    path = tmp_path / "outside.png"
    save_image(path, "PNG", (3, 4))
    await manager._archive_sources(Event(path), "c1", [str(path)])
    with sqlite3.connect(storage.database_path) as db:
        db.execute("UPDATE context_images SET path=?, is_animated=NULL", (str(path),))
    with patch("astrbot_plugin_media_manager.storage.detect_animation") as detect:
        await storage.backfill_image_animation(manager.root)
        detect.assert_not_called()
    assert (await storage.list_context_images())[0]["is_animated"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("animated", [False, True])
async def test_legacy_group_cache_schema_is_upgraded_without_renewing_expiry(
    tmp_path, animated
):
    cache = GroupImageCache(tmp_path / "cache", 256, 24, 10)
    path = tmp_path / "image.gif"
    path.write_bytes(animation_bytes(count=2 if animated else 1))
    record = await cache.put(str(path), "p", "g", "c")
    with sqlite3.connect(cache.root / "index.sqlite3") as db:
        before = db.execute("SELECT received_at, expires_at FROM images").fetchone()
        db.execute("ALTER TABLE images DROP COLUMN is_animated")
    restored = GroupImageCache(cache.root, 256, 24, 10)
    assert await restored.read(record["image_ref"], "p", "g", "c") == (
        path.read_bytes(),
        "image/gif",
    )
    with sqlite3.connect(cache.root / "index.sqlite3") as db:
        after = db.execute(
            "SELECT received_at, expires_at, is_animated FROM images"
        ).fetchone()
    assert after == (*before, animated)
