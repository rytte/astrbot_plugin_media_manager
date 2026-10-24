from __future__ import annotations
import asyncio
from unittest.mock import patch
import pytest
from astrbot_plugin_media_manager.group_image_cache import GroupImageCache
from test_context_images import save_image


@pytest.mark.asyncio
async def test_cache_deduplicates_bytes_but_isolates_platform_group_and_conversation(
    tmp_path,
):
    cache = GroupImageCache(tmp_path / "cache", 256, 24, 10)
    path = tmp_path / "image.png"
    save_image(path, "PNG", (3, 3))
    first = await cache.put(str(path), "p", "group-a", "c1")
    again = await cache.put(str(path), "p", "group-a", "c1")
    other = await cache.put(str(path), "p", "group-b", "c1")
    assert first == again
    assert other["image_ref"] != first["image_ref"]
    assert len(list(cache.root.glob("blob_*"))) == 1
    for platform, origin, cid in [
        ("q", "group-a", "c1"),
        ("p", "group-b", "c1"),
        ("p", "group-a", "c2"),
    ]:
        assert await cache.read(first["image_ref"], platform, origin, cid) is None
    await cache.bind([first["image_ref"]], "p", "group-a", "c2")
    assert await cache.read(first["image_ref"], "p", "group-a", "c2") is None
    restored = GroupImageCache(cache.root, 256, 24, 10)
    assert await restored.read(first["image_ref"], "p", "group-a", "c1") is not None


@pytest.mark.asyncio
async def test_cache_expires_after_24_hours_even_when_inspected(tmp_path):
    cache = GroupImageCache(tmp_path / "cache", 256, 24, 10)
    path = tmp_path / "image.png"
    save_image(path, "PNG", (3, 3))
    with patch(
        "astrbot_plugin_media_manager.group_image_cache.time.time", return_value=100
    ):
        record = await cache.put(str(path), "p", "g", "c")
    with patch(
        "astrbot_plugin_media_manager.group_image_cache.time.time", return_value=86499
    ):
        assert await cache.read(record["image_ref"], "p", "g", "c") is not None
    with patch(
        "astrbot_plugin_media_manager.group_image_cache.time.time", return_value=86500
    ):
        assert await cache.read(record["image_ref"], "p", "g", "c") is None
    assert not list(cache.root.glob("blob_*"))


@pytest.mark.asyncio
async def test_shared_file_survives_until_all_references_expire(tmp_path):
    cache = GroupImageCache(tmp_path / "cache", 256, 24, 10)
    path = tmp_path / "image.png"
    save_image(path, "PNG", (3, 3))
    with patch(
        "astrbot_plugin_media_manager.group_image_cache.time.time", return_value=100
    ):
        first = await cache.put(str(path), "p", "g1", "c")
    with patch(
        "astrbot_plugin_media_manager.group_image_cache.time.time", return_value=200
    ):
        second = await cache.put(str(path), "p", "g2", "c")
    with patch(
        "astrbot_plugin_media_manager.group_image_cache.time.time", return_value=86500
    ):
        assert await cache.read(first["image_ref"], "p", "g1", "c") is None
        assert await cache.read(second["image_ref"], "p", "g2", "c") is not None
        assert len(list(cache.root.glob("blob_*"))) == 1
    with patch(
        "astrbot_plugin_media_manager.group_image_cache.time.time", return_value=86600
    ):
        await cache.cleanup()
        assert not list(cache.root.glob("blob_*"))


@pytest.mark.asyncio
async def test_oversized_or_invalid_image_does_not_evict_existing_cache(tmp_path):
    cache = GroupImageCache(tmp_path / "cache", 256, 24, 10)
    path = tmp_path / "image.png"
    payload = save_image(path, "PNG", (3, 3))
    record = await cache.put(str(path), "p", "g", "c")
    cache.max_bytes = len(payload)
    bad_path = tmp_path / "bad.bin"
    for contents in (b"not an image", payload + b"larger"):
        bad_path.write_bytes(contents)
        with pytest.raises((OSError, ValueError)):
            await cache.put(str(bad_path), "p", "g", "c")
        assert await cache.read(record["image_ref"], "p", "g", "c") is not None


@pytest.mark.asyncio
async def test_restart_removes_unindexed_files_before_capacity_check(tmp_path):
    cache = GroupImageCache(tmp_path / "cache", 256, 24, 10)
    orphan = cache.root / "blob_interrupted.partial"
    orphan.write_bytes(b"interrupted write")
    path = tmp_path / "image.png"
    payload = save_image(path, "PNG", (3, 3))
    cache.max_bytes = len(payload)
    await cache.put(str(path), "p", "g", "c")
    assert not orphan.exists()
    assert sum(item.stat().st_size for item in cache.root.glob("blob_*")) == len(
        payload
    )


@pytest.mark.asyncio
async def test_global_capacity_evicts_oldest_unique_image(tmp_path):
    cache = GroupImageCache(tmp_path / "cache", 256, 24, 10)
    first_path, second_path = tmp_path / "first.png", tmp_path / "second.png"
    first_bytes = save_image(first_path, "PNG", (3, 3))
    second_bytes = save_image(second_path, "PNG", (4, 4))
    cache.max_bytes = max(len(first_bytes), len(second_bytes))
    first = await cache.put(str(first_path), "p", "g1", "c")
    second = await cache.put(str(second_path), "p", "g2", "c")
    assert await cache.read(first["image_ref"], "p", "g1", "c") is None
    assert await cache.read(second["image_ref"], "p", "g2", "c") is not None
    assert (
        sum(path.stat().st_size for path in cache.root.glob("blob_*"))
        <= cache.max_bytes
    )


@pytest.mark.asyncio
async def test_parallel_duplicate_captures_share_file_and_reference(tmp_path):
    cache = GroupImageCache(tmp_path / "cache", 256, 24, 10)
    path = tmp_path / "image.png"
    save_image(path, "PNG", (3, 3))
    records = await asyncio.gather(
        *(cache.put(str(path), "p", "g", "c") for _ in range(8))
    )
    assert len({record["image_ref"] for record in records}) == 1
    assert len(list(cache.root.glob("blob_*"))) == 1
