from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from astrbot_plugin_media_manager.context_images import ContextImageManager
from astrbot_plugin_media_manager.group_image_cache import GroupImageCache
from astrbot_plugin_media_manager.migration import import_legacy_images, legacy_settings
from astrbot_plugin_media_manager.storage import Storage
from test_context_images import Event, build_manager, save_image


async def legacy_manager(tmp_path):
    root = tmp_path / "legacy"
    storage = Storage(root / "qq_enhance.sqlite3")
    await storage.initialize()
    manager = ContextImageManager(
        SimpleNamespace(),
        storage,
        root / "context_images",
        3,
        30,
        64,
        10,
        GroupImageCache(root / "group_image_cache", 256, 24, 10),
    )
    return root, manager


@pytest.mark.asyncio
async def test_import_preserves_ids_scope_lifetime_paths_and_source(tmp_path):
    legacy, old = await legacy_manager(tmp_path)
    path = tmp_path / "source.png"
    payload = save_image(path, "PNG", (9, 7))
    event = Event(path)
    received = int(time.time()) - 3600
    with patch(
        "astrbot_plugin_media_manager.context_images.time.time", return_value=received
    ):
        records = await old._archive_sources(event, "c1", [str(path)])
    first = records[str(path)]
    with patch(
        "astrbot_plugin_media_manager.group_image_cache.time.time",
        return_value=received,
    ):
        group = await old.group_cache.put(str(path), "p", "group", "")
    # Exercise real pre-animation-metadata source schemas, not the current schema.
    with sqlite3.connect(old.storage.database_path) as db:
        db.execute("ALTER TABLE context_images DROP COLUMN is_animated")
    with sqlite3.connect(old.group_cache.root / "index.sqlite3") as db:
        db.execute("ALTER TABLE images DROP COLUMN is_animated")
    manager, storage, _ = await build_manager(tmp_path / "new")
    counts = await import_legacy_images(
        storage, manager.root, manager.group_cache, legacy
    )
    assert counts == {"images": 1, "group_images": 1, "missing": 0}
    record = (await storage.list_context_images())[0]
    assert record["image_ref"] == first["image_ref"]
    assert record["is_animated"] == 0
    assert record["created_at"] == received
    assert record["conversation_id"] == "c1"
    assert Path(record["path"]).parent == manager.root
    assert Path(record["path"]).read_bytes() == payload
    assert Path(first["path"]).read_bytes() == payload
    assert (
        await storage.resolve_context_image(
            first["image_ref"], event.platform_id, event.unified_msg_origin, "other"
        )
        is None
    )
    await manager.group_cache.bind([group["image_ref"]], "p", "group", "c1")
    assert await manager.group_cache.read(group["image_ref"], "p", "group", "c1") == (
        payload,
        "image/png",
    )
    with sqlite3.connect(manager.group_cache.root / "index.sqlite3") as db:
        expiry = db.execute("SELECT expires_at FROM images").fetchone()[0]
        assert db.execute("SELECT is_animated FROM images").fetchone()[0] == 0
    assert expiry == received + 86400
    # A completed import must not resurrect data later evicted by the new owner.
    with sqlite3.connect(storage.database_path) as db:
        db.execute("DELETE FROM context_images")
    assert not any(
        (
            await import_legacy_images(
                storage, manager.root, manager.group_cache, legacy
            )
        ).values()
    )
    assert await storage.list_context_images() == []


@pytest.mark.asyncio
async def test_failed_import_retries_without_regenerating_refs(tmp_path, monkeypatch):
    legacy, old = await legacy_manager(tmp_path)
    path = tmp_path / "source.png"
    save_image(path, "PNG", (3, 4))
    first = (await old._archive_sources(Event(path), "c1", [str(path)]))[str(path)]
    manager, storage, _ = await build_manager(tmp_path / "new")
    with monkeypatch.context() as patcher:

        def fail_copy(*_args):
            raise OSError("copy interrupted")

        patcher.setattr("astrbot_plugin_media_manager.migration._copy", fail_copy)
        with pytest.raises(OSError, match="interrupted"):
            await import_legacy_images(
                storage, manager.root, manager.group_cache, legacy
            )
    with sqlite3.connect(storage.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM migrations").fetchone()[0] == 0
    await import_legacy_images(storage, manager.root, manager.group_cache, legacy)
    assert (await storage.list_context_images())[0]["image_ref"] == first["image_ref"]


@pytest.mark.asyncio
async def test_import_rejects_index_path_outside_old_root(tmp_path):
    legacy, old = await legacy_manager(tmp_path)
    path = tmp_path / "outside.png"
    save_image(path, "PNG", (3, 4))
    await old._archive_sources(Event(path), "c1", [str(path)])
    with sqlite3.connect(old.storage.database_path) as db:
        db.execute("UPDATE context_images SET path=?", (str(path),))
    manager, storage, _ = await build_manager(tmp_path / "new")
    with pytest.raises(ValueError, match="目录"):
        await import_legacy_images(storage, manager.root, manager.group_cache, legacy)
    assert await storage.list_context_images() == []


def test_legacy_settings_ignore_removed_options_and_preserve_limits_and_views(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "context_images": {"enabled": False, "retention_days": 0},
                "files": {"max_base64_size_mb": 17},
                "limits": {"max_components": 5},
                "platform": {"platform_id": "qq-a"},
                "toolsets": {"disabled_operations": ["qq_media.inspect_animation"]},
            }
        ),
        encoding="utf-8",
    )
    assert legacy_settings(path) == {
        "retention_days": 0,
        "max_inspect_mb": 17,
        "platform_ids": ["qq-a"],
        "disabled_views": ["animation"],
    }


@pytest.mark.asyncio
async def test_missing_legacy_files_do_not_produce_live_references(tmp_path):
    legacy, old = await legacy_manager(tmp_path)
    path = tmp_path / "source.png"
    save_image(path, "PNG", (3, 4))
    record = (await old._archive_sources(Event(path), "c1", [str(path)]))[str(path)]
    Path(record["path"]).unlink()
    manager, storage, _ = await build_manager(tmp_path / "new")
    counts = await import_legacy_images(
        storage, manager.root, manager.group_cache, legacy
    )
    assert counts == {"images": 0, "group_images": 0, "missing": 1}
    assert await storage.list_context_images() == []
    assert old.storage.database_path.is_file()


@pytest.mark.asyncio
async def test_absent_legacy_source_can_be_discovered_on_a_later_reload(tmp_path):
    manager, storage, _ = await build_manager(tmp_path / "new")
    await import_legacy_images(
        storage, manager.root, manager.group_cache, tmp_path / "legacy"
    )
    with sqlite3.connect(storage.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM migrations").fetchone()[0] == 0
    legacy, old = await legacy_manager(tmp_path)
    path = tmp_path / "source.png"
    save_image(path, "PNG", (3, 4))
    await old._archive_sources(Event(path), "c1", [str(path)])
    result = await import_legacy_images(
        storage, manager.root, manager.group_cache, legacy
    )
    assert result["images"] == 1
