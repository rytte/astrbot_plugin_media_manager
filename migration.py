"""Idempotently copy legacy QQ images; never delete the source data."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from astrbot.api import logger

from .config import DEFAULT_CONFIG
from .image_metadata import detect_animation

MIGRATION_NAME = "qq_enhance_images_v1"


def legacy_settings(config_path: Path) -> dict:
    """Map legacy limits once, including restrictions on the local view tools."""
    if not config_path.is_file():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    result = {
        key: value
        for key, value in data.get("context_images", {}).items()
        if key in DEFAULT_CONFIG
    }
    for group, old, new in (
        ("files", "max_base64_size_mb", "max_inspect_mb"),
        ("files", "cleanup_interval_seconds", "cleanup_interval_seconds"),
        ("limits", "max_components", "max_components"),
    ):
        if old in data.get(group, {}):
            result[new] = data[group][old]
    platform_id = data.get("platform", {}).get("platform_id")
    if platform_id:
        result["platform_ids"] = [platform_id]
    tools = data.get("toolsets", {})
    packs = tools.get("enabled_packs", [])
    disabled = tools.get("disabled_operations", [])
    result["disabled_views"] = [
        view
        for view, operation in (
            ("original", "inspect"),
            ("animation", "inspect_animation"),
        )
        if (packs and "media" not in packs) or f"qq_media.{operation}" in disabled
    ]
    return result


def _snapshot(path):
    """Use SQLite backup so WAL data is included without copying a live DB file."""
    snapshot = sqlite3.connect(":memory:")
    try:
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10)
        ) as source:
            source.backup(snapshot)
        snapshot.row_factory = sqlite3.Row
        return snapshot
    except BaseException:
        snapshot.close()
        raise


def _copy(source: Path, destination: Path, root: Path):
    source = source.resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        raise ValueError(f"旧图片缺失或不在插件目录内：{source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"图片迁移目标冲突：{destination.name}")
        return
    staging = destination.with_name(destination.name + ".partial")
    try:
        shutil.copyfile(source, staging)
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)


async def import_legacy_images(storage, image_root, group_cache, legacy_root):
    """Commit a completion marker only after both indexes and all copies succeed."""
    await group_cache._run(lambda _db: None)

    def migrate():
        counts = {"images": 0, "group_images": 0, "missing": 0}
        with closing(sqlite3.connect(storage.database_path, timeout=10)) as target:
            if target.execute(
                "SELECT 1 FROM migrations WHERE name = ?", (MIGRATION_NAME,)
            ).fetchone():
                return counts
            legacy_db = legacy_root / "qq_enhance.sqlite3"
            if legacy_db.is_file():
                with closing(_snapshot(legacy_db)) as source:
                    has_table = source.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='context_images'"
                    ).fetchone()
                    if has_table:
                        for row in source.execute("SELECT * FROM context_images"):
                            if not re.fullmatch(r"img_[0-9a-f]{24}", row["image_ref"]):
                                raise ValueError("旧图片引用格式无效")
                            old_path = Path(row["path"])
                            if not old_path.resolve().is_relative_to(
                                (legacy_root / "context_images").resolve()
                            ):
                                raise ValueError("旧图片路径超出归档目录")
                            if not old_path.is_file():
                                counts["missing"] += 1
                                continue
                            path = image_root / old_path.name
                            if target.execute(
                                "SELECT 1 FROM context_images WHERE image_ref=?",
                                (row["image_ref"],),
                            ).fetchone():
                                continue
                            _copy(old_path, path, legacy_root / "context_images")
                            keys = (
                                "image_ref",
                                "platform_id",
                                "unified_msg_origin",
                                "conversation_id",
                                "mime_type",
                                "width",
                                "height",
                                "size_bytes",
                                "created_at",
                                "last_accessed_at",
                                "orphaned_at",
                            )
                            target.execute(
                                "INSERT INTO context_images ("
                                + ",".join(keys)
                                + ",path,is_animated) VALUES ("
                                + ",".join("?" for _ in range(13))
                                + ")",
                                tuple(row[key] for key in keys)
                                + (str(path), detect_animation(path)),
                            )
                            counts["images"] += 1
                target.commit()
            old_cache = legacy_root / "group_image_cache"
            if (old_cache / "index.sqlite3").is_file():
                with (
                    closing(_snapshot(old_cache / "index.sqlite3")) as source,
                    closing(
                        sqlite3.connect(group_cache.root / "index.sqlite3", timeout=10)
                    ) as cache,
                ):
                    has_table = source.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='images'"
                    ).fetchone()
                    if has_table:
                        with cache:
                            for row in source.execute("SELECT * FROM images"):
                                if not re.fullmatch(
                                    r"gimg_[0-9a-f]{24}", row["image_ref"]
                                ) or not re.fullmatch(
                                    r"blob_[0-9a-f]{64}", row["filename"]
                                ):
                                    raise ValueError("旧群缓存索引格式无效")
                                if not (old_cache / row["filename"]).is_file():
                                    counts["missing"] += 1
                                    continue
                                if cache.execute(
                                    "SELECT 1 FROM images WHERE image_ref=?",
                                    (row["image_ref"],),
                                ).fetchone():
                                    continue
                                _copy(
                                    old_cache / row["filename"],
                                    group_cache.root / row["filename"],
                                    old_cache,
                                )
                                cache.execute(
                                    "INSERT INTO images (image_ref, platform_id, origin, conversation_id, "
                                    "filename, size_bytes, mime_type, width, height, received_at, expires_at, is_animated) "
                                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    tuple(
                                        row[key]
                                        for key in (
                                            "image_ref",
                                            "platform_id",
                                            "origin",
                                            "conversation_id",
                                            "filename",
                                            "size_bytes",
                                            "mime_type",
                                            "width",
                                            "height",
                                            "received_at",
                                            "expires_at",
                                        )
                                    )
                                    + (
                                        detect_animation(
                                            group_cache.root / row["filename"]
                                        ),
                                    ),
                                )
                                counts["group_images"] += 1
            # No source yet: allow a later installation/import to discover it.
            if legacy_db.is_file() or (old_cache / "index.sqlite3").is_file():
                target.execute(
                    "INSERT INTO migrations VALUES (?, ?)",
                    (MIGRATION_NAME, int(time.time())),
                )
                target.commit()
        return counts

    task = asyncio.create_task(asyncio.to_thread(migrate))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
    if any(result.values()):
        logger.info("Media 管理：旧 QQ 图片导入结果 %s；源数据保留用于回退", result)
    return result
