from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .image_metadata import detect_animation


class Storage:
    """Own image indexes independently of QQ actions, audits and confirmations."""

    def __init__(self, database_path: Path):
        self.database_path = database_path

    async def initialize(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        def initialize_sync():
            with sqlite3.connect(self.database_path, timeout=10) as connection:
                connection.executescript("""
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS migrations (
                        name TEXT PRIMARY KEY, completed_at INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS context_images (
                        image_ref TEXT PRIMARY KEY,
                        platform_id TEXT NOT NULL,
                        unified_msg_origin TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        path TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        width INTEGER NOT NULL,
                        height INTEGER NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        created_at INTEGER NOT NULL,
                        last_accessed_at INTEGER NOT NULL,
                        orphaned_at INTEGER,
                        is_animated INTEGER
                    );
                    CREATE INDEX IF NOT EXISTS idx_context_images_scope
                        ON context_images(
                            platform_id, unified_msg_origin, conversation_id
                        );
                    CREATE INDEX IF NOT EXISTS idx_context_images_orphaned
                        ON context_images(orphaned_at);
                    CREATE INDEX IF NOT EXISTS idx_context_images_created
                        ON context_images(created_at);
                """)
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(context_images)")
                }
                if "is_animated" not in columns:
                    connection.execute(
                        "ALTER TABLE context_images ADD COLUMN is_animated INTEGER"
                    )

        await asyncio.to_thread(initialize_sync)

    async def backfill_image_animation(self, image_root: Path):
        """Fill missing metadata for owned legacy images without changing lifetimes."""
        root = image_root.resolve()

        def backfill():
            with sqlite3.connect(self.database_path, timeout=10) as connection:
                rows = connection.execute(
                    "SELECT image_ref, path FROM context_images WHERE is_animated IS NULL"
                ).fetchall()
                for image_ref, filename in rows:
                    path = Path(filename).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        continue
                    state = detect_animation(path)
                    if state is not None:
                        connection.execute(
                            "UPDATE context_images SET is_animated=? WHERE image_ref=? AND is_animated IS NULL",
                            (state, image_ref),
                        )

        task = asyncio.create_task(asyncio.to_thread(backfill))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def add_context_images(
        self,
        records: list[dict[str, Any]],
        max_storage_bytes: int,
    ) -> list[Path] | None:
        """Atomically evict oldest local-day buckets and persist image records.

        Args:
            records: Fully validated image metadata records.
            max_storage_bytes: Maximum bytes allowed for all context images.

        Returns:
            Evicted file paths, or ``None`` when the batch alone exceeds capacity.
        """

        def add_sync() -> list[Path] | None:
            with sqlite3.connect(self.database_path, timeout=10) as connection:
                connection.execute("BEGIN IMMEDIATE")
                stored_bytes = int(
                    connection.execute(
                        "SELECT COALESCE(SUM(size_bytes), 0) FROM context_images"
                    ).fetchone()[0]
                )
                incoming_bytes = sum(int(record["size_bytes"]) for record in records)
                if incoming_bytes > max_storage_bytes:
                    connection.rollback()
                    return None
                evicted_paths: list[Path] = []
                while stored_bytes + incoming_bytes > max_storage_bytes:
                    oldest_row = connection.execute(
                        "SELECT MIN(created_at) FROM context_images"
                    ).fetchone()
                    oldest_at = oldest_row[0] if oldest_row else None
                    if oldest_at is None:
                        raise RuntimeError("context image capacity accounting failed")
                    oldest_day = datetime.fromtimestamp(int(oldest_at)).date()
                    next_day_start = int(
                        datetime.combine(
                            oldest_day + timedelta(days=1), datetime.min.time()
                        ).timestamp()
                    )
                    day_rows = connection.execute(
                        """
                        SELECT path, size_bytes FROM context_images
                        WHERE created_at < ?
                        """,
                        (next_day_start,),
                    ).fetchall()
                    if not day_rows:
                        raise RuntimeError("oldest context image day was empty")
                    connection.execute(
                        "DELETE FROM context_images WHERE created_at < ?",
                        (next_day_start,),
                    )
                    stored_bytes -= sum(int(row[1]) for row in day_rows)
                    evicted_paths.extend(Path(row[0]) for row in day_rows)
                connection.executemany(
                    """
                    INSERT INTO context_images (
                        image_ref, platform_id, unified_msg_origin,
                        conversation_id, path, mime_type, width, height,
                        size_bytes, created_at, last_accessed_at, orphaned_at, is_animated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    [
                        (
                            record["image_ref"],
                            record["platform_id"],
                            record["unified_msg_origin"],
                            record["conversation_id"],
                            str(record["path"]),
                            record["mime_type"],
                            record["width"],
                            record["height"],
                            record["size_bytes"],
                            record["created_at"],
                            record["last_accessed_at"],
                            record.get("is_animated"),
                        )
                        for record in records
                    ],
                )
                connection.commit()
                return evicted_paths

        return await asyncio.to_thread(add_sync)

    async def resolve_context_image(
        self,
        image_ref: str,
        platform_id: str,
        unified_msg_origin: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        """Resolve one image only inside its originating conversation scope.

        Args:
            image_ref: Opaque persistent image identifier.
            platform_id: Current platform instance ID.
            unified_msg_origin: Current AstrBot session origin.
            conversation_id: Current AstrBot conversation ID.

        Returns:
            Stored metadata, or ``None`` when the reference is out of scope.
        """

        now = int(time.time())

        def resolve_sync() -> dict[str, Any] | None:
            with sqlite3.connect(self.database_path, timeout=10) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT * FROM context_images
                    WHERE image_ref = ? AND platform_id = ?
                      AND unified_msg_origin = ? AND conversation_id = ?
                    """,
                    (
                        image_ref,
                        platform_id,
                        unified_msg_origin,
                        conversation_id,
                    ),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return None
                connection.execute(
                    """
                    UPDATE context_images SET last_accessed_at = ?
                    WHERE image_ref = ?
                    """,
                    (now, image_ref),
                )
                connection.commit()
                result = dict(row)
                result["last_accessed_at"] = now
                return result

        return await asyncio.to_thread(resolve_sync)

    async def list_context_images(self) -> list[dict[str, Any]]:
        """List context image metadata for lifecycle reconciliation.

        Returns:
            All stored context image records.
        """

        def list_sync() -> list[dict[str, Any]]:
            with sqlite3.connect(self.database_path, timeout=10) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT * FROM context_images ORDER BY created_at"
                ).fetchall()
                return [dict(row) for row in rows]

        return await asyncio.to_thread(list_sync)

    async def reconcile_context_images(
        self,
        liveness: dict[str, bool],
        checked_at: int,
        orphan_cutoff_exclusive: int,
        retention_cutoff_exclusive: int | None,
    ) -> list[Path]:
        """Update checked references and remove naturally expired image records.

        Args:
            liveness: Checked image references mapped to their history presence.
            checked_at: Timestamp used when first marking an orphan.
            orphan_cutoff_exclusive: Orphans before this local-day boundary expire.
            retention_cutoff_exclusive: Images before this local-day boundary expire.

        Returns:
            Paths belonging to deleted records.
        """

        def reconcile_sync() -> list[Path]:
            with sqlite3.connect(self.database_path, timeout=10) as connection:
                connection.execute("BEGIN IMMEDIATE")
                for image_ref, is_live in liveness.items():
                    if is_live:
                        connection.execute(
                            """
                            UPDATE context_images SET orphaned_at = NULL
                            WHERE image_ref = ?
                            """,
                            (image_ref,),
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE context_images
                            SET orphaned_at = COALESCE(orphaned_at, ?)
                            WHERE image_ref = ?
                            """,
                            (checked_at, image_ref),
                        )
                expired_rows: list[tuple[str, str]] = []
                expired_refs: set[str] = set()
                if retention_cutoff_exclusive is not None:
                    expired_rows.extend(
                        connection.execute(
                            """
                            SELECT image_ref, path FROM context_images
                            WHERE created_at < ?
                            """,
                            (retention_cutoff_exclusive,),
                        ).fetchall()
                    )
                    expired_refs.update(row[0] for row in expired_rows)
                for image_ref, is_live in liveness.items():
                    if is_live or image_ref in expired_refs:
                        continue
                    row = connection.execute(
                        """
                        SELECT image_ref, path FROM context_images
                        WHERE image_ref = ? AND orphaned_at IS NOT NULL
                          AND orphaned_at < ?
                        """,
                        (image_ref, orphan_cutoff_exclusive),
                    ).fetchone()
                    if row is None:
                        continue
                    expired_rows.append(row)
                    expired_refs.add(image_ref)
                connection.executemany(
                    "DELETE FROM context_images WHERE image_ref = ?",
                    [(image_ref,) for image_ref in expired_refs],
                )
                connection.commit()
                return [Path(row[1]) for row in expired_rows]

        return await asyncio.to_thread(reconcile_sync)
