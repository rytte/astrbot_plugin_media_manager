from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot_plugin_media_manager.config import (
    DEFAULT_CONFIG,
    group_config,
    migrate_config_file,
    validate_config,
)


def test_flat_config_migration_preserves_settings_when_loaded_by_astrbot(tmp_path):
    path = tmp_path / "media_config.json"
    settings = {
        "group_cache_enabled": False,
        "group_cache_max_mb": 128,
        "group_cache_ttl_hours": 12,
        "orphan_grace_days": 7,
        "retention_days": 0,
        "max_storage_mb": 512,
        "max_inspect_mb": 17,
        "cleanup_interval_seconds": 120,
        "disabled_views": ["animation"],
        "platform_ids": ["qq-a"],
    }
    path.write_text(
        json.dumps(
            {
                **settings,
                "enabled": False,
                "max_components": 5,
                "import_qq_enhance": False,
            }
        ),
        encoding="utf-8-sig",
    )
    assert migrate_config_file(path)
    schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    loaded = AstrBotConfig(str(path), schema=schema)
    assert validate_config(loaded) == {**DEFAULT_CONFIG, **settings}
    assert loaded["lightening"]["image_lighten_enabled"] is True
    saved = path.read_bytes()
    assert not migrate_config_file(path)
    assert path.read_bytes() == saved


@pytest.mark.parametrize(
    "section,option,value",
    [("group_cache", "max_components", 5), ("general", "import_qq_enhance", True)],
)
def test_grouped_config_removes_retired_options_and_preserves_settings(
    tmp_path, section, option, value
):
    path = tmp_path / "media_config.json"
    expected = group_config({"group_cache_max_mb": 128, "retention_days": 7})
    previous = group_config(expected)
    previous[section][option] = value
    path.write_text(json.dumps(previous), encoding="utf-8")
    assert migrate_config_file(path)
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert not migrate_config_file(path)


def test_grouped_values_win_over_flat_values_during_migration(tmp_path):
    path = tmp_path / "media_config.json"
    path.write_text(
        json.dumps(
            {
                "retention_days": 7,
                "max_storage_mb": 128,
                "archive": {"retention_days": 14},
                "private_chat": {"private_image_lighten_enabled": False},
            }
        ),
        encoding="utf-8",
    )
    assert migrate_config_file(path)
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["lightening"]["retention_days"] == 14
    assert migrated["lightening"]["max_storage_mb"] == 128
    assert migrated["lightening"]["image_lighten_enabled"] is False
    assert "archive" not in migrated
    assert "private_chat" not in migrated


@pytest.mark.parametrize("enabled", [True, False])
def test_old_private_and_archive_groups_merge_without_resetting_settings(
    tmp_path, enabled
):
    path = tmp_path / "media_config.json"
    expected = group_config(
        {
            "image_lighten_enabled": enabled,
            "max_storage_mb": 128,
            "retention_days": 7,
            "orphan_grace_days": 5,
            "group_cache_enabled": False,
        }
    )
    previous = group_config(expected)
    previous.pop("lightening")
    previous["private_chat"] = {"private_image_lighten_enabled": enabled}
    previous["archive"] = {
        "max_storage_mb": 128,
        "retention_days": 7,
        "orphan_grace_days": 5,
    }
    path.write_text(json.dumps(previous), encoding="utf-8-sig")
    assert migrate_config_file(path)
    schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    loaded = AstrBotConfig(str(path), schema=schema)
    assert loaded == expected
    assert not migrate_config_file(path)


def test_new_lightening_settings_take_precedence_over_legacy_values():
    config = {
        "private_chat": {"private_image_lighten_enabled": False},
        "archive": {"retention_days": 7},
        "lightening": {"image_lighten_enabled": True, "retention_days": 14},
    }
    validated = validate_config(config)
    assert validated["image_lighten_enabled"] is True
    assert validated["retention_days"] == 14


def test_failed_migration_keeps_original_config_and_removes_staging_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "media_config.json"
    original = b'{"retention_days": 7}'
    path.write_bytes(original)

    def fail_replace(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        migrate_config_file(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_absent_or_grouped_config_needs_no_migration(tmp_path):
    path = tmp_path / "media_config.json"
    assert not migrate_config_file(path)
    original = json.dumps(group_config()).encode("utf-8")
    path.write_bytes(original)
    assert not migrate_config_file(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "config",
    [
        {"lightening": False},
        {"lightening": {"image_lighten_enabled": "false"}},
        {"private_chat": False},
        {"private_chat": {"private_image_lighten_enabled": "false"}},
        {"archive": {"retention_days": False}},
        {"archive": {"unknown_option": 1}},
        {"general": {"retention_days": 7}},
    ],
)
def test_invalid_grouped_config_is_rejected(config):
    with pytest.raises(ValueError):
        validate_config(config)
