from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

DEFAULT_CONFIG = {
    "image_lighten_enabled": True,
    "group_cache_enabled": True,
    "group_cache_max_mb": 256,
    "group_cache_ttl_hours": 24,
    "orphan_grace_days": 3,
    "retention_days": 30,
    "max_storage_mb": 2048,
    "max_inspect_mb": 10,
    "cleanup_interval_seconds": 600,
    "disabled_views": [],
    "platform_ids": [],
}

CONFIG_GROUPS = {
    "general": ("platform_ids",),
    "lightening": (
        "image_lighten_enabled",
        "max_storage_mb",
        "retention_days",
        "orphan_grace_days",
    ),
    "group_cache": (
        "group_cache_enabled",
        "group_cache_max_mb",
        "group_cache_ttl_hours",
    ),
    "inspection": ("max_inspect_mb", "disabled_views", "cleanup_interval_seconds"),
}

LEGACY_CONFIG_GROUPS = {
    "private_chat": ("private_image_lighten_enabled",),
    "archive": ("max_storage_mb", "retention_days", "orphan_grace_days"),
}


def validate_config(config=None):
    config = {} if config is None else dict(config)
    for group, keys in {**LEGACY_CONFIG_GROUPS, **CONFIG_GROUPS}.items():
        if group not in config:
            continue
        section = config.pop(group)
        if not isinstance(section, dict):
            raise ValueError(f"{group} 必须是配置对象")
        unknown = set(section) - set(keys)
        if unknown:
            raise ValueError(f"未知 {group} 配置：{', '.join(sorted(unknown))}")
        # Explicit grouped settings take precedence during upgrades from flat keys.
        config.update(section)
    if "private_image_lighten_enabled" in config:
        config.setdefault(
            "image_lighten_enabled", config.pop("private_image_lighten_enabled")
        )
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"未知 Media 管理配置：{', '.join(sorted(unknown))}")
    result = {**deepcopy(DEFAULT_CONFIG), **deepcopy(config)}
    for key in ("image_lighten_enabled", "group_cache_enabled"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} 必须是布尔值")
    ranges = {
        "group_cache_max_mb": (1, 10240),
        "group_cache_ttl_hours": (1, 168),
        "orphan_grace_days": (1, 30),
        "max_storage_mb": (64, 10240),
        "max_inspect_mb": (1, 100),
        "cleanup_interval_seconds": (60, 86400),
    }
    for key, (minimum, maximum) in ranges.items():
        if type(result[key]) is not int or not minimum <= result[key] <= maximum:
            raise ValueError(f"{key} 必须是 {minimum}～{maximum} 的整数")
    if type(result["retention_days"]) is not int or result["retention_days"] < 0:
        raise ValueError("retention_days 必须是大于等于 0 的整数")
    for key in ("platform_ids", "disabled_views"):
        if not isinstance(result[key], list) or any(
            not isinstance(item, str) for item in result[key]
        ):
            raise ValueError(f"{key} 必须是字符串列表")
    if set(result["disabled_views"]) - {"original", "animation"}:
        raise ValueError("disabled_views 仅支持 original、animation")
    return result


def group_config(config=None):
    flat = validate_config(config)
    return {
        group: {key: flat[key] for key in keys} for group, keys in CONFIG_GROUPS.items()
    }


def migrate_config_file(path: Path) -> bool:
    """Preserve flat settings before AstrBot prunes keys absent from the new schema."""
    if not path.is_file():
        return False
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Media 管理配置必须是配置对象")
    retired_keys = {"enabled", "max_components", "import_qq_enhance"}
    cache_config = config.get("group_cache")
    has_retired_cache_limit = (
        isinstance(cache_config, dict) and "max_components" in cache_config
    )
    general_config = config.get("general")
    has_retired_import = (
        isinstance(general_config, dict) and "import_qq_enhance" in general_config
    )
    legacy_keys = (
        set(DEFAULT_CONFIG)
        | retired_keys
        | set(LEGACY_CONFIG_GROUPS)
        | {"private_image_lighten_enabled"}
    )
    if not (set(config) & legacy_keys or has_retired_cache_limit or has_retired_import):
        return False
    for key in retired_keys:
        config.pop(key, None)
    if has_retired_cache_limit:
        cache_config.pop("max_components")
    if has_retired_import:
        general_config.pop("import_qq_enhance")
    grouped = group_config(config)
    staging = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            staging = Path(stream.name)
            json.dump(grouped, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        staging.replace(path)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
    return True
