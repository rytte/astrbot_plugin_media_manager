from __future__ import annotations

from copy import deepcopy

DEFAULT_CONFIG = {
    "enabled": True,
    "group_cache_enabled": True,
    "group_cache_max_mb": 256,
    "group_cache_ttl_hours": 24,
    "orphan_grace_days": 3,
    "retention_days": 30,
    "max_storage_mb": 2048,
    "max_inspect_mb": 10,
    "max_components": 30,  # Legacy key: group-cache images per message only.
    "cleanup_interval_seconds": 600,
    "disabled_views": [],
    "platform_ids": [],
    "import_qq_enhance": True,
}


def validate_config(config=None):
    config = {} if config is None else dict(config)
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"未知 Media 管理配置：{', '.join(sorted(unknown))}")
    result = {**deepcopy(DEFAULT_CONFIG), **deepcopy(config)}
    for key in ("enabled", "group_cache_enabled", "import_qq_enhance"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} 必须是布尔值")
    ranges = {
        "group_cache_max_mb": (1, 10240),
        "group_cache_ttl_hours": (1, 168),
        "orphan_grace_days": (1, 30),
        "max_storage_mb": (64, 10240),
        "max_inspect_mb": (1, 100),
        "max_components": (1, 100),
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
