from __future__ import annotations

import json
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from astrbot.api.provider import ProviderRequest
from astrbot.api.message_components import Image, Reply
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.agent.message import (
    ImageURLPart,
    Message,
    TextPart,
    dump_messages_with_checkpoints,
)
from astrbot_plugin_media_manager.config import (
    DEFAULT_CONFIG,
    group_config,
    validate_config,
)
from astrbot_plugin_media_manager.context_images import ContextImageError
from astrbot_plugin_media_manager.group_image_cache import GROUP_IMAGES_EXTRA
from astrbot_plugin_media_manager.image_links import IMAGE_LINKS_EXTRA
from astrbot_plugin_media_manager.main import MediaManagerPlugin
from test_animation_preview import animation_bytes
from test_context_images import Event, build_manager, save_image


async def make_plugin(tmp_path, config=None):
    manager, storage, conversations = await build_manager(tmp_path)
    plugin = object.__new__(MediaManagerPlugin)
    plugin.config = validate_config(config)
    plugin.images = manager
    plugin.images.archive_enabled = plugin.config["image_lighten_enabled"]
    plugin.images.disabled_views = set(plugin.config["disabled_views"])
    plugin.storage = storage
    plugin.ready = True
    plugin._last_cleanup_day = None
    plugin.context = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="c1")
        ),
        get_config=lambda **kwargs: {},
    )
    return plugin, conversations


@pytest.mark.asyncio
async def test_all_images_receive_stable_ids_beyond_the_cache_limit(tmp_path):
    plugin, _ = await make_plugin(tmp_path)
    event = Event()
    event.message_obj.message = [
        Reply(id="old", chain=[Image(file="quoted.png")]),
        *[Image(file="repeated.png") for _ in range(105)],
    ]
    await plugin.assign_image_ids(event)
    links = dict(event.get_extra(IMAGE_LINKS_EXTRA))
    assert len(links) == len(set(links.values())) == 106
    await plugin.assign_image_ids(event)
    assert event.get_extra(IMAGE_LINKS_EXTRA) == links


@pytest.mark.asyncio
@pytest.mark.parametrize("lighten_enabled", [True, False])
async def test_group_cache_limit_preserves_all_visual_input_and_references(
    tmp_path, monkeypatch, lighten_enabled
):
    plugin, _ = await make_plugin(
        tmp_path,
        {"image_lighten_enabled": lighten_enabled},
    )
    monkeypatch.setattr(
        "astrbot.core.star.session_plugin_manager.SessionPluginManager.is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    plugin.context.get_config = lambda **_: {
        "provider_ltm_settings": {"group_icl_enable": True}
    }
    paths = [tmp_path / f"image-{index}.png" for index in range(32)]
    for index, path in enumerate(paths):
        save_image(path, "PNG", (index + 3, 4))
    event = Event(unified_msg_origin="platform-a:GroupMessage:group-a")
    event.get_group_id = lambda: "group-a"
    event.message_str = "查看这些图片"
    event.message_obj.message = [
        Reply(id="old", chain=[Image(file=str(paths[-1]))]),
        *[Image(file=str(path)) for path in paths],
    ]
    await plugin.assign_image_ids(event)
    links = dict(event.get_extra(IMAGE_LINKS_EXTRA))
    await plugin.cache_group_images(event)
    cached = event.get_extra(GROUP_IMAGES_EXTRA)
    assert len(cached) == 30
    assert (
        [item["source"] for item in cached]
        == [
            str(paths[-1]),  # Quoted images count in message order.
            *map(str, paths[:29]),
        ]
    )
    assert event.get_extra(IMAGE_LINKS_EXTRA) == links
    visual_sources = [str(paths[-1]), *map(str, paths)]
    request = ProviderRequest(
        prompt=event.message_str,
        conversation=SimpleNamespace(cid="c1"),
        image_urls=visual_sources.copy(),
    )
    await plugin.prepare_images(event, request)
    assert request.image_urls == visual_sources
    reference_text = "\n".join(
        [request.prompt, *[part.text for part in request.extra_user_content_parts]]
    )
    cached_sources = {item["source"] for item in cached}
    for source, link in zip(visual_sources, links.values()):
        assert reference_text.count(link) == int(
            lighten_enabled or source in cached_sources
        )
    current = Message.model_validate(await request.assemble_context())
    run_context = SimpleNamespace(messages=[current])
    await plugin.bind_image_refs(event, run_context)
    visuals = [part for part in current.content if isinstance(part, ImageURLPart)]
    assert len(visuals) == 33
    assert all(not part._no_save for part in visuals)
    assert sum(bool(part.image_url.id) for part in visuals) == (
        33 if lighten_enabled else 31
    )
    await plugin.lighten_history(event, run_context, None)
    history = json.dumps(dump_messages_with_checkpoints([current]))
    assert ("data:image" not in history) is lighten_enabled
    for source, link in zip(visual_sources, links.values()):
        assert (link in history) == (lighten_enabled or source in cached_sources)
    assert len(await plugin.storage.list_context_images()) == (
        2 if lighten_enabled else 0
    )
    result = await plugin.media_inspect(event, cached[0]["image_ref"])
    assert not result.isError


@pytest.mark.asyncio
@pytest.mark.parametrize("lighten_enabled", [True, False])
@pytest.mark.parametrize("is_private", [True, False])
async def test_lightening_switch_controls_private_and_uncached_group_images(
    tmp_path, monkeypatch, lighten_enabled, is_private
):
    plugin, _ = await make_plugin(
        tmp_path,
        {"lightening": {"image_lighten_enabled": lighten_enabled}},
    )
    monkeypatch.setattr(
        "astrbot.core.star.session_plugin_manager.SessionPluginManager.is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    path = tmp_path / "input.png"
    save_image(path, "PNG", (4, 5))
    origin = (
        "platform-a:FriendMessage:10001"
        if is_private
        else "platform-a:GroupMessage:group-a"
    )
    event = Event(path, unified_msg_origin=origin)
    request = ProviderRequest(
        prompt="查看图片",
        system_prompt="原始提示",
        conversation=SimpleNamespace(cid="c1"),
        image_urls=[str(path)],
    )
    request.func_tool = SimpleNamespace(remove_tool=Mock())
    await plugin.assign_image_ids(event)
    await plugin.prepare_images(event, request)
    assert request.image_urls == [str(path)]
    current = Message.model_validate(await request.assemble_context())
    run_context = SimpleNamespace(messages=[current])
    await plugin.bind_image_refs(event, run_context)
    await plugin.lighten_history(event, run_context, None)
    history = json.dumps(dump_messages_with_checkpoints([current]))
    records = await plugin.storage.list_context_images()
    if not lighten_enabled:
        assert not records
        assert request.prompt == "查看图片"
        assert request.extra_user_content_parts == []
        assert "data:image" in history
        assert "Media ImageRef" not in history
        if is_private:
            assert not event.extras
            assert request.system_prompt == "原始提示"
            request.func_tool.remove_tool.assert_called_once_with("media_inspect")
        else:
            request.func_tool.remove_tool.assert_not_called()
        result = await plugin.media_inspect(event, "img_" + "a" * 24)
        assert result.isError
        assert json.loads(result.content[0].text)["error"]["code"] == (
            "capability_unavailable"
        )
    else:
        assert len(records) == 1
        assert "data:image" not in history
        assert records[0]["image_ref"] in history
        request.func_tool.remove_tool.assert_not_called()
        result = await plugin.media_inspect(event, records[0]["image_ref"])
        assert not result.isError


@pytest.mark.asyncio
async def test_failed_cache_attempts_still_respect_the_per_message_limit(tmp_path):
    plugin, _ = await make_plugin(tmp_path)
    plugin.context.get_config = lambda **_: {
        "provider_ltm_settings": {"group_icl_enable": True}
    }
    plugin.images.group_cache.put = AsyncMock(side_effect=OSError("unavailable"))
    event = Event()
    event.get_group_id = lambda: "group-a"
    event.message_obj.message = [
        Image(file=f"missing-{index}.png") for index in range(34)
    ]
    await plugin.cache_group_images(event)
    assert plugin.images.group_cache.put.await_count == 30
    assert len(event.get_extra(IMAGE_LINKS_EXTRA)) == 34
    assert event.get_extra(GROUP_IMAGES_EXTRA) is None


@pytest.mark.asyncio
async def test_standalone_animation_tool_does_not_rearchive_its_preview(
    tmp_path, monkeypatch
):
    plugin, _ = await make_plugin(tmp_path)
    monkeypatch.setattr(
        "astrbot.core.star.session_plugin_manager.SessionPluginManager.is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    path = tmp_path / "animation.gif"
    path.write_bytes(animation_bytes())
    event = Event(path)
    request = ProviderRequest(
        conversation=SimpleNamespace(cid="c1"), image_urls=[str(path)]
    )
    await plugin.prepare_images(event, request)
    image_ref = (await plugin.storage.list_context_images())[0]["image_ref"]
    result = await plugin.media_inspect(event, image_ref, "animation")
    assert not result.isError
    assert "采样 9 帧" in result.content[0].text
    assistant = Message(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "type": "function",
                "id": "call1",
                "function": {
                    "name": "media_inspect",
                    "arguments": json.dumps(
                        {"image_ref": image_ref, "view": "animation"}
                    ),
                },
            }
        ],
    )
    tool = Message(
        role="tool",
        tool_call_id="call1",
        content=result.content[0].text
        + "\nImage returned and cached at path='preview.png'.",
    )
    attachment = Message(
        role="user",
        content=[
            TextPart(text="[Image from tool 'media_inspect', path='preview.png']"),
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(
                    url="data:image/png;base64," + result.content[1].data
                )
            ),
        ],
    )
    await plugin.lighten_history(
        event, SimpleNamespace(messages=[assistant, tool, attachment]), None
    )
    assert attachment._no_save
    assert tool.content == result.content[0].text
    assert len(await plugin.storage.list_context_images()) == 1
    assert "preview.png" not in str(dump_messages_with_checkpoints([assistant, tool]))


@pytest.mark.asyncio
async def test_views_and_session_disable_are_enforced_by_public_tool(
    tmp_path, monkeypatch
):
    plugin, _ = await make_plugin(tmp_path, {"disabled_views": ["animation"]})
    event = Event()
    with pytest.raises(ContextImageError, match="禁用"):
        await plugin.inspect_reference(event, "img_" + "a" * 24, "animation")
    monkeypatch.setattr(
        "astrbot.core.star.session_plugin_manager.SessionPluginManager.is_plugin_enabled_for_session",
        AsyncMock(return_value=False),
    )
    result = await plugin.media_inspect(event, "img_" + "a" * 24)
    assert result.isError
    assert "停用" in result.content[0].text


@pytest.mark.asyncio
async def test_current_conversation_is_resolved_again_before_view(
    tmp_path, monkeypatch
):
    plugin, _ = await make_plugin(tmp_path)
    path = tmp_path / "image.png"
    save_image(path, "PNG", (3, 4))
    event = Event(path)
    await plugin.prepare_images(
        event,
        ProviderRequest(conversation=SimpleNamespace(cid="c1"), image_urls=[str(path)]),
    )
    image_ref = (await plugin.storage.list_context_images())[0]["image_ref"]
    monkeypatch.setattr(
        "astrbot.core.star.session_plugin_manager.SessionPluginManager.is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    plugin.context.conversation_manager.get_curr_conversation_id.return_value = "c2"
    result = await plugin.media_inspect(event, image_ref)
    assert result.isError
    assert "target_not_found" in result.content[0].text


@pytest.mark.asyncio
async def test_history_liveness_accepts_new_and_old_reference_formats(tmp_path):
    plugin, conversations = await make_plugin(tmp_path)
    path = tmp_path / "image.png"
    save_image(path, "PNG", (3, 4))
    event = Event(path)
    records = []
    for _ in range(3):
        records.append(
            (await plugin.images._archive_sources(event, "c1", [str(path)]))[str(path)]
        )
    refs = [record["image_ref"] for record in records]
    conversations.histories[event.unified_msg_origin, "c1"] = [
        {
            "role": "user",
            "content": f"[Media ImageRef image_ref={refs[0]}, 3x4；…] [QQ ImageRef image_ref={refs[1]}, 3x4；…] [QQ component|QQ图片；image_ref={refs[2]}；3x4]",
        }
    ]
    await plugin.images.reconcile()
    assert all(
        row["orphaned_at"] is None for row in await plugin.storage.list_context_images()
    )


@pytest.mark.asyncio
async def test_lifecycle_reconciles_at_most_once_per_local_day(tmp_path):
    plugin, _ = await make_plugin(tmp_path)
    plugin.images.reconcile = AsyncMock()
    with patch(
        "astrbot_plugin_media_manager.main.time.localtime",
        return_value=time.struct_time((2026, 10, 24, 1, 0, 0, 5, 297, -1)),
    ):
        await plugin.cleanup()
        await plugin.cleanup()
    plugin.images.reconcile.assert_awaited_once()
    with patch(
        "astrbot_plugin_media_manager.main.time.localtime",
        return_value=time.struct_time((2026, 10, 25, 1, 0, 0, 6, 298, -1)),
    ):
        await plugin.cleanup()
    assert plugin.images.reconcile.await_count == 2


@pytest.mark.parametrize(
    "config",
    [
        {"max_inspect_mb": 0},
        {"lightening": {"image_lighten_enabled": 1}},
        {"group_cache_enabled": 1},
        {"retention_days": False},
        {"disabled_views": ["video"]},
    ],
)
def test_media_config_rejects_invalid_limits_and_views(config):
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_ref,view",
    [
        (None, "original"),
        ([], "original"),
        ("img_" + "a" * 24, []),
        ("pic_" + "a" * 16, "original"),
    ],
)
async def test_invalid_tool_values_return_structured_errors(tmp_path, image_ref, view):
    plugin, _ = await make_plugin(tmp_path)
    result = await plugin.media_inspect(Event(), image_ref, view)
    assert result.isError
    assert json.loads(result.content[0].text)["error"]["code"] == "invalid_parameters"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ready,config", [(False, {}), (True, {"platform_ids": ["another-platform"]})]
)
async def test_unready_and_filtered_platforms_do_not_process_input(
    tmp_path, ready, config
):
    plugin, _ = await make_plugin(tmp_path, config)
    plugin.ready = ready
    event = Event(tmp_path / "missing.png")
    tools = SimpleNamespace(remove_tool=Mock())
    request = SimpleNamespace(func_tool=tools)
    await plugin.assign_image_ids(event)
    await plugin.prepare_images(event, request)
    assert not event.extras
    assert await plugin.storage.list_context_images() == []
    tools.remove_tool.assert_called_once_with("media_inspect")
    result = await plugin.media_inspect(event, "img_" + "a" * 24)
    assert result.isError
    assert "未启用" in result.content[0].text


@pytest.mark.asyncio
async def test_all_views_disabled_remove_tool_and_its_entry_hints(tmp_path):
    plugin, _ = await make_plugin(
        tmp_path, {"disabled_views": ["original", "animation"]}
    )
    path = tmp_path / "input.png"
    save_image(path, "PNG", (4, 5))
    request = ProviderRequest(
        conversation=SimpleNamespace(cid="c1"), image_urls=[str(path)]
    )
    request.func_tool = SimpleNamespace(remove_tool=Mock())
    await plugin.prepare_images(Event(path), request)
    request.func_tool.remove_tool.assert_called_once_with("media_inspect")
    refs = "\n".join(part.text for part in request.extra_user_content_parts)
    assert "Media ImageRef" in refs
    assert "media_inspect(" not in refs
    assert "view=animation" not in request.system_prompt
    assert request.image_urls == [str(path)]


def test_schema_defaults_are_loadable_by_astrbot():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    loader = AstrBotConfig.__new__(AstrBotConfig)
    defaults = loader._config_schema_to_default_config(schema)
    assert defaults == group_config()
    assert validate_config(defaults) == DEFAULT_CONFIG


@pytest.mark.asyncio
async def test_real_initialization_preserves_settings_and_stops_cleanup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "astrbot_plugin_media_manager.main.get_astrbot_plugin_data_path",
        lambda: str(tmp_path / "data"),
    )
    context = SimpleNamespace(
        conversation_manager=SimpleNamespace(get_conversation=AsyncMock())
    )
    config = group_config(
        {
            "retention_days": 7,
            "max_storage_mb": 128,
            "group_cache_enabled": False,
            "image_lighten_enabled": False,
        }
    )
    plugin = MediaManagerPlugin(context, config)
    try:
        await plugin.initialize()
        assert plugin.ready
        assert plugin.images.retention_days == 7
        assert plugin.images.max_storage_bytes == 128 * 1024 * 1024
        assert not plugin.config["group_cache_enabled"]
        assert not plugin.config["image_lighten_enabled"]
        assert not plugin.images.archive_enabled
        assert config == group_config(plugin.config)
        await asyncio.sleep(0)
    finally:
        await plugin.terminate()
    assert not plugin.ready
    assert plugin.cleanup_task.done()
    config["lightening"]["retention_days"] = 30
    reloaded = MediaManagerPlugin(context, config)
    try:
        await reloaded.initialize()
        assert reloaded.config["retention_days"] == 30
        assert reloaded.config["max_storage_mb"] == 128
    finally:
        await reloaded.terminate()
