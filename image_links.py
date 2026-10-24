"""Versioned, optional per-event image association shared with other plugins."""

from __future__ import annotations

import secrets

from astrbot.api.message_components import Image, Reply

IMAGE_LINKS_EXTRA = "_image_context_links_v1"


def new_image_id() -> str:
    return f"pic_{secrets.token_hex(8)}"


def image_components(event):
    """Yield occurrences, keeping quoted and current images distinct."""
    for component in getattr(getattr(event, "message_obj", None), "message", []) or []:
        if isinstance(component, Image):
            yield component
        elif isinstance(component, Reply):
            yield from (
                part for part in component.chain or [] if isinstance(part, Image)
            )


def ensure_image_links(event):
    """Allocate IDs before QQ semantics; object IDs never leave this event."""
    links = event.get_extra(IMAGE_LINKS_EXTRA, {})
    if not isinstance(links, dict):
        links = {}
    for component in image_components(event):
        if id(component) not in links:
            links[id(component)] = new_image_id()
    event.set_extra(IMAGE_LINKS_EXTRA, links)
    return links


def image_occurrences(event):
    links = event.get_extra(IMAGE_LINKS_EXTRA, {})
    return [
        (component, links[id(component)])
        for component in image_components(event)
        if id(component) in links
    ]
