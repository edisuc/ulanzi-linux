"""Backend ranking and argv construction for key injection.

Two callers dispatch shortcuts: ``ActionRunner`` runs them in-process, and the
session agent runs them on behalf of the daemon, which has no session of its
own. Both must agree on which tool to reach for, so the decision lives here
rather than in either of them. It used to live only in ``ActionRunner``, and
the agent's private copy knew nothing of ``ydotool`` — under Wayland that
confined every shortcut the agent sent to XWayland clients.
"""

from __future__ import annotations

from collections.abc import Mapping

from ulanzi_linux.infrastructure.keysym_evdev import translate_shortcut


def is_wayland_session(env: Mapping[str, str]) -> bool:
    """True when ``env`` describes a Wayland session."""
    if env.get("WAYLAND_DISPLAY"):
        return True
    return env.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def shortcut_tool_order(env: Mapping[str, str]) -> tuple[str, ...]:
    """Rank the shortcut backends for the session described by ``env``.

    Under Wayland ``xdotool`` still runs and still exits 0, but the compositor
    only routes its synthetic events to XWayland clients — so a shortcut aimed
    at a native Wayland application is silently dropped while the log claims
    success, and anything reading ``/dev/input`` never sees the key at all.
    ``ydotool`` writes to ``/dev/uinput``, below the display server, and
    therefore reaches every client; it goes first there. On X11 ``xdotool``
    leads because it consumes keysym names directly, with no translation step
    that could lose fidelity.
    """
    if is_wayland_session(env):
        return ("ydotool", "xdotool", "wtype")
    return ("xdotool", "ydotool", "wtype")


def shortcut_argv(tool: str, keys: str) -> list[str] | None:
    """Build the argv for ``tool``, or ``None`` if it cannot express ``keys``."""
    if tool == "xdotool":
        return ["xdotool", "key", keys]
    if tool == "ydotool":
        # ydotool speaks raw evdev codes, never keysym names.
        codes = translate_shortcut(keys)
        return ["ydotool", "key", *codes] if codes else None
    if tool == "wtype":
        # wtype uses different syntax; best-effort approximation.
        return ["wtype", "-M", keys]
    return None
