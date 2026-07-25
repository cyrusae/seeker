"""Off-screen notifications for events nobody's watching a terminal for —
overnight cron runs, a backgrounded/clustered server, anything not sitting in
a visible tab.

Two independent channels, both optional (unset = silent no-op):
  - Discord webhook (DISCORD_WEBHOOK_URL): human-readable pings — pipeline
    finished (with stats) or crashed.
  - healthchecks.io: dead-man's-switch heartbeats, one check per scheduled
    job (HEALTHCHECKS_LIVENESS_URL, HEALTHCHECKS_PIPELINE_URL) so a missed
    nightly pipeline run and a missed liveness probe alert distinctly rather
    than one job's silence masking the other's. Ping on success;
    healthchecks.io itself pages you if a ping doesn't land within its grace
    window, which is the one failure mode a "notify on error" call can never
    catch — the job not running at all. ok=False also pings /fail for a
    faster signal on an actual exception, on top of that coverage.

Falls back to a macOS notification when neither is configured, so local dev
keeps behaving the way it did before either channel was wired up.
"""
import platform
import subprocess

import httpx

from .config import settings

_HTTP_TIMEOUT = 10.0


def _osa_quote(s: str) -> str:
    """Escape for embedding in an AppleScript double-quoted string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _notify_mac(title: str, message: str) -> None:
    if platform.system() != "Darwin":
        return
    script = (
        f'display notification "{_osa_quote(message[:500])}" '
        f'with title "{_osa_quote(title)}" sound name "Basso"'
    )
    try:
        subprocess.run(["osascript", "-e", script], timeout=5,
                        capture_output=True, check=False)
    except Exception:
        pass  # notification is a nicety — never let it break the caller


def notify(title: str, message: str) -> None:
    """Human-facing notification: pipeline finished, pipeline crashed, etc."""
    if settings.discord_webhook_url:
        try:
            httpx.post(settings.discord_webhook_url,
                       json={"content": f"**{title}**\n{message[:1800]}"},
                       timeout=_HTTP_TIMEOUT)
        except Exception:
            pass  # never let a dead webhook break the caller
        return
    _notify_mac(title, message)


def _ping(url: str, ok: bool) -> None:
    if not url:
        return
    if not ok:
        url = url.rstrip("/") + "/fail"
    try:
        httpx.get(url, timeout=_HTTP_TIMEOUT)
    except Exception:
        pass


def ping_liveness(ok: bool = True) -> None:
    """Heartbeat for the daily liveness probe. Call once per run: ok=True on a
    clean pass, ok=False on an exception. No-ops if HEALTHCHECKS_LIVENESS_URL
    isn't set."""
    _ping(settings.healthchecks_liveness_url, ok)


def ping_pipeline(ok: bool = True) -> None:
    """Heartbeat for the full nightly pipeline (run_full_cycle). Call once per
    run: ok=True whether it finished cleanly or was stopped early by request
    (both mean it ran), ok=False on an exception. No-ops if
    HEALTHCHECKS_PIPELINE_URL isn't set."""
    _ping(settings.healthchecks_pipeline_url, ok)
