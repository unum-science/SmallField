"""Rich-backed logger mirroring the C++ ``smallfield::logger_t`` surface.

Every script constructs a local ``Logger`` (or uses the module-level
``log`` singleton where the caller is a one-off) rather than patching a
process-wide global — mirrors the "owned on context" pattern from the C++
side.

Output lines are prefixed with ``[YYYY-MM-DD HH:MM:SS | +HH:MM:SS]`` — wall
clock plus elapsed from the logger's ``start_time``. Colors are handled by
`rich`; ``NO_COLOR`` and non-TTY sinks are honored automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.theme import Theme

_theme = Theme(
    {
        "info": "default",
        "warn": "yellow",
        "error": "bold red",
        "trace": "cyan",
        "trace2": "grey50",
        "rate": "green",
        "dim": "grey42",
    }
)


def _format_elapsed(seconds: float) -> str:
    """Render ``seconds`` as ``HH:MM:SS``, with hours unbounded so a run past a day keeps counting up."""
    total = int(seconds)
    return f"{total // 3600:02}:{(total // 60) % 60:02}:{total % 60:02}"


def thousands(count: int) -> str:
    """Format an integer with comma grouping: ``1_234_567 → '1,234,567'``."""
    return f"{count:,}"


@dataclass
class Logger:
    """Leveled, colored, timestamped logger — one instance per logical caller.

    Tune verbosity via ``log.verbose_level = 1`` before emitting traces.
    """

    verbose_level: int = 0
    start_time: float = field(default_factory=time.monotonic)
    console: Console = field(default_factory=lambda: Console(theme=_theme, stderr=False))
    err_console: Console = field(default_factory=lambda: Console(theme=_theme, stderr=True))

    def timestamp_prefix(self) -> str:
        """The ``[wall clock | +elapsed]`` prefix every line carries, elapsed measured from `start_time`."""
        wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = _format_elapsed(time.monotonic() - self.start_time)
        return f"[{wall} | +{elapsed}]"

    def _emit(self, console: Console, style: str, message: str, *args: Any) -> None:
        """Print the prefixed ``message`` to ``console`` in the `rich` theme entry named by ``style``.

        ``args`` are applied with ``str.format`` and only when non-empty, so a lone message keeps its braces.
        """
        text = message.format(*args) if args else message
        console.print(f"{self.timestamp_prefix()} [{style}]{text}[/]")

    def info(self, message: str, *args: Any) -> None:
        """Emit ``message`` formatted with ``args`` at info level, on stdout."""
        self._emit(self.console, "info", message, *args)

    def warn(self, message: str, *args: Any) -> None:
        """Emit ``message`` formatted with ``args`` at warning level, on stderr."""
        self._emit(self.err_console, "warn", message, *args)

    def error(self, message: str, *args: Any) -> None:
        """Emit ``message`` formatted with ``args`` at error level, on stderr."""
        self._emit(self.err_console, "error", message, *args)

    def trace(self, message: str, *args: Any) -> None:
        """Emit ``message`` formatted with ``args`` on stdout, only once `verbose_level` reaches 1."""
        if self.verbose_level >= 1:
            self._emit(self.console, "trace", message, *args)

    def trace2(self, message: str, *args: Any) -> None:
        """Emit ``message`` formatted with ``args`` on stdout, only once `verbose_level` reaches 2."""
        if self.verbose_level >= 2:
            self._emit(self.console, "trace2", message, *args)


#: Module-level logger for one-shot scripts; library code constructs a local ``Logger`` instead.
log = Logger()

__all__ = ["Logger", "log", "thousands"]
