"""Compatibility entrypoint for the refactored dashboard suite."""

from __future__ import annotations

from dashboard.app import main as _app_main


def main() -> None:
    """Delegate to the consolidated dashboard index."""

    _app_main()


if __name__ == "__main__":
    main()


__all__ = ["main"]
