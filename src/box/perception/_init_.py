"""Compatibility shim. Prefer importing from package __init__.py."""

try:
    from .__init__ import *  # noqa: F401,F403
except Exception:
    pass
