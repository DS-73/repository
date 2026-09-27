"""Entrypoint for the API service.

Run with ``python -m suretyseven.main`` (or the ``suretyseven-api`` console
script, or ``uvicorn suretyseven.main:app``).  ``app`` is exposed at module level
so ASGI servers can import it by path.
"""

from __future__ import annotations

import os

import uvicorn

from suretyseven.api import create_app
from suretyseven.config import get_settings

settings = get_settings()

#: ASGI application object (``uvicorn suretyseven.main:app``).
app = create_app(settings)


def main() -> None:
    """Run the API with uvicorn (log config left to the app)."""
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)


if __name__ == "__main__":
    main()
