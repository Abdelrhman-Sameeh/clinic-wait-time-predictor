"""Shared Chroma client and path resolution for the clinic application."""

import os
from pathlib import Path
from threading import RLock
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "chroma_db"
_CLIENT: Any | None = None
_CLIENT_PATH: Path | None = None
_CLIENT_LOCK = RLock()
INDEX_LOCK = RLock()


def get_chroma_path(override: str | Path | None = None) -> Path:
    """Resolve Chroma storage to one absolute path, independent of CWD."""
    configured = (
        override
        if override is not None
        else os.getenv("CHROMA_PATH") or os.getenv("CHROMA_DIR")
    )
    if configured is None:
        path = DEFAULT_CHROMA_PATH
    else:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    return path.resolve()


def get_chroma_client(path: str | Path | None = None) -> Any:
    """Return the one process-wide PersistentClient for the configured path."""
    global _CLIENT, _CLIENT_PATH
    resolved_path = get_chroma_path(path)
    with _CLIENT_LOCK:
        if _CLIENT is None or _CLIENT_PATH != resolved_path:
            _CLIENT = chromadb.PersistentClient(path=str(resolved_path))
            _CLIENT_PATH = resolved_path
        return _CLIENT
