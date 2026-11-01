"""Local user and session identifiers for the CLI demonstration mode.

This module deliberately has no authentication concerns.  It only creates
file-system-safe identifiers and derives a per-user/per-session storage layout.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def validate_local_id(value: str, label: str) -> str:
    """Reject path-like identifiers before they are used in local paths."""
    normalized = (value or "").strip()
    if not _SAFE_ID.fullmatch(normalized):
        raise ValueError(f"{label} 只能包含字母、数字、下划线和连字符，长度 1-128")
    return normalized


def new_user_id() -> str:
    """Create a stable identifier to show once and reuse on later CLI runs."""
    return str(uuid.uuid4())


def new_session_id() -> str:
    """Create a human-sortable session id with a random collision-resistant suffix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class LocalSessionContext:
    user_id: str
    session_id: str
    user_dir: Path
    memory_dir: Path
    session_dir: Path
    session_path: Path
    context_archive_dir: Path


def build_local_session_context(
    user_id: str,
    session_id: str | None = None,
    storage_root: str | Path = "app/sessions/users",
) -> LocalSessionContext:
    """Derive isolated local paths; no files are created by this function."""
    safe_user_id = validate_local_id(user_id, "user_id")
    safe_session_id = validate_local_id(session_id or new_session_id(), "session_id")
    user_dir = Path(storage_root) / safe_user_id
    session_dir = user_dir / "sessions" / safe_session_id
    return LocalSessionContext(
        user_id=safe_user_id,
        session_id=safe_session_id,
        user_dir=user_dir,
        memory_dir=user_dir / "memory",
        session_dir=session_dir,
        session_path=session_dir / "state.json",
        context_archive_dir=session_dir / "context",
    )
