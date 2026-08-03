"""torchaudio API calls that were REMOVED upstream must be feature-guarded.

`torchaudio.set_audio_backend()` was deprecated through 2.x and deleted in
2.9. Calling it unguarded raises AttributeError at *import* of
``backend/main.py`` — the backend never starts, and it fails before any route
exists to report why, so the user sees a dead app rather than an error.

This is not theoretical: the Windows AMD/ROCm path installs torchaudio 2.9
(no 2.8 ROCm build exists for Windows on any channel), so every opted-in AMD
user hit it. Source-level check, in the same spirit as the other repo-wide
rules (``test_no_hardcoded_cjk``, ``test_no_literal_borders``): the failure
mode is "import-time crash on a torch build CI does not install", which no
unit test running against the pinned torch can catch.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# API name → the version that removed it. Extend as upstream drops more.
REMOVED_IN = {
    "set_audio_backend": "2.9",
    "get_audio_backend": "2.9",
    "list_audio_backends": "2.9",
}

_PY_FILES = [
    p for p in (REPO / "backend").rglob("*.py")
    if "__pycache__" not in p.parts and ".venv" not in p.parts
]


@pytest.mark.parametrize("api,gone_in", sorted(REMOVED_IN.items()))
def test_removed_torchaudio_apis_are_guarded(api, gone_in):
    call = re.compile(rf"(?<!\w)torchaudio\.{api}\s*\(")
    offenders = []
    for path in _PY_FILES:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not call.search(text):
            continue
        # A guard must be visible in the file: either an explicit capability
        # check or a swallowed AttributeError around the call site.
        guarded = (
            f'hasattr(torchaudio, "{api}")' in text
            or f"hasattr(torchaudio, '{api}')" in text
            or "AttributeError" in text
        )
        if not guarded:
            for i, line in enumerate(text.splitlines(), 1):
                if call.search(line):
                    offenders.append(f"{path.relative_to(REPO)}:{i}")
    assert not offenders, (
        f"torchaudio.{api}() was removed in torchaudio {gone_in}; an unguarded "
        f"call is an import-time AttributeError that stops the backend from "
        f"starting at all. Wrap it in `if hasattr(torchaudio, \"{api}\"):`. "
        f"Offenders: {offenders}"
    )
