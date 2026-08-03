"""torchaudio 2.9 made TorchCodec a hard requirement for save AND load.

Without that wheel every audio read and write raises, so synthesis completes on
the GPU and then dies writing the file ("Couldn't synthesize audio"), and voice
cloning cannot read a reference clip at all. Depending on ``torchcodec`` is not
a fix: it needs FFmpeg *shared* libraries that the bundled imageio-ffmpeg
binary does not provide, so on Windows it fails at import with "Could not load
libtorchcodec" instead.

``core.torchaudio_compat`` therefore falls back to libsndfile (already a hard
dependency — it is what torchaudio's own soundfile backend used before 2.9).

The round-trip test passes on both sides of the change: on torchaudio < 2.9 it
exercises the native path, on 2.9-without-TorchCodec the fallback. That is the
point — the shim must be invisible.
"""
from __future__ import annotations

import io
import sys

import pytest

sys.path.insert(0, "backend")


def test_detects_torchcodec_failures_in_either_form():
    """torchaudio reports this as ImportError from some paths and RuntimeError
    from others, sometimes wrapped — all three must be recognised."""
    from core.torchaudio_compat import _is_torchcodec_problem

    direct = ImportError(
        "TorchCodec is required for save_with_torchcodec. "
        "Please install torchcodec to use this function."
    )
    unloadable = RuntimeError("Could not load libtorchcodec. Likely causes: ...")
    assert _is_torchcodec_problem(direct)
    assert _is_torchcodec_problem(unloadable)

    # Wrapped in a cause chain (how it surfaces through audio_io's handler).
    try:
        try:
            raise direct
        except ImportError as inner:
            raise RuntimeError("Writing the audio file failed") from inner
    except RuntimeError as wrapped:
        assert _is_torchcodec_problem(wrapped)


def test_unrelated_errors_are_not_swallowed():
    """A full disk or a bad dtype must still surface as itself — the fallback
    exists for one specific upstream breakage, not as a blanket retry."""
    from core.torchaudio_compat import _is_torchcodec_problem

    for exc in (
        OSError("[Errno 28] No space left on device"),
        RuntimeError("Input tensor has to be 2D"),
        ValueError("Expected float32"),
        PermissionError("Access is denied"),
    ):
        assert not _is_torchcodec_problem(exc), exc


@pytest.mark.parametrize(
    "fmt,bits", [("wav", 16), ("wav", 32), ("flac", 16)]
)
def test_save_load_round_trip(tmp_path, fmt, bits):
    """Audio survives a write/read round-trip whichever backend handles it."""
    import torch
    import torchaudio
    from core.torchaudio_compat import install_torchaudio_fallbacks

    install_torchaudio_fallbacks()

    src = torch.zeros(1, 8000)
    src[0, ::50] = 0.4
    path = tmp_path / f"probe.{fmt}"
    torchaudio.save(
        str(path), src, 16000, format=fmt,
        encoding="PCM_F" if bits == 32 else "PCM_S", bits_per_sample=bits,
    )
    assert path.stat().st_size > 0

    back, sr = torchaudio.load(str(path))
    assert sr == 16000
    assert back.shape == src.shape
    # 16-bit quantisation is ~3e-05; anything larger means the samples were
    # mangled (wrong dtype, wrong channel order, silent file).
    assert (back - src).abs().max().item() < 1e-3


def test_in_memory_round_trip():
    """BytesIO destinations work too — the OpenAI-compat route streams audio
    from memory rather than a path."""
    import torch
    import torchaudio
    from core.torchaudio_compat import install_torchaudio_fallbacks

    install_torchaudio_fallbacks()

    src = torch.zeros(1, 4000)
    src[0, ::40] = 0.25
    buf = io.BytesIO()
    torchaudio.save(buf, src, 16000, format="wav", bits_per_sample=16)
    assert buf.getbuffer().nbytes > 0

    buf.seek(0)
    back, sr = torchaudio.load(buf)
    assert sr == 16000
    assert back.shape == src.shape


def test_install_is_idempotent():
    """main.py and audio_io both install it; double-wrapping would stack a
    fallback on a fallback and bury the original error."""
    from core import torchaudio_compat

    assert torchaudio_compat.install_torchaudio_fallbacks() is True
    import torchaudio
    first = torchaudio.save
    assert torchaudio_compat.install_torchaudio_fallbacks() is True
    assert torchaudio.save is first, "shim re-wrapped an already-wrapped save"
