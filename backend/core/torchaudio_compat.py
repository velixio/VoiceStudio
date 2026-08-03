"""Keep ``torchaudio.save`` / ``torchaudio.load`` working on torchaudio 2.9+.

**The problem.** torchaudio 2.9 rerouted both I/O entry points through
TorchCodec and made it a *hard* requirement. On a 2.9 install without the
separately-published ``torchcodec`` wheel, every read and every write raises::

    ImportError: TorchCodec is required for save_with_torchcodec.
    ImportError: TorchCodec is required for load_with_torchcodec.

For this app that means synthesis runs to completion on the GPU and then dies
writing the file — the user sees "Couldn't synthesize audio" — and voice
cloning cannot read a reference clip at all.

**Why not just depend on torchcodec.** It needs FFmpeg *shared libraries*
(libavcodec/libavformat), which the bundled ``imageio-ffmpeg`` (a standalone
binary) does not provide. Installing the wheel on Windows gets you as far as::

    RuntimeError: Could not load libtorchcodec. Likely causes:
      1. FFmpeg is not properly installed in your environment...

so it would trade a clean failure for a confusing one.

**Why a shim rather than fixing call sites.** ``torchaudio.load`` alone has 17
in-tree call sites plus uses inside the vendored ``omnivoice`` package, and the
right fallback is identical at every one. Patching the two functions once keeps
the fix in a single auditable place — the same approach ``hf_progress`` already
takes with ``huggingface_hub.utils.tqdm``.

``soundfile`` (libsndfile) is already a hard dependency — it is precisely what
torchaudio's own soundfile backend used before 2.9 — so this adds no install
surface and writes byte-identical WAV/FLAC.

Idempotent, and a complete no-op on torchaudio < 2.9 where the native paths
work: the wrappers only engage on the TorchCodec error.
"""
from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger("omnivoice.torchaudio_compat")

_INSTALLED = False

# soundfile subtype per (format, bits_per_sample) for the formats this app
# writes. Anything unmapped lets soundfile choose its own default.
_SF_SUBTYPE = {
    ("wav", 16): "PCM_16", ("wav", 24): "PCM_24", ("wav", 32): "FLOAT",
    ("flac", 16): "PCM_16", ("flac", 24): "PCM_24",
}


def _is_torchcodec_problem(exc: BaseException) -> bool:
    """Is `exc` torchaudio failing because TorchCodec is missing or unloadable?

    Walks the cause/context chain and matches on the message, because
    torchaudio surfaces this as ImportError from some paths and RuntimeError
    ("Could not load libtorchcodec") from others.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if "torchcodec" in str(cur).lower():
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def install_torchaudio_fallbacks() -> bool:
    """Wrap torchaudio's I/O with libsndfile fallbacks. Returns True if applied.

    Safe to call more than once. Never raises: a failure here must not stop
    the backend from starting, since the native path may well be fine.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
    except Exception as exc:  # torchaudio/soundfile absent — nothing to patch
        logger.debug("torchaudio compat shim skipped: %s", exc)
        return False

    _native_save = torchaudio.save
    _native_load = torchaudio.load

    def _save(uri: Any, src: Any, sample_rate: int, *args: Any, **kwargs: Any) -> Any:
        try:
            return _native_save(uri, src, sample_rate, *args, **kwargs)
        except Exception as exc:
            if not _is_torchcodec_problem(exc):
                raise
            # A partial write may already sit in the buffer; rewind so the
            # fallback starts at byte 0. (Paths are simply overwritten.)
            if hasattr(uri, "seek") and hasattr(uri, "truncate"):
                try:
                    uri.seek(0)
                    uri.truncate(0)
                except (OSError, io.UnsupportedOperation):
                    pass
            fmt = str(kwargs.get("format") or "wav").lower()
            bits = kwargs.get("bits_per_sample") or 16
            subtype = _SF_SUBTYPE.get((fmt, bits))
            if subtype is None and fmt in ("ogg", "mp3"):
                subtype = {"ogg": "VORBIS", "mp3": "MPEG_LAYER_III"}[fmt]
            data = src.detach().cpu().numpy() if torch.is_tensor(src) else np.asarray(src)
            if data.ndim == 1:
                data = data[None, :]
            # torchaudio hands us (channels, samples); libsndfile wants
            # (samples, channels).
            data = data.T
            if hasattr(uri, "write"):
                sf.write(uri, data, sample_rate, subtype=subtype, format=fmt.upper())
            else:
                sf.write(uri, data, sample_rate, subtype=subtype)
            logger.debug("torchaudio.save fell back to soundfile (%s)", fmt)
            return None

    def _load(uri: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return _native_load(uri, *args, **kwargs)
        except Exception as exc:
            if not _is_torchcodec_problem(exc):
                raise
            if hasattr(uri, "seek"):
                try:
                    uri.seek(0)
                except (OSError, io.UnsupportedOperation):
                    pass
            data, sr = sf.read(uri, dtype="float32", always_2d=True)
            # libsndfile returns (samples, channels); torchaudio's contract is
            # (channels, samples).
            wav = torch.from_numpy(data.T.copy())
            logger.debug("torchaudio.load fell back to soundfile")
            return wav, sr

    torchaudio.save = _save  # type: ignore[assignment]
    torchaudio.load = _load  # type: ignore[assignment]
    _INSTALLED = True
    logger.debug("torchaudio TorchCodec fallbacks installed")
    return True


__all__ = ["install_torchaudio_fallbacks", "_is_torchcodec_problem"]
