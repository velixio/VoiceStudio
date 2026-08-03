# Phase 5 — AMD GPU Test Results (Windows / ROCm)

**Date:** 2026-08-03
**Branch:** `feat/amd-windows-rocm-support`
**Status:** Windows AMD validated end-to-end. Linux/Docker untested here (no hardware).

---

## 1. Test host

| Property | Value |
|---|---|
| GPU (discrete) | **AMD Radeon RX 6800M** — Navi 22, PCI `1002:73DF` |
| GPU arch | **gfx1031** (RDNA2), 20 CUs, 11.98 GB |
| GPU (integrated) | Radeon Graphics (Cezanne, `1002:1638`) — not used |
| Driver | `30.0.13002.13003`, dated **2021-12-15** (Adrenalin ~21.12) |
| OS | Windows 11 Pro 10.0.22631 |
| Python | 3.12.7 |
| Pre-existing | HIP SDK 6.2 at `C:\Program Files\AMD\ROCm\6.2\` |

**This host is *doubly* outside AMD's documented Windows support matrix**: RDNA2 is
absent from it (only gfx1100/1101/1200/1201 are listed), and the driver is four
years older than the stated Adrenalin 26.2.2+ requirement. It works anyway. See §6.

---

## 2. Stack under test

Installed into an isolated venv — **the repo's own `.venv` was never touched**.

```
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1031]==2.9.1+rocm7.13.0"
```

Resolved to:

| Package | Version |
|---|---|
| `torch` | **2.9.1+rocm7.13.0** |
| `amd-torch-device-gfx1031` | 2.9.1+rocm7.13.0 |
| `rocm-sdk-core` / `-libraries` / `-device-gfx1031` | 7.13.0 |
| `torchaudio` | 2.9.0+rocm7.13.0 |
| `transformers` | **5.3.0** (repo's locked version — see §5) |
| `torch.version.hip` | `7.13.99004-3309c611` |

`repo.amd.com/rocm/whl-multi-arch/` **is a real PEP 503 index** (unlike
`repo.radeon.com`, which is a flat directory listing requiring full wheel URLs).
Arch selection is via the `[device-gfxNNNN]` extra.

---

## 3. Capability probe — 11/11 passed

Escalating from metadata to real kernels; each stage independently guarded.

| # | Stage | Result |
|---|---|---|
| 1 | Build metadata | ROCm build confirmed, `hip 7.13.99004` |
| 2 | Device visibility | `is_available()` True, RX 6800M, gfx1031, 11.85 GB free |
| 3 | **Repo's own arch gate** | `arch_unsupported()` → **None**; `detect_host_caps()` → **`family="rocm"`**, zero notes |
| 4 | Kernel launch (add) | correct |
| 5 | matmul fp32 (rocBLAS) | max err **4.77e-05** vs CPU |
| 6 | matmul fp16 | correct within fp16 tolerance |
| 7 | Conv1d (MIOpen) | correct |
| 8 | **ConvTranspose1d** | correct, max err **5.96e-07** |
| 9 | STFT→iSTFT (rocFFT) | round-trip err **7.15e-07** |
| 10 | SDPA | works; falls back to math backend |
| 11 | Sustained load | 50× 1024³ matmul+tanh in 0.02 s ≈ **5.4 TFLOPS** fp32 |

### 3.1 The ConvTranspose1d result matters most

[TheRock#4681](https://github.com/ROCm/TheRock/issues/4681) (open) reports
`ConvTranspose1d` **hanging the GPU** on gfx110x. That op is the backbone of
HiFi-GAN/BigVGAN-class vocoders, so Phase 1 named it the single biggest threat to
claiming ROCm for any TTS engine.

**It does not reproduce on gfx1031** — the op is numerically correct to 6e-07 and
never hung across many runs. The bug appears to be RDNA3-specific.
**Still UNVERIFIED on gfx110x**; do not claim RDNA3 on the strength of this result.

---

## 4. End-to-end model load + synthesis

Real model, exactly as `backend/services/model_manager.py` loads it:
`OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=..., dtype=..., load_asr=False)`.

| Measurement | GPU (gfx1031, fp16) | CPU (fp32) |
|---|---|---|
| Model load (warm HF cache) | **8.2 s** | 6.1 s |
| Synthesis — **cold** (MIOpen tuning) | 5.78 s → RTF **1.357×** | — |
| Synthesis — **warm** | **2.04 s** → RTF **0.474×** | 940.24 s → RTF **218.153×** |
| VRAM (weights / peak) | 1.89 GB / 2.05 GB | — |

- **GPU is ~461× faster than CPU** on this host.
- **Warm GPU runs at 2.1× faster than realtime** (RTF 0.474×).
- Output verified as real audio: finite, peak 0.50, rms 0.067, not silence.

### 4.1 Cold-start penalty is real but one-time

The first generation is **~2.8× slower** than subsequent ones, because MIOpen
benchmarks every unique convolution shape on first encounter and caches the winner
to `~/.cache/miopen`. Consequences:

- Do not benchmark on a first run — it measures tuning, not inference.
- A user's very first synthesis after install will feel slow. This is expected.
- **An earlier draft of this document reported RTF 2.99× as the GPU result. That was a
  cold-run measurement and was wrong.** The warm figure (0.474×) is the correct one.

### 4.2 MIOpen `workspace=0` warnings appear but do not dominate

Generation emits, repeatedly:

```
MIOpen(HIP): Warning [IsEnoughWorkspace] [EvaluateInvokers]
  Solver <GemmFwdRest>, workspace required: 11927552, provided ptr: 0x0 size: 0
```

This is [TheRock#3077](https://github.com/ROCm/TheRock/issues/3077) — PyTorch-ROCm on
Windows passes a null workspace, so MIOpen cannot use its best solver. That issue
reports a Qwen3-TTS decoder at ~12.6× RTF. **We do not see anything like that
severity**: warm RTF is 0.474×. It is plausible the 12.6× report was itself a
cold-start measurement. The warnings are cosmetically alarming and worth
suppressing in logs, but they are not a blocker on this host.

Also benign, seen once at load: `CK grouped conv library not found for device
gfx1031` and `xnack 'Off' was requested for a processor that does not support it`.

---

## 5. The one genuine Windows blocker — and its patch

**Windows ROCm PyTorch ships without `torch.distributed`:**

```
torch.distributed.is_available()      -> False        # torch reports this correctly
hasattr(torch._C, "_distributed_c10d") -> False
```

`transformers` ≥ (some version after 5.3.0) has `transformers/distributed/fsdp.py`,
which at line 34 does `from torch.distributed._composable.fsdp import fully_shard`
**unconditionally**. On Windows ROCm that raises, and the failure surfaces far from
its cause as:

```
ModuleNotFoundError: Could not import module 'AutoFeatureExtractor'
```

### Current exposure

| Version | `distributed/fsdp.py` | Windows ROCm |
|---|---|---|
| **5.3.0** (locked in `uv.lock`) | does not exist | ✅ works |
| **5.14.1** (latest) | present, unguarded import | ❌ breaks |

The repo is safe **only because `uv.lock` pins 5.3.0**. `pyproject.toml` declares
`transformers>=5.3.0`, so **a routine lockfile refresh would silently break Windows
AMD** with an error message that names neither `torch.distributed` nor ROCm.

### Patch

Minimal and upstream-friendly — guard on the API torch already provides:

```python
# transformers/distributed/fsdp.py
if torch.distributed.is_available():
    from torch.distributed._composable.fsdp import fully_shard
```

Defensively, this repo should also carry an upper bound on `transformers` until
that lands upstream, plus a regression test asserting the import survives when
`torch.distributed.is_available()` is False.

---

## 6. Findings that contradict the Phase 1 research

`research.md` was written from documentation. Hardware disagrees with it in four
places. Corrections belong in that document.

| Claim in `research.md` | Reality on this host |
|---|---|
| gfx1031 is in **no** PyTorch ROCm wheel, any version | True of `download.pytorch.org`; **false** of `repo.amd.com`, which ships native gfx1031 |
| Windows AMD needs RDNA3/RDNA4 | **RDNA2 works** — AMD's matrix states what is *supported*, not what *functions* |
| Windows AMD needs Adrenalin 26.2.2+ | **A 2021 driver works** with the TheRock multi-arch stack |
| `ConvTranspose1d` may hang the GPU (TheRock#4681) | **Does not reproduce** on gfx1031 |

The TheRock wheel's compiled arch list is far wider than the official one:

```
gfx1010 1011 1012 | 1030 1031 1032 1033 1034 1035 1036
gfx1100 1101 1102 1103 | 1150 1151 1152 | 1200 1201
```

That is **all of RDNA1 and RDNA2** — architectures `download.pytorch.org` has never
shipped. `HSA_OVERRIDE_GFX_VERSION` is therefore unnecessary here, which is
fortunate: that variable is inoperative on Windows.

---

## 7. Routing validation — the proposed fix, on real hardware

`resolve_routing()` evaluated against the live `HostCaps` (`family="rocm"`):

| `gpu_compat` | Status | Effective device | User-visible notice |
|---|---|---|---|
| `("cuda","mps","cpu")` — **current** OmniVoice | `cpu_fallback` | **cpu** | ⚠️ *"declares CUDA only; ROCm not in its compat set"* — **on every generation** |
| `("cuda","rocm","mps","cpu")` — **proposed** | `accelerated` | **rocm** | none |
| `("cuda","cpu")` — current WhisperX | `cpu_fallback` | cpu | ⚠️ same warning |
| `("cpu",)` — KittenTTS | `cpu_only` | cpu | none (correct, unchanged) |

**The routing and detection layers need no changes whatsoever.** Adding one string
to a tuple flips a false CPU warning into correct GPU routing — on a machine that
is, measurably, 461× faster on the GPU.

---

## 8. Not yet tested

Recorded honestly rather than assumed.

- **Linux ROCm** (`whl/rocm6.4` + torch 2.8.0) — no Linux AMD host available.
- **Docker `:rocm` image** — same.
- **RDNA3 / gfx110x** — the ConvTranspose1d hang (TheRock#4681) is unresolved there.
- **CTranslate2 ROCm wheels** (ASR path) — not exercised; WhisperX/Faster-Whisper untested on AMD.
- **Long-form / batch synthesis** — only single short utterances measured.
- **The repo's full test suite** under ROCm torch — the test venv is deliberately minimal.
- **VRAM pressure** — peak was 2.05 GB against 11.98 GB available; larger engines untested.

---

## 9. Reproduction

```powershell
# 1. isolated venv (Python 3.12)
python -m venv rocmtest

# 2. ROCm torch with native gfx1031 kernels
.\rocmtest\Scripts\python.exe -m pip install `
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  "torch[device-gfx1031]==2.9.1+rocm7.13.0"

# 3. verify
.\rocmtest\Scripts\python.exe -c "import torch; print(torch.version.hip, torch.cuda.get_device_properties(0).gcnArchName)"
# -> 7.13.99004-3309c611 gfx1031
```

Substitute your own `gfx` target (`hipInfo.exe` reports `gcnArchName`, or read
`torch.cuda.get_device_properties(0).gcnArchName`).

**Watching GPU usage on Windows:** Task Manager's default GPU graphs (3D / Copy /
Video) do **not** show ROCm work. HIP dispatches to the **Compute** engine — select
it from a graph dropdown, or it will look like the GPU is idle while it is at 100%.
Measured during a sustained load: `compute 1` at 105.7%, `3d` at 5.4%.
