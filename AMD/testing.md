# Phase 5 — AMD GPU Test Results (Windows / ROCm)

**Date:** 2026-08-03 (superseded 2026-08-04 — see the stability warning below)
**Branch:** `feat/amd-windows-rocm-support`
**Status:** ⛔ **RDNA2-on-Windows is NOT usable.** Functionally correct, but it
hard-resets the host under sustained load. Linux/Docker untested here (no hardware).

---

## ⛔ 0. STABILITY WARNING — RDNA2 (gfx1031) on Windows hard-resets the machine

**Do not use, and do not recommend, the Windows ROCm path on RDNA2.**

Everything measured in §3 and §4 below is real and reproducible: the model
loads, synthesis is numerically correct, the Whisper round-trip is 0% WER, and
the GPU is ~464× faster than CPU. **That was all measured in short bursts.**

Under *sustained* real use — generating from the app rather than a scripted
one-shot — the host **powered off and rebooted, four times**. Windows logged:

| Signal | Result |
|---|---|
| Unexpected shutdown (event 6008) | **3 logged** |
| BugCheck (event 1001) | **none** |
| Kernel-Power (event 41) | **none** |
| WHEA / display-driver events | **none** |
| Crash dump | **none written** |

A driver fault writes a bugcheck and a dump. Nothing was written at all, which
means the machine lost power instantly rather than blue-screening. On a *mobile*
RX 6800M that is the signature of **power delivery or an over-current/thermal
protection trip** under sustained compute — a hardware limit, not something any
amount of application code can catch or recover from.

**This is exactly the risk AMD's support matrix encodes.** AMD lists only
RDNA3/RDNA4 for Windows ROCm; RDNA2 is absent. The wheels exist on TheRock's
multi-arch channel and they *run*, which is what made this look supportable. It
is not. An omission from a vendor support matrix is a claim about stability
under load, and short benchmarks cannot falsify it.

**Correction to the original write-up:** this document previously said Windows
AMD was "validated end-to-end". That over-generalised from passing short tests.
The honest statement is: *correct, fast, and unstable enough to reset the host.*

Contributing factor worth noting for anyone retesting: the test machine's
Adrenalin driver was from **2021-12-15**, four years older than the 26.2.2+ that
every ROCm-on-Windows requirement specifies. That alone disqualifies the
configuration; it does not explain a power cut, but it must be fixed before any
retest is meaningful.

**If you are retesting this, do it on RDNA3/RDNA4 with a current driver, on AC
power, watching temperatures — and expect to lose the machine mid-run.**

---

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

## 4.3 Audio correctness — A/B verified against CPU

"Not silent and finite" is a weak bar, so the same sentence was synthesised on
both devices and the output verified by **transcribing it back** — if Whisper
reads the sentence we asked for, synthesis is genuinely correct.

Text: `"The quick brown fox jumps over the lazy dog."`

| | GPU (ROCm fp16) | CPU (fp32) |
|---|---|---|
| Generation time | **1.64 s** | **761.10 s** |
| RTF | **0.586×** | 271.822× |
| Duration | 2.80 s | 2.80 s |
| Peak / RMS | 0.500 / 0.0646 | 0.500 / 0.0628 |
| Clipped / NaN / DC | 0 / 0 / −0.00000 | 0 / 0 / +0.00004 |
| Voiced frames | 71.4% | 67.0% |
| Spectral centroid | 1532 Hz | 1692 Hz |
| **Whisper transcript** | *exact match* | *exact match* |
| **WER** | **0.0% PASS** | **0.0% PASS** |

**GPU is 464× faster** on identical work, at identical output length.

Acoustic comparison of the two renderings:

```
log-spectrogram correlation : +0.9026
RMS envelope correlation    : +0.9620
crest factor                : GPU 17.8 dB / CPU 18.0 dB
waveform correlation        : +0.0294
```

The low *waveform* correlation is expected and is **not** a defect: OmniVoice is a
diffusion model, so the two devices draw different RNG streams and produce
different realisations of the same utterance. Sample-wise they diverge; spectrally,
in envelope, in duration and in transcript they agree. That is the signature of two
valid renderings, not a broken one. A bitwise GPU/CPU comparison would be the wrong
test for this model class.

Artifacts live in `AMD/samples/` (`sample_cuda.wav`, `sample_cpu.wav`). They are
**left untracked deliberately** — binary output, regenerable via §9.

### Bonus: ASR verified on ROCm too

The Whisper model used for the check ran **on the AMD GPU** via the `transformers`
pipeline. That independently validates the `PyTorchWhisperBackend` path
(`asr_backend.py:1193`), whose comment currently reads *"ROCm-via-HIP would also
work but is left unclaimed pending verification."* It is now verified on RDNA2.

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
