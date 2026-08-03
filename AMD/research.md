# Phase 1 — AMD GPU (ROCm/HIP) Dependency & Capability Research

**Date:** 2026-08-03
**Scope:** `goal.md` Phase 1 — complete dependency audit prior to any code change.
**Repo state:** `main` @ `5f39f9ff`, app version `0.4.2`.

> ⚠️ **Superseded in places by hardware testing — see [`testing.md`](testing.md).**
> This document was written from documentation and upstream sources. Subsequent
> testing on a real Windows AMD host (RX 6800M / gfx1031) contradicted four of its
> conclusions: gfx1031 *is* shipped by AMD's own wheel index, RDNA2 *does* work on
> Windows, a 2021 driver *is* sufficient, and the `ConvTranspose1d` hang does *not*
> reproduce. Individual corrections are marked **[CORRECTED]** inline below.
> Where the two documents disagree, `testing.md` wins — it measured, this inferred.

---

## 0. Executive summary

**AMD support in OmniVoice Studio is not missing. It is implemented, working, and then disclaimed by its own metadata.**

The device probe, the HIP/gfx arch gate, the `HSA_OVERRIDE_GFX_VERSION` remapper, the opt-in ROCm wheel installer, the first-run wizard's AMD detection, and a first-class `:rocm` Docker image all already ship and are covered by tests. What does *not* ship is a single `"rocm"` string in any engine's `gpu_compat` tuple. The consequence is a user-visible false negative, described in §2.

Phase 1 set out to find what blocks AMD support. It found that **every technical blocker the codebase cites as its reason to withhold ROCm has either been fixed upstream since the comment was written, or was never true for torch-based engines in the first place.** The single most consequential example: `asr_backend.py:484-486` states "CTranslate2 has no upstream HIP build." CTranslate2 merged HIP support in [PR #1989](https://github.com/OpenNMT/CTranslate2/pull/1989) and has shipped official ROCm wheels for Linux *and* Windows since v4.7.0 on 2026-02-03.

Two genuine constraints survived the audit and must shape the design:

1. **Windows AMD cannot be a default feature.** No torch 2.8.0 Windows ROCm wheel exists on any channel. AMD's supported Windows matrix is torch ≥ 2.9.1, Python 3.12 only, RDNA3/RDNA4 only. This collides with the repo's `constraint-dependencies = ["torch==2.8.0"]` and `requires-python = ">=3.11"`. It is opt-in territory under the project's cross-platform parity rule — which is where the existing ROCm install path already correctly sits.
2. **One open upstream bug threatens the core TTS workload.** `ConvTranspose1d` — the backbone op of HiFi-GAN/BigVGAN-class vocoders — is reported to hang the GPU on gfx110x (RDNA3) via a MIOpen solver failure ([TheRock#4681](https://github.com/ROCm/TheRock/issues/4681), open as of 2026-04-20). This is the one finding that could make a blanket `"rocm"` claim on a TTS engine dishonest, and it is the reason Phase 5 must gate on a real vocoder smoke test rather than an import check.
   > **[CORRECTED]** Tested on gfx1031 (RDNA2): the op is correct to 5.96e-07 and does not hang. The bug appears RDNA3-specific and remains **UNVERIFIED on gfx110x** — the smoke-test requirement stands for RDNA3, not for RDNA2. See `testing.md` §3.1.

**Recommended shape of the fix (detail deferred to Phases 2–3):** this is predominantly a *correctness-of-metadata* change plus a dependency bump, not an architectural build. The abstraction `goal.md` Phase 2 asks for already exists in `backend/core/device_caps.py` and `backend/services/engine_routing.py`, is well-factored, and should be extended rather than replaced.

---

## 1. Method and verification status

Four parallel audits were run: the torch stack, the ASR/non-torch runtimes, an exhaustive in-repo inventory, and the ROCm platform/gfx landscape. Upstream claims were checked against primary sources (PyPI JSON, `download.pytorch.org` / `repo.amd.com` index listings, PyTorch CI build files read at release tags, GitHub issues/PRs, AMD ROCm docs).

Conventions used throughout this document:

- Claims sourced to a primary artifact carry a link.
- Claims that could not be confirmed from a primary source are marked **UNVERIFIED** and must not be built on without checking.
- Fetches to `rocm.docs.amd.com` were intermittently blocked in this environment; where a fact could only be sourced there, it is marked accordingly.

One agent-reported finding was **rejected on review**: the claim that `ROCM_GFX_OVERRIDES` entries for gfx1101/gfx1102 are "now harmful." They are not, because `_configure_rocm_if_needed` (`backend/services/model_manager.py:762-765`) returns early when the native arch is present in the build's arch list. The guard makes stale entries inert. See §7.3.

---

## 2. The core defect: metadata contradicts runtime

On an AMD host with ROCm torch installed, the two halves of the system disagree about what is happening.

**What actually happens at runtime:**

`get_best_device()` returns the literal string `"cuda"` on a ROCm host (`backend/services/model_manager.py:860`). This is correct — PyTorch for HIP [intentionally reuses the `torch.cuda` interfaces](https://docs.pytorch.org/docs/2.13/notes/hip.html). Engines call `.to("cuda")`, and the model genuinely runs on the AMD GPU.

**What the app tells the user:**

`resolve_routing()` reads the engine's `gpu_compat` tuple, finds no `"rocm"` member, and takes the fallback branch at `backend/services/engine_routing.py:118-129`, returning `cpu_fallback` with the purpose-built reason string `"declares CUDA only; ROCm not in its compat set"` (`:121-122`). `routing_notice()` (`:157-165`) then raises a user-visible warning **on every generation**.

**Net effect:** a user who installed ROCm userspace, opted into the ROCm wheel, and is getting GPU-speed output sees a permanent "running on CPU, ~10× slower" banner and an engine matrix showing CPU chips. The metadata is lying in the pessimistic direction.

This is enforced, not accidental. `tests/test_asr_gpu_compat.py:67-70` (`test_no_asr_engine_falsely_claims_rocm`) asserts that no ASR engine claims ROCm, and the ABC comment at `backend/services/asr_backend.py:284-286` gives the reasoning:

> ROCm is intentionally NOT claimed yet for any ASR engine — an unverified `rocm` claim would route ROCm hosts to a broken GPU path, strictly worse than the honest `cpu_fallback`.

That was a defensible call when nothing had been verified. The purpose of this research phase was to do the verification it was waiting for.

### 2.1 Complete `gpu_compat` inventory

Not one class in the entire backend declares `"rocm"`. The string appears only in ABC docstrings, the routing resolver, the frontend type union, and tests that forbid it.

| Family | Class | `gpu_compat` | Anchor |
|---|---|---|---|
| TTS | `TTSBackend` (ABC) | `("cpu",)` | `tts_backend.py:283` |
| TTS | `OmniVoiceBackend` | `("cuda","mps","cpu")` | `tts_backend.py:517` |
| TTS | `VoxCPM2Backend` | `("cuda","mps","cpu")` | `tts_backend.py:788` |
| TTS | `MossTTSNanoBackend` | `("cuda","cpu")` | `tts_backend.py:957` |
| TTS | `KittenTTSBackend` | `("cpu",)` | `tts_backend.py:1074` |
| TTS | `MLXAudioBackend` | `("mps","cpu")` | `tts_backend.py:1315` |
| TTS | `CosyVoiceBackend` | `("cuda","cpu")` | `tts_backend.py:1497` |
| TTS | `GPTSoVITSBackend` | `("cuda","cpu")` | `tts_backend.py:1661` |
| TTS | `SherpaOnnxBackend` | `("cuda","cpu")` | `tts_backend.py:1770` |
| TTS | `Confucius4Backend` | `("cuda","cpu")` | `engines/confucius4/__init__.py:67` |
| TTS | `DotsTTSBackend` | `("cuda","cpu")` | `engines/dots_tts/__init__.py:75` |
| TTS | `MossTTSV15Backend` | `("cuda","cpu")` | `engines/moss_tts_v15/__init__.py:94` |
| TTS | `Supertonic3Backend` | `("cpu",)` | `engines/supertonic3/backend.py:76` |
| TTS | `IndexTTS2Backend` | `("cuda","cpu")` | `engines/indextts/__init__.py:87` |
| TTS | `OmniVoiceSubprocessBackend` | `("cuda","mps","cpu")` | `engines/omnivoice_subprocess/__init__.py:54` |
| TTS | `OmniVoiceGGUFBackend` | `("cuda","mps","cpu")` | `engines/omnivoice_gguf/backend.py:311` |
| ASR | `ASRBackend` (ABC) | `("cpu",)` | `asr_backend.py:287` |
| ASR | `WhisperXBackend` | `("cuda","cpu")` | `asr_backend.py:487` |
| ASR | `FasterWhisperBackend` | `("cuda","cpu")` | `asr_backend.py:923` |
| ASR | `MLXWhisperBackend` | `("mps","cpu")` | `asr_backend.py:1102` |
| ASR | `PyTorchWhisperBackend` | `("cuda","mps","cpu")` | `asr_backend.py:1193` |
| ASR | `NeMoASRBackend` | `("cuda","cpu")` | `asr_backend.py:1281` |
| ASR | `ParakeetMLXBackend` | `("mps",)` | `asr_backend.py:1402` |
| ASR | `MoonshineASRBackend` | `("cpu",)` | `asr_backend.py:1525` |
| ASR | `SherpaDictationBackend` | `("cpu",)` | `asr_backend.py:1650` |
| ASR | `FunASRBackend` | `("cuda","cpu")` | `asr_backend.py:1828` |
| ASR | `OpenAICompatASRBackend` | `("cpu",)` | `asr_backend.py:2075` |
| ASR | `IsolatedFasterWhisperBackend` | `("cuda","cpu")` | `subprocess_asr.py:154` |
| LLM | `LLMBackend` (ABC) | `()` — by design | `llm_backend.py:45` |

The UI is already capable of rendering the result. `frontend/src/components/EngineCompatibilityMatrix.jsx:108` defines a `rocm: 'ROCm'` label and `:120` an AMD-red chip style. Nothing has ever produced one.

---

## 3. What already exists (inventory)

Recorded so Phase 2 extends rather than rebuilds.

| Layer | Implementation | Anchor |
|---|---|---|
| Canonical probe | `family="rocm"` distinguished from CUDA via `torch.version.hip`; VRAM, driver, notes | `core/device_caps.py:297-336` |
| gfx arch gate | `arch_unsupported()` branches on HIP vs CUDA — gfx names vs `sm_` tags (#1228) | `core/device_caps.py:177-242` |
| Arch normalisation | `"gfx90a:xnack+"` → `"gfx90a"` | `core/device_caps.py:94-96` |
| GFX remap table | `ROCM_GFX_OVERRIDES` | `core/device_caps.py:70-79` |
| HSA env writer | Applies override only when native arch absent **and** remap target present | `services/model_manager.py:718-782` |
| Loader | Delegates family to probe; returns `"cuda"` for ROCm | `services/model_manager.py:822-888` |
| Routing | Dedicated ROCm branch + reason string | `services/engine_routing.py:118-129` |
| torch.compile gate | ROCm-aware, avoids disabling compile on AMD (#1228) | `services/engine_env.py:60-98` |
| Installer opt-in | `OMNIVOICE_TORCH_VARIANT=rocm` → `whl/rocm6.4` | `src-tauri/src/bootstrap.rs:890,930-937,1885-1909` |
| Wizard detection | `/sys/class/drm` vendor `0x1002`; `rocm`/`amd` split on userspace presence | `src-tauri/src/setup.rs:369-380,440-459` |
| cuDNN skip | HIP probed **before** `cuda.is_available()`, so AMD boxes skip the ~700 MB cuDNN wheel | `src-tauri/src/bootstrap.rs:1165-1173` |
| Docker | `:rocm` variant + build-time `torch.version.hip` assertion | `deploy/Dockerfile:3,85-89`; `.github/workflows/docker.yml:167` |
| Tests | 290-line `#1228` suite; ROCm cases in device-caps and routing | `tests/test_rocm_arch_gate.py` |
| Docs | AMD section, Docker ROCm quick-start, Windows "CPU-only" statement | `docs/install/linux.md:220-294`; `docs/install/docker.md:63-100` |

**Assessment:** the existing design is sound and honest-by-construction. The `HSA_OVERRIDE_GFX_VERSION` logic in particular is more careful than most upstream projects — it refuses to remap onto an architecture the build does not ship, which is precisely the failure mode §7.2 documents as harmful elsewhere.

---

## 4. Dependency audit

Per `goal.md` Phase 1. `AMD status` values: **native** (upstream-supported), **inherit** (device-agnostic PyTorch, works with no ROCm-specific code), **community-patch**, **unsupported**, **n/a**.

### 4.1 Cross-cutting fact

A ROCm torch build presents through `torch.cuda`: `torch.cuda.is_available()` → `True`, `torch.version.hip` is set, `torch.version.cuda` is `None`, and `torch.device("cuda")` dispatches to HIP. `torch.cuda.get_arch_list()` returns gfx names.

**Therefore any dependency that is device-agnostic PyTorch works on ROCm with zero code changes, and any dependency whose auto-detect is `"cuda" if torch.cuda.is_available() else ...` picks the AMD GPU correctly.** This single fact resolves the majority of the table below.

### 4.2 Core torch stack

| Package | Pin | CUDA | ROCm | CPU | AMD status | Limitations / issues |
|---|---|---|---|---|---|---|
| `torch` / `torchaudio` | **2.8.0** (hard constraint, `pyproject.toml:253-256`) | ✅ | ✅ Linux | ✅ | native | Linux `whl/rocm6.4` carries 2.8.0. **No Windows ROCm wheel at 2.8.0 on any channel** — see §5 |
| `transformers` | 5.3.0 | ✅ | ✅ | ✅ | inherit | SDPA silently falls back to math backend on RDNA3 without `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — [aotriton#16](https://github.com/ROCm/aotriton/issues/16), [pytorch#112997](https://github.com/pytorch/pytorch/issues/112997) |
| `accelerate` | 1.13.0 | ✅ | ✅ | ✅ | inherit | HF validates on Instinct only; consumer RDNA "not validated but expected to work" |
| `pyannote-audio` | 3.4.0 | ✅ | ✅ | ✅ | inherit | **3.1 removed onnxruntime** — now pure PyTorch ([#1557](https://github.com/pyannote/pyannote-audio/issues/1557)). AMD [demonstrates it on ROCm](https://rocm.blogs.amd.com/artificial-intelligence/speech_models/README.html). VRAM-hungry (~14 GB, [#1580](https://github.com/pyannote/pyannote-audio/issues/1580)) |
| `speechbrain` / `pytorch-lightning` / `julius` / `asteroid-filterbanks` / `torch-audiomentations` | locked | ✅ | ✅ | ✅ | inherit | Pure PyTorch, no compiled CUDA extensions |
| `demucs` | 4.0.1 | ✅ | ✅ | ✅ | inherit | Verified: zero `cuda`/`autocast` refs in `apply.py`; device default is `"cuda" if th.cuda.is_available()` → HIP. Already correct |
| `audioseal` | 0.2.0 | ✅ | ✅ | ✅ | n/a | Pure-Python wheel. **Moot** — repo pins it to CPU with its own 1-worker pool (`model_manager.py:676`), outside device routing entirely |
| `pedalboard` | 0.9.22 | ❌ | ❌ | ✅ | n/a | JUCE/CPU DSP, `numpy`-only dep. No GPU surface |
| `triton` | 3.4.0 | ✅ | — | — | **swap required** | ROCm needs `pytorch-triton-rocm`; see §4.5 |

**No CUDA-only compiled kernels resolve in `uv.lock`.** `flash-attn`, `deepspeed`, `bitsandbytes`, and `xformers` are all absent from the resolution. The entire CUDA-only compiled surface is the `nvidia-*-cu12` set, `triton`, and (until bumped) `ctranslate2`.

### 4.3 ASR and non-torch runtimes

| Package | Pin | CUDA | ROCm | CPU | AMD status | Limitations / issues |
|---|---|---|---|---|---|---|
| **`ctranslate2`** | **4.4.0** | ✅ | ✅ **since 4.7.0** | ✅ | **native** | See §4.4 — the headline finding |
| `whisperx` | 3.4.2 | ✅ | ✅ inherited | ✅ | native-via-CT2 | No code change needed; `device="cuda"` is what a HIP CT2 build expects |
| `faster-whisper` | 1.2.1 | ✅ | ✅ inherited | ✅ | native-via-CT2 | Same |
| `argostranslate` | 1.11.0 | ✅ | ✅ inherited | ✅ | native-via-CT2 | `ARGOS_DEVICE_TYPE` passes straight to CT2. Use `float16`, not `int8`, on HIP |
| `onnxruntime` | 1.24.4 | ✅ | ❌ **ROCm EP removed in 1.23** | ✅ | native via `onnxruntime-migraphx` | Successor publishes **1.24.4 — exact version match**. [**Linux only**](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html), no Windows/macOS at any version |
| `sherpa-onnx` | 1.13.3 | ✅ | ❌ | ✅ | **unsupported** | Bundles its own `libonnxruntime` — swapping the pip package does nothing. [#196](https://github.com/k2-fsa/sherpa-onnx/issues/196) open since 2023-07-05; [PR#1110](https://github.com/k2-fsa/sherpa-onnx/pull/1110) and [PR#2370](https://github.com/k2-fsa/sherpa-onnx/pull/2370) unmerged |
| `nemo_toolkit` | optional | ✅ | ⚠️ **broken** | ✅ | community-patch | [Speech#15905](https://github.com/NVIDIA-NeMo/Speech/issues/15905) open — `dlopen libcuda.so.1` crash on ROCm torch/gfx1100. Fix [PR#15957](https://github.com/NVIDIA-NeMo/Speech/pull/15957) open. Hard `cuda-bindings` dep |
| `funasr` | 1.3.1 (eval extra) | ✅ | ✅ by inheritance | ✅ | inherit, untested | Pure PyTorch + pure Python deps, zero CUDA-only packages. No upstream ROCm testing |
| llama.cpp / GGML | bundled C++ binary | ✅ | ✅ **HIP on Win+Linux** | ✅ | native + Vulkan | See §4.6 |
| MLX family | darwin-arm64 markers | — | — | — | n/a | Apple Silicon only; correctly gated already |

### 4.4 CTranslate2 — the headline finding

**The repo's stated blocker is stale.** Two in-code claims are now factually wrong:

- `backend/services/asr_backend.py:484-486` — "CTranslate2 has no upstream HIP build"
- `backend/api/routers/setup/models.py:562-566` — user-facing AMD/ROCm preset copy repeating it

| Fact | Value |
|---|---|
| Upstream PR | [#1989 "Introduce AMD GPU support with ROCm HIP"](https://github.com/OpenNMT/CTranslate2/pull/1989), merged 2026-02-02 |
| First release | **v4.7.0**, 2026-02-03 |
| Latest | v4.8.1, 2026-07-03 |
| Repo pin | **4.4.0** — three minor versions behind |
| Distribution | GitHub **release assets**, not PyPI: `rocm-python-wheels-Linux.zip` (271 MB), `rocm-python-wheels-Windows.zip` (131 MB) |
| Built against | ROCm 7.2 |
| Compiled archs | `gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201` |
| Device string | Unchanged — reuses `Device::CUDA` under a hipified `CT2_WITH_CUDA`, so `device="cuda"` is correct |

Notable: CT2's arch coverage is **wider than official PyTorch ROCm wheels**, which omit gfx1150/gfx1151 below torch 2.10.

**Why this matters disproportionately:** CTranslate2 is the shared substrate of WhisperX, Faster-Whisper, *and* Argos translation. One dependency bump unblocks ASR and dub-translation simultaneously — which is also the risk: one native library, one blast radius, one regression surface spanning two subsystems.

**Traps:**
- Wheels are **off-PyPI**. `pip install ctranslate2` will never get the ROCm build.
- [CT2#2016](https://github.com/OpenNMT/CTranslate2/issues/2016) — open, no maintainer reply since 2026-02-14. Windows wheels are built against ROCm 7.2, but the standalone Windows HIP SDK installer tops out at 7.1.1, where the library is `libhipblas.dll` rather than `hipblas.dll` → crash at load. Windows users must install the ROCm 7.2 **pip** SDK wheels, not the HIP SDK installer.
- Not FA-tuned; wave32 tuned for RDNA. GPU int8 not validated — plan on `float16`.
- Python/platform tag matrix inferred from absence of a `CIBW_BUILD` filter — **UNVERIFIED** (not byte-inspected inside the zip).

### 4.5 The `triton` → `pytorch-triton-rocm` packaging problem

The only item in the torch stack requiring real engineering. It is a **resolution-time** problem, not a runtime one.

On ROCm, torch depends on `pytorch-triton-rocm`, not `triton`. Three traps stack:

1. **PyPI's `pytorch-triton-rocm` is stale at 2.1.0 (Apr 2023).** The 3.x line exists *only* on `download.pytorch.org/whl/rocmX.Y`. Good news: `rocm6.4` carries **3.4.0**, exactly matching torch 2.8.0's pairing.
2. **[uv#10712](https://github.com/astral-sh/uv/issues/10712)** — resolving torch from a ROCm index fails with "No solution found" unless `pytorch-triton-rocm` is declared explicitly against that same index. **In-repo precedent exists**: `pyproject.toml` already does exactly this for `sherpa-onnx-core`.
3. **`triton` and `pytorch-triton-rocm` install into the same `triton/` directory.** The current `triton` marker is `platform_machine == 'x86_64' and sys_platform == 'linux'` — a ROCm variant's markers must be provably **disjoint**, or they collide on the identical platform.

**Keep-main-green impact:** adding a ROCm index introduces a third `torch` entry in `uv.lock` with its own resolution markers, plus `pytorch-triton-rocm` and a `torchaudio` twin. `deploy/Dockerfile` runs `--frozen-lockfile`, so CI-green would not prove Docker-green. Related: [pytorch#167411](https://github.com/pytorch/pytorch/issues/167411).

### 4.6 GGML/Vulkan — an available win, currently unclaimed

`backend/engines/omnivoice_gguf/` does **not** use `llama-cpp-python`. It spawns a bundled binary from `ServeurpersoCom/omnivoice.cpp` which, per `backend.py:11`, "statically links GGML/GGML-cpu and **dlopens GGML-cuda / GGML-vulkan when present**."

But `scripts/build-omnivoice-tts.sh` configures plain CPU for `linux-x86_64` and `windows-x86_64`, and Metal only for `darwin-arm64`. **`-DGGML_VULKAN=ON` appears nowhere.**

Adding that flag to the Linux and Windows build arms gives AMD *and* Intel GPU acceleration on both OSes, through a path the runtime already dlopens, with **no ROCm install required by the user**. Unlike ROCm, Vulkan behaves identically on Windows and Linux — so this *satisfies* the default-parity rule rather than straining it.

For reference, upstream llama.cpp release b10240 (2026-08-03) ships first-party `win-hip-radeon-x64`, `ubuntu-rocm-7.2-x64`, `win-vulkan-x64`, and `ubuntu-vulkan-x64` assets — the widely-repeated claim that "llama.cpp HIP is Linux-only" is false.

---

## 5. Platform matrix

### 5.1 Linux — coherent today

`torch==2.8.0` + `https://download.pytorch.org/whl/rocm6.4` is a valid, existing combination (index verified: torch 2.8.0 cp39–cp313, `manylinux_2_28_x86_64`). This is exactly what `bootstrap.rs:890` already installs. **No change required to the Linux install path.**

### 5.2 Windows — a version conflict, not a technology gap

Native Windows ROCm PyTorch is real and AMD-shipped. Its headline limitation — "ML training not supported, inference only" — **does not bite here**, because OmniVoice is inference-only. The binding constraints are purely mechanical:

| Constraint | AMD's Windows matrix | This repo |
|---|---|---|
| torch | **≥ 2.9.1** (no 2.8.0 Windows ROCm wheel exists on *any* channel) | `torch==2.8.0` hard pin |
| Python | **3.12 only** (supported matrix) | `requires-python = ">=3.11"` admits 3.11/3.12/3.13 |
| OS | Windows 11 | — |
| Driver | Adrenalin 26.2.2+ | — |
| GPUs | gfx1100/1101/1200/1201 — RDNA3/RDNA4 only | — |

Two AMD channels exist, and they differ importantly:

- **`repo.radeon.com/rocm/windows/`** — direct wheel URLs. **Not a PEP 503 index**; `--index-url` will not work. AMD's own docs instruct installing full wheel URLs verbatim. (`rocm-rel-6.4.4/` is the one exception, an old preview layout carrying only a torch 2.8.0**a0** alpha.)
- **`repo.amd.com/rocm/whl-multi-arch/`** — **a real PEP 503 index**, torch 2.9.1–2.12.0, cp310–cp314, both `linux_x86_64` and `win_amd64`, selected by extras (`torch[device-gfx1100]`). This is the modern TheRock path and the one to use if Windows ROCm is ever adopted. It also honours `ROCM_SDK_TARGET_FAMILY`, a cleaner lever than `HSA_OVERRIDE_GFX_VERSION`.

**RX 6000 / RDNA2 on Windows: not officially supported — but empirically working. [CORRECTED]** AMD's Windows matrix lists only gfx1201/1200/1100/1101, and that remains the *supported* set. However, an RX 6800M (gfx1031) was tested end-to-end on Windows 11 with a **2021** Adrenalin driver: full model load, correct synthesis, 2.1× faster than realtime, ~461× faster than CPU (`testing.md` §4). The matrix describes what AMD supports, not what functions. Treat RDNA2-on-Windows as **working but unsupported** — viable behind opt-in, not something to promise. TheRock's `SUPPORTED_GPUS.md` does mark gfx1030–gfx1036 release-ready on Windows and publishes `amd-torch-device-gfx1030` win_amd64 wheels at torch ≥ 2.10 — but that file's own caveat is that a build-passing tick "does not imply the runtime is functional on target hardware." Experimental only; never a default.

**Conclusion: Windows AMD must remain opt-in.** It requires a different index, a torch bump, and effectively a pinned interpreter — a separate resolution, not a variant of the default one. This is consistent with where the project already puts ROCm.

### 5.3 Windows alternatives — all dead ends

| Path | Verdict |
|---|---|
| **ZLUDA** | No PyTorch support. FAQ's "PyTorch by Q4 2025" target was missed; v6 notes only "PyTorch fixes", never a claim it runs. Lost commercial funding ~March 2026, now a hobby project. Works for llama.cpp/Blender only |
| **Vulkan (as a torch backend)** | PyTorch's Vulkan backend is unmaintained, now under `/tutorials/unstable/`. [pytorch#160230](https://github.com/pytorch/pytorch/issues/160230) requesting desktop support is open with no maintainer decision. (Vulkan *via GGML* is a different and viable thing — §4.6) |
| **torch-directml** | Frozen at `torch==2.4.1`, last release 2024-09-15, DirectML explicitly in maintenance mode. **Confirms the DirectML branches in `device_caps.py:372` and `get_best_device()` are unreachable** — `torch-directml` appears nowhere in `pyproject.toml`/`uv.lock`, and its pin collides head-on with `torch==2.8.0` |
| **WSL2 + ROCm** | Works for RDNA3/RDNA4 + Strix APUs, but: no multi-GPU, no `rocm-smi`/profiler/debugger, and AMD documents "lower than expected performance compared to native Linux". Heavy install burden; a documentation path at best |

### 5.4 macOS

No AMD path. MPS is Apple Silicon only; Intel Macs with AMD dGPUs have no ROCm. Out of scope — correctly.

---

## 6. gfx architecture coverage

Authoritative source is `PYTORCH_ROCM_ARCH` in [`.ci/docker/manywheel/build.sh`](https://github.com/pytorch/pytorch/blob/main/.ci/docker/manywheel/build.sh), read at release tags. **The list is unconditional at each tag — it is a function of the torch version, not of which rocm index you install from.**

| Target | Hardware | In official wheel? |
|---|---|---|
| gfx900, gfx906 | Vega / Radeon VII | ✅ all versions (but **not in AMD's ROCm 7.x support matrix** — compiled-for, runtime-abandoned) |
| gfx908, gfx90a, gfx942 | CDNA1–3 Instinct | ✅ all versions |
| gfx950 | CDNA4 | from 2.10.0 |
| **gfx1030** | RX 6800/6900, W6800 | ✅ all versions |
| **gfx1031/1032/1034/1035/1036** | RX 6700/6600/6500, RDNA2 iGPU | ❌ never on `download.pytorch.org` — **[CORRECTED]** but ✅ **shipped natively by `repo.amd.com/rocm/whl-multi-arch/`**, verified running (`testing.md`) |
| **gfx1100** | RX 7900 | ✅ all versions |
| **gfx1101** | RX 7800/7700 | ✅ all versions |
| **gfx1102** | RX 7600 | from 2.7.0 |
| gfx1103 | Phoenix APU / 780M | from 2.13.0 |
| **gfx1150/1151** | Strix Point / Strix Halo | from 2.10.0 |
| **gfx1200/1201** | RDNA4 / RX 9000 | from 2.7.0 |
| gfx1152/1153 | newer APUs | ❌ not even on main |

**Consequence for this repo's `torch==2.8.0` pin:** covers gfx900/906/908/90a/942/1030/1100/1101/1102/1200/1201. **Does not cover gfx1150/gfx1151 (Strix Halo/Point — needs ≥2.10) or gfx1103.** Those hosts depend on the `HSA_OVERRIDE_GFX_VERSION` remap, which is exactly what `ROCM_GFX_OVERRIDES` provides.

RDNA4 is covered: gfx1200/1201 entered `PYTORCH_ROCM_ARCH` via [#148562](https://github.com/pytorch/pytorch/pull/148562), first shipping in torch 2.7.0. A torch 2.8.0 + rocm6.4 install accelerates RX 9000.

**Caveat:** presence in `PYTORCH_ROCM_ARCH` means HIP kernels are compiled. It does **not** mean AOTriton/flash-attention kernels exist for that arch — see §7.1.

---

## 7. ROCm-specific runtime hazards

These are the findings that determine whether a `"rocm"` claim is *honest*, and they should drive Phase 5's test design.

### 7.1 `ConvTranspose1d` on RDNA3 — the one genuine blocker

[TheRock#4681](https://github.com/ROCm/TheRock/issues/4681), **open** as of 2026-04-20: `ConvTranspose1d` hangs the GPU on gfx110x. MIOpen solver evaluation fails (`"Invalid elapsed time detected in EvaluateInvokers"`) → "No suitable algorithm" → hang or abort.

`ConvTranspose1d` is the backbone op of HiFi-GAN/BigVGAN-class neural vocoders — i.e. the final stage of most TTS engines in this repo. **This is the single finding that could make a blanket `"rocm"` claim dishonest**, and it is arch-specific rather than universal, so it cannot be resolved by reading code. Phase 5 must gate the claim on an actual end-to-end vocoder smoke test on RDNA3 hardware, not an import check.

### 7.2 Silent performance degradation

| Hazard | Effect |
|---|---|
| **SDPA math fallback** | `TORCH_WARN_ONCE("Flash attention was not compiled for current AMD GPU architecture")`, then quiet and ~10× slower. Experimental archs gated behind `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. AMD ships FA **off by default on RX 7000** |
| **Never pin SDPA backends** | `sdpa_kernel([...])` excluding `math` converts the fallback into a hard `"No available kernel. Aborting execution."` ([ROCm#5404](https://github.com/ROCm/ROCm/issues/5404)) |
| **FA perf inversion on gfx1100** | FA SDPA measured **2.5× slower** than compiled math at head_dim 256 ([pytorch#152595](https://github.com/pytorch/pytorch/issues/152595)) |
| **Windows MIOpen `workspace=0`** | Picks a naive direct conv. A Qwen3-TTS decoder runs at **~12.6× realtime factor** on gfx1200/ROCm 7.2. `MIOPEN_FIND_MODE=2` has no effect. Open ([TheRock#3077](https://github.com/ROCm/TheRock/issues/3077)) — a strong independent argument against Windows ROCm as a default |
| **MIOpen JIT cache poisoning** | A first conv call with an atypical batch in bf16 on gfx1100 caches a bad kernel → **1327× slowdown** thereafter ([pytorch#165141](https://github.com/pytorch/pytorch/issues/165141)). Warm with *representative* shapes |
| **RNN/LSTM** | No fused MIOpen kernel equivalent to cuDNN's; ~4.5× deficit, "Under Investigation" ([ROCm#4677](https://github.com/ROCm/ROCm/issues/4677)). Worse: MIOpen release notes state RNNs may return **incorrect results** for some GEMM-size configurations — a correctness risk, not just perf. GRU on Windows fails to build code objects ([ROCm#6314](https://github.com/ROCm/ROCm/issues/6314)) |
| **gfx11 SDPA head-dim cap** | `sdp_utils.cpp` hard-caps head dim at 256 for `gfx11*` (`if (arch.starts_with("gfx11")) { return 256; }`) — above that, no flash path exists at all |
| **AOTriton may be absent entirely** | `"AOTriton not available: No module named 'pyaotriton'"` — the wheel may ship without it, so capability must be probed, not assumed |
| **torch.compile** | Known codegen failures on gfx1100 ([sglang#30245](https://github.com/sgl-project/sglang/issues/30245), [pytorch#147377](https://github.com/pytorch/pytorch/issues/147377)). `mode="reduce-overhead"` (CUDA graphs) must not be used on ROCm. **Recommend defaulting compile OFF on ROCm** |
| **STFT/FFT** | `cufft_plan_cache` tuning is explicitly unsupported on ROCm → variable-length STFT batches re-plan unboundedly. Pad/bucket input lengths. No confirmed wrong-results bug — **UNVERIFIED**, but nothing upstream guards it, so a golden mel-spectrogram test is warranted |

### 7.3 Precision

- **bf16 is hardware-accelerated only on RDNA3+ and CDNA1+.** RDNA2 has no matrix cores (WMMA arrived with RDNA3) — bf16 there is a vector-ALU perf trap. **Prefer fp16 on RDNA2.**
- **`torch.cuda.is_bf16_supported()` returns True on gfx1030** — it checks allocability, not tensor-core support. Useless as a dtype gate ([pytorch#75427](https://github.com/pytorch/pytorch/issues/75427)).
- TF32 does not exist on any RDNA arch. `allow_fp16_reduced_precision_reduction` is explicitly unsupported on ROCm — setting it is silently meaningless.
- Never size memory off `get_device_properties().total_memory` on an APU: ROCm reports VRAM+GTT summed while `hipMalloc` is capped at the BIOS carve-out ([ROCm#6004](https://github.com/ROCm/ROCm/issues/6004)).

### 7.4 `HSA_OVERRIDE_GFX_VERSION` — current guidance, and a correction

The variable is **not documented by AMD** — it appears nowhere in the [ROCm environment-variables reference](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html). It is a community workaround.

**Still needed:** gfx1031/1032/1034 → `10.3.0` (never in `PYTORCH_ROCM_ARCH`); gfx1103 → `11.0.0` for torch ≤ 2.12.

> ⚠️ **Open risk on the one remaining load-bearing entry.** A SIGSEGV on gfx1031/gfx1032 under `HSA_OVERRIDE_GFX_VERSION` with ROCm 6.4.3+ is reported. Since `rocm6.4` is exactly the index `bootstrap.rs:890` installs, the repo's `gfx1031/1032/1034 → gfx1030` remap — the *only* entry that is both active and still necessary — may be the one that crashes. **UNVERIFIED** (single community report, no primary AMD source, no reproduction on this project's code path). Must be resolved before RX 6700/6600 can be claimed; until then those cards are correctly left on `cpu_fallback`.
**No longer needed on newer torch:** gfx1150/1151 natively supported from torch 2.10 ([ROCm#6034](https://github.com/ROCm/ROCm/issues/6034)); gfx1101/1102 natively supported everywhere.
**Harmful when misapplied:** setting it when the native arch is already in the wheel broke gfx1030 discovery and silently fell back to CPU ([project-nomad#810](https://github.com/Crosstalk-Solutions/project-nomad/issues/810)); cross-family remaps yield `invalid device function` ([llama.cpp#20839](https://github.com/ggml-org/llama.cpp/issues/20839)). It is **process-wide** — a footgun on iGPU+dGPU laptops. **It does nothing on Windows** ([ollama#3107](https://github.com/ollama/ollama/issues/3107)).

**Correction to an audit finding.** One agent reported that this repo's `ROCM_GFX_OVERRIDES` entries for gfx1101/gfx1102/gfx1150/gfx1151 are "stale and now harmful." **This is rejected on review.** `_configure_rocm_if_needed` (`model_manager.py:762-765`) returns early when `gfx_id in arch_list`, and `arch_unsupported()` (`device_caps.py:222-223`) checks membership before consulting the map. Stale entries are therefore **inert**, not harmful — the guard already implements exactly the "don't remap a natively-supported GPU" rule the upstream issues warn about. Furthermore, for the pinned torch 2.8.0 the gfx1150/1151 → gfx1100 remap is *correct and necessary*, since 2.8.0's arch list genuinely lacks those targets.

The only genuinely dead entry is `"gfx906": "gfx906"` (`device_caps.py:78`), a self-map that can never change behaviour.

### 7.5 Detection gotchas relevant to this repo

Most are already handled; recorded for completeness.

| Gotcha | Repo status |
|---|---|
| Strip `:xnack-` / `:sparse+` suffixes from `gcnArchName` | ✅ `_normalize_arch()` |
| `get_arch_list()` missing on old wheels → `_get_arch_list()` | ✅ `build_arch_list()` |
| gfx code objects are **not** forward-compatible like PTX — exact match is right on ROCm | ✅ ROCm branch does exact membership; `cuda_build_covers()` correctly applies forward-compat logic only to CUDA |
| `HSA_OVERRIDE_GFX_VERSION` changes what `gcnArchName` reports | ✅ `gfx_for_hsa_override()` inverse |
| Device 0 is often the iGPU on APU+dGPU laptops | ⚠️ **Gap** — probe indexes device 0 unconditionally (`device_caps.py:318-326`), emitting only an advisory note. Not ROCm-specific, but bites AMD laptops hardest |
| No `rocminfo`/`rocm-smi`/`amd-smi` on Windows | ⚠️ Relevant only if Windows ROCm is adopted; `setup.rs:369-380` is already `#[cfg(target_os = "linux")]` |

---

## 8. Stale claims in-repo requiring correction

Per the docs-sync hard rule, these are owed regardless of any feature work.

| Location | Claim | Reality |
|---|---|---|
| `backend/services/asr_backend.py:484-486` | "CTranslate2 has no upstream HIP build" | False since 2026-02-03 (CT2 v4.7.0) |
| `backend/api/routers/setup/models.py:562-566` | Same, in user-facing preset copy | Same |
| `backend/config/models.yaml:72-77` | Same note on the AMD/ROCm curated entry | Same |
| `backend/api/routers/setup/wizard.py:431-436` | Tells AMD users to re-sync against **`rocm6.1`** | `bootstrap.rs:890` correctly uses `rocm6.4`; `rocm6.1` carries only torch 2.6.0 and cannot satisfy the 2.8.0 pin. **This is a live cause of the CPU-fallback failure #972 was about** |
| `backend/api/routers/setup/wizard.py:426` | AMD host status hard-coded `"warn"` even when ROCm is ready | Should reflect actual readiness |
| `backend/core/device_caps.py:78` | `"gfx906": "gfx906"` self-map | Inert no-op |

---

## 9. Findings that reshape the goal

`goal.md` was written on the premise that AMD support must be built. Three findings revise that:

1. **Phase 2 (architecture) is largely satisfied by existing code.** `device_caps.py` + `engine_routing.py` already implement automatic CUDA/ROCm/MPS/CPU detection with no user configuration. Phase 2 should *document and extend* this design, and explicitly justify not rebuilding it — replacing it would violate the "minimize refactoring" constraint in `goal.md` Phase 4 for no gain.
2. **The primary deliverable is metadata correctness plus a dependency bump**, not a new abstraction. That is a much smaller and safer change than `goal.md` anticipated — and it lands squarely inside the "fix the whole class of the bug" standard, because the fix is per-engine verification, not a blanket flag.
3. **Two items must be decided by the owner, not inferred** (deferred to `architecture.md`):
   - **Windows AMD**: accept opt-in-only via `repo.amd.com/whl-multi-arch` at torch ≥ 2.9.1 + Python 3.12, or decline Windows ROCm entirely and document it. Both are defensible; the second preserves the single-resolution simplicity the repo currently enjoys.
   - **CTranslate2 ROCm wheels**: they are off-PyPI GitHub release assets, which is a meaningfully different supply-chain posture from a pinned PyPI dependency. Adopting them needs an explicit yes.

---

## 10. Open questions for Phase 2

1. Does the pinned `torch==2.8.0` need to move? Linux says no. Windows ROCm says yes (≥2.9.1). The answer follows from the Windows decision, not the reverse.
2. Can `"rocm"` be claimed per-engine without hardware, or must each claim be gated behind a runtime capability probe? §7.1 argues for a probe for any vocoder-bearing engine.
3. Do subprocess engines (`indextts2`, `omnivoice-subprocess`, `faster-whisper-isolated`) inherit the ROCm environment correctly? Their sidecar venvs install their own torch, which the ROCm opt-in reinstall does not touch, and `HSA_OVERRIDE_GFX_VERSION` is only inherited by children spawned *after* `get_best_device()` has run (`model_manager.py:778`). **Claiming `"rocm"` for these without fixing the venv would be precisely the false promise `asr_backend.py:284-286` warns against.**
4. Should the GGML Vulkan build flag (§4.6) be in scope? It is the only change here that improves AMD support *identically on Windows and Linux*, and it benefits Intel GPUs too — but it is arguably a separate feature from ROCm.
5. What is the minimum honest test bar for a `"rocm"` claim, given Phase 5 requires testing on NVIDIA / AMD ROCm / CPU and AMD hardware may not be available to the maintainer?

---

## Appendix — primary sources

**PyTorch/ROCm:** [wheel index root](https://download.pytorch.org/whl/) · [rocm6.4 torch](https://download.pytorch.org/whl/rocm6.4/torch/) · [previous-versions matrix](https://pytorch.org/get-started/previous-versions/) · [PYTORCH_ROCM_ARCH](https://github.com/pytorch/pytorch/blob/main/.ci/docker/manywheel/build.sh) · [HIP semantics](https://docs.pytorch.org/docs/2.13/notes/hip.html) · [#159520 Windows ROCm CI RFC](https://github.com/pytorch/pytorch/issues/159520) · [#148562 gfx12](https://github.com/pytorch/pytorch/pull/148562) · [#164854 gfx1150/1151](https://github.com/pytorch/pytorch/pull/164854)

**AMD channels:** [repo.amd.com whl-multi-arch](https://repo.amd.com/rocm/whl-multi-arch/) · [repo.radeon.com windows](https://repo.radeon.com/rocm/windows/) · [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) · [TheRock SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)

**CTranslate2:** [PR#1989](https://github.com/OpenNMT/CTranslate2/pull/1989) · [issue#2016 Windows HIP SDK](https://github.com/OpenNMT/CTranslate2/issues/2016) · [WhisperX-on-AMD guide](https://github.com/m-bain/whisperX/discussions/1364)

**ONNX Runtime:** [ROCm EP removal notice](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html) · [MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

**sherpa-onnx:** [#196](https://github.com/k2-fsa/sherpa-onnx/issues/196) · [PR#1110](https://github.com/k2-fsa/sherpa-onnx/pull/1110) · [PR#2370](https://github.com/k2-fsa/sherpa-onnx/pull/2370)

**Hazards:** [TheRock#4681 ConvTranspose1d](https://github.com/ROCm/TheRock/issues/4681) · [TheRock#3077 Windows MIOpen](https://github.com/ROCm/TheRock/issues/3077) · [pytorch#165141 cache poisoning](https://github.com/pytorch/pytorch/issues/165141) · [aotriton#16](https://github.com/ROCm/aotriton/issues/16) · [ROCm#5404](https://github.com/ROCm/ROCm/issues/5404) · [ROCm#6034](https://github.com/ROCm/ROCm/issues/6034)

**Packaging:** [uv#10712](https://github.com/astral-sh/uv/issues/10712) · [pytorch#167411](https://github.com/pytorch/pytorch/issues/167411)
