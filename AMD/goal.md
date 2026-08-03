# Goal: AMD GPU Support for OmniVoice Studio

Reference:
- https://github.com/debpalash/OmniVoice-Studio
- https://github.com/k2-fsa/OmniVoice/issues/67

## Objective

Implement first-class AMD GPU (ROCm/HIP) support across OmniVoice Studio while keeping existing CUDA, MPS and CPU support unchanged.

The implementation should reuse existing AMD-compatible libraries whenever possible. If a dependency does not officially support AMD, investigate whether a minimal compatibility patch can be written instead of replacing the dependency.

Do not change models. Do not introduce alternative inference engines.

---

## Phase 1 — Research

Before writing code, perform a complete dependency audit.

For every inference dependency determine:

- Supports CUDA
- Supports ROCm/HIP
- Supports CPU
- AMD status
- Required version
- Known limitations
- Existing upstream issues
- Existing community patches

Produce:

`AMD/research.md`

---

## Phase 2 — Architecture

Design a backend abstraction for device selection.

Requirements:

- CUDA
- ROCm
- MPS
- CPU

Automatic detection.

No user configuration required.

Document the design in

`AMD/architecture.md`

---

## Phase 3 — Implementation Plan

Produce a detailed implementation plan before coding.

Include:

- files to modify
- dependency changes
- backend initialization
- device detection
- runtime dispatch
- testing strategy
- rollback strategy
- risks

Save as

`AMD/implementation-plan.md`

---

## Phase 4 — Implementation

Implement only after the plan is complete.

Requirements

- preserve CUDA behavior
- preserve MPS behavior
- preserve CPU behavior
- add ROCm support
- avoid duplicated code
- minimize refactoring

---

## Phase 5 — Testing

Test on

- NVIDIA
- AMD ROCm
- CPU

Document unsupported features.

Save results in

`AMD/testing.md`

---

## Constraints

- Do not replace existing models.
- Do not recommend alternative models.
- Prefer upstream AMD support whenever available.
- Only write compatibility patches when necessary.
- Keep patches small and maintainable.
- Keep changes upstream-friendly.

---

## Claude Code Instructions

Use **Plan Mode** before making any code changes.

For every major phase, spawn specialized subagents in parallel to review:

- architecture
- dependency compatibility
- ROCm support
- implementation risks
- testing strategy

Challenge assumptions and compare approaches before selecting one.

Seek consensus between subagents before implementation.

Do **not** begin coding until:

- research is complete
- architecture is approved
- implementation plan is finalized

Then implement incrementally with small, reviewable commits.

The goal is complete only when:

- AMD GPU support is implemented where technically possible.
- Existing CUDA functionality remains unchanged.
- Documentation is complete.
- All implementation artifacts under `AMD/` are finished.