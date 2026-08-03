/**
 * First-run Compute card — the AMD/ROCm option's platform gating.
 *
 * fail-before/pass-after: the card hard-coded `rocmAvailable = os === 'linux'`,
 * so a Windows AMD user was never offered GPU acceleration at all — even though
 * AMD's TheRock channel publishes Windows ROCm wheels. These tests pin the
 * three-way behaviour that replaced it:
 *
 *   linux + kind 'rocm' → offered AND pre-selected (ROCm userspace verified)
 *   windows + kind 'amd' → offered, NOT pre-selected (see below)
 *   macos                → never offered (no ROCm build exists)
 *
 * The windows case is the subtle one. It stays un-pre-selected on purpose: the
 * Windows wheels pin a higher torch than the app's default build, and AMD's own
 * Windows support matrix covers only RDNA3/RDNA4 — so it is something a user
 * opts into, never something switched on for them.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import FirstRunSetup from './FirstRunSetup';

const invokeMock = vi.fn();

vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...args) => invokeMock(...args),
}));
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(async () => () => {}),
}));
vi.mock('@tauri-apps/plugin-opener', () => ({
  revealItemInDir: vi.fn(),
  openUrl: vi.fn(),
}));

/** A `get_setup_state` payload with only the bits the card reads varied. */
function setupState({ os, kind, gpu }) {
  return {
    firstRun: true,
    os,
    defaults: {
      installMode: 'installed',
      envDir: '/tmp/env',
      dataDir: '/tmp/data',
      modelsDir: '/tmp/models',
      region: 'auto',
      updateChannel: 'stable',
      torchVariant: 'auto',
    },
    portable: { available: false, baseDir: null, reason: 'no_anchor' },
    requirements: { envBytes: 1, modelsBytes: 1, dataBytes: 1 },
    hardware: {
      gpu,
      kind,
      osName: os,
      arch: 'x86_64',
      cpuCores: 8,
      ramGb: 32,
    },
  };
}

function mountWith(state) {
  invokeMock.mockReset();
  invokeMock.mockImplementation(async (cmd) => {
    if (cmd === 'get_setup_state') return state;
    if (cmd === 'check_install_target') {
      return { path: '/tmp', exists: true, writable: true, freeBytes: 9e11 };
    }
    return null;
  });
  return render(<FirstRunSetup />);
}

beforeEach(() => {
  window.__TAURI_INTERNALS__ = {};
});

describe('Compute card — ROCm platform gating', () => {
  it('offers ROCm on Windows for an AMD GPU, without pre-selecting it', async () => {
    mountWith(setupState({ os: 'windows', kind: 'amd', gpu: 'AMD Radeon RX 6800M' }));

    const card = await screen.findByText('AMD GPU (ROCm)');
    expect(card).toBeInTheDocument();
    // Offered-but-unconfirmed: we can't verify it works until install runs.
    expect(screen.getByText('AMD GPU detected')).toBeInTheDocument();
    // …and crucially NOT the "matches this machine" badge, which is what
    // pre-selection looks like.
    expect(screen.queryByText('matches this machine')).not.toBeInTheDocument();
  });

  it('pre-selects ROCm on Linux when the ROCm userspace was verified', async () => {
    mountWith(setupState({ os: 'linux', kind: 'rocm', gpu: 'AMD GPU' }));

    expect(await screen.findByText('AMD GPU (ROCm)')).toBeInTheDocument();
    expect(screen.getByText('matches this machine')).toBeInTheDocument();
  });

  it('never offers ROCm on macOS — no ROCm build exists for it', async () => {
    mountWith(setupState({ os: 'macos', kind: 'mps', gpu: 'Apple Silicon' }));

    // Anchor on the Auto card, which every platform renders. Without waiting
    // for something the Compute section actually shows, the absence assertion
    // below would pass vacuously while the payload is still loading.
    await screen.findByText('Auto (NVIDIA CUDA / Apple MPS / CPU)');
    expect(screen.queryByText('AMD GPU (ROCm)')).not.toBeInTheDocument();
  });

  it('drops the Linux-only wording now that Windows is supported', async () => {
    mountWith(setupState({ os: 'windows', kind: 'amd', gpu: 'AMD Radeon RX 6800M' }));

    await screen.findByText('AMD GPU (ROCm)');
    // The old strings named Linux in both the title and the description; a
    // Windows user reading "on Linux" would reasonably conclude it can't work.
    expect(screen.queryByText(/ROCm, Linux/)).not.toBeInTheDocument();
    expect(screen.queryByText(/AMD graphics cards on Linux/)).not.toBeInTheDocument();
  });
});
