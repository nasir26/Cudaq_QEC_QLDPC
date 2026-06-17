#!/usr/bin/env python3
"""
CUDA-Q Multi-Circuit Hardware Benchmark
=========================================
Mirrors the FPGA benchmark script's circuit complexity and metric coverage,
ported to CUDA-Q on the nvidia (GPU statevector) backend.

Circuit families
  GHZ  — H(0) + CX chain (n-1 entangling gates)
  QFT  — full n-qubit Quantum Fourier Transform (n²/2 two-qubit gates)
  RQC  — random quantum circuit, Rx/Ry/Rz per qubit + CX brickwork
  VQE  — hardware-efficient ansatz, Ry/Rz layers + CX ladder

Metrics per (circuit, n_qubits) point
  - CUDA-Q statevector kernel time   [perf_counter]
  - CPU numpy statevector time       [perf_counter, capped at MAX_CPU_Q]
  - GPU/CPU speedup
  - State fidelity vs analytic ideal (GHZ, QFT) or CPU reference (RQC, VQE)
  - Estimated energy (GPU TDP × utilisation fraction × time)
  - Gate count, circuit depth

Output
  - Console table (one row per data point)
  - 6-panel publication-quality PNG
  - JSON results file

Known CUDA-Q API constraints observed during development
  - cr1()  not crz() for QFT phase gates
  - cx()   not swap() as noise-injectable primitive
  - unset_noise()  not reset_noise()
  - List[float] kernel arguments for parameterised circuits
  - get_state() returns amplitudes; no measurements in kernel

Usage
  python cudaq_benchmark.py [--max-qubits 29] [--max-cpu-qubits 22]
                            [--trials 3] [--output results.json]
                            [--no-noise] [--no-plots]

Author : Nasir Ali — C-DAC / NQM Qniverse
"""

import sys
import math
import time
import json
import argparse
from datetime import datetime
from typing import List

import numpy as np

try:
    import cudaq
except ImportError:
    print("ERROR: cudaq not installed.  Activate the cudaqx_qec environment.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_OK = True
except ImportError:
    PLOT_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  Hardware constants — RTX A4000
# ─────────────────────────────────────────────────────────────────────────────
VRAM_GB       = 16.0
GPU_TDP_W     = 140.0   # A4000 TDP
GPU_UTIL_FRAC = 0.70    # estimated fraction of TDP consumed during QSim

# Simulation parameters
MAX_QUBITS_DEFAULT    = 29   # 29q = 8 GB statevector — safe for 16 GB VRAM
MAX_CPU_Q_DEFAULT     = 22   # beyond this numpy statevector exceeds ~64 GB RAM
TRIALS                = 3    # timed repetitions (excluding warmup)
WARMUP                = 1    # un-timed warmup runs

# RQC / VQE hyperparameters (fixed for reproducibility)
RQC_DEPTH  = 20
VQE_LAYERS = 3
RQC_SEED   = 42
VQE_SEED   = 123

# ─────────────────────────────────────────────────────────────────────────────
#  CUDA-Q kernel definitions
# ─────────────────────────────────────────────────────────────────────────────

@cudaq.kernel
def ghz_kernel(n: int):
    """
    GHZ state: H(q0) then CX(q_i, q_{i+1}) for all i.
    Produces (|00…0⟩ + |11…1⟩) / √2.
    Gate count: 1 H + (n-1) CX.
    """
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        cx(q[i], q[i + 1])


@cudaq.kernel
def qft_kernel(n: int):
    """
    Full QFT on |0…0⟩.

    Gate choice: cr1(θ)  — diagonal phase gate diag(1, e^{iθ}).
    NOT crz(θ)  — that applies diag(e^{-iθ/2}, e^{+iθ/2}), wrong unitary.
    Angle: θ = 2π / 2^(j-i+1)  for control j, target i.

    Gate count: n H + n(n-1)/2 CR1 + ⌊n/2⌋ SWAP.
    """
    q = cudaq.qvector(n)
    for i in range(n):
        h(q[i])
        for j in range(i + 1, n):
            angle = 2.0 * math.pi / (2 ** (j - i + 1))
            cr1(angle, q[j], q[i])
    for i in range(n // 2):
        swap(q[i], q[n - i - 1])


@cudaq.kernel
def rqc_kernel(n: int, depth: int, angles: List[float]):
    """
    Random Quantum Circuit.
    Each layer: Rx(θ), Ry(φ), Rz(ψ) per qubit + CX brickwork on even pairs.
    Angles pre-generated on the host; kernel receives as List[float].
    3 angles per qubit per layer → len(angles) = depth * n * 3.

    Gate count: depth × (3n single-qubit + ⌊n/2⌋ CX).
    """
    q = cudaq.qvector(n)
    for d in range(depth):
        for i in range(n):
            base = (d * n + i) * 3
            rx(angles[base],     q[i])
            ry(angles[base + 1], q[i])
            rz(angles[base + 2], q[i])
        for i in range(0, n - 1, 2):
            cx(q[i], q[i + 1])


@cudaq.kernel
def vqe_kernel(n: int, layers: int, angles: List[float]):
    """
    Hardware-efficient VQE ansatz.
    Each layer: Ry(θ), Rz(φ) per qubit + full CX ladder.
    2 angles per qubit per layer → len(angles) = layers * n * 2.

    Gate count: layers × (2n single-qubit + (n-1) CX).
    """
    q = cudaq.qvector(n)
    for l in range(layers):
        for i in range(n):
            base = (l * n + i) * 2
            ry(angles[base],     q[i])
            rz(angles[base + 1], q[i])
        for i in range(n - 1):
            cx(q[i], q[i + 1])


# ─────────────────────────────────────────────────────────────────────────────
#  Angle generators (host-side, reproducible)
# ─────────────────────────────────────────────────────────────────────────────

def rqc_angles(n: int, depth: int, seed: int = RQC_SEED) -> List[float]:
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, 2.0 * math.pi, depth * n * 3).tolist()


def vqe_angles(n: int, layers: int, seed: int = VQE_SEED) -> List[float]:
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, math.pi, layers * n * 2).tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  Gate count helpers (must match kernel definitions exactly)
# ─────────────────────────────────────────────────────────────────────────────

def gate_counts(family: str, n: int) -> dict:
    if family == 'ghz':
        return {'H': 1, 'CX': n - 1, 'total': n}
    elif family == 'qft':
        n_cr1  = n * (n - 1) // 2
        n_swap = n // 2
        return {'H': n, 'CR1': n_cr1, 'SWAP': n_swap,
                'total': n + n_cr1 + n_swap}
    elif family == 'rqc':
        sq = RQC_DEPTH * n * 3
        tq = RQC_DEPTH * (n // 2)
        return {'1Q': sq, 'CX': tq, 'total': sq + tq}
    elif family == 'vqe':
        sq = VQE_LAYERS * n * 2
        tq = VQE_LAYERS * (n - 1)
        return {'1Q': sq, 'CX': tq, 'total': sq + tq}
    return {'total': 0}


# ─────────────────────────────────────────────────────────────────────────────
#  CPU numpy reference simulator
# ─────────────────────────────────────────────────────────────────────────────

class CPUSimulator:
    """
    Exact statevector simulator using numpy.
    Applies gates as tensor contractions on the [2]*n shaped statevector.
    Used for:  (a) CPU timing comparison  (b) fidelity reference for RQC/VQE.
    """

    # Pre-built gate matrices
    _H  = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
    _X  = np.array([[0, 1], [1,  0]], dtype=np.complex128)

    @staticmethod
    def _apply1(sv: np.ndarray, U: np.ndarray, k: int, n: int) -> np.ndarray:
        """Apply 2×2 unitary U to qubit k of n-qubit statevector."""
        sv = sv.reshape([2] * n)
        sv = np.tensordot(U, sv, axes=[[1], [k]])
        sv = np.moveaxis(sv, 0, k)
        return sv.reshape(-1)

    @staticmethod
    def _apply_cx(sv: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
        """CX gate: flip tgt qubit when ctrl qubit is |1⟩."""
        sv = sv.reshape([2] * n)
        # Select the ctrl=1 slice, flip along tgt axis within that slice
        idx = [slice(None)] * n
        idx[ctrl] = 1
        sub = sv[tuple(idx)]
        tgt_in_sub = tgt if tgt < ctrl else tgt - 1
        sv[tuple(idx)] = np.flip(sub, axis=tgt_in_sub)
        return sv.reshape(-1)

    @staticmethod
    def _apply_cr1(sv: np.ndarray, angle: float,
                   ctrl: int, tgt: int, n: int) -> np.ndarray:
        """Controlled-R1: apply diag(1, e^{iθ}) to tgt when ctrl=|1⟩."""
        sv = sv.reshape([2] * n)
        idx = [slice(None)] * n
        idx[ctrl] = 1
        idx[tgt]  = 1
        sv[tuple(idx)] *= np.exp(1j * angle)
        return sv.reshape(-1)

    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def _rz(phi: float) -> np.ndarray:
        return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)]).astype(np.complex128)

    @staticmethod
    def _rx(theta: float) -> np.ndarray:
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    def simulate_ghz(self, n: int) -> np.ndarray:
        sv = np.zeros(2 ** n, dtype=np.complex128)
        sv[0] = 1.0
        sv = self._apply1(sv, self._H, 0, n)
        for i in range(n - 1):
            sv = self._apply_cx(sv, i, i + 1, n)
        return sv

    def simulate_qft(self, n: int) -> np.ndarray:
        sv = np.zeros(2 ** n, dtype=np.complex128)
        sv[0] = 1.0
        for i in range(n):
            sv = self._apply1(sv, self._H, i, n)
            for j in range(i + 1, n):
                angle = 2.0 * math.pi / (2 ** (j - i + 1))
                sv = self._apply_cr1(sv, angle, j, i, n)
        for i in range(n // 2):
            sv_r = sv.reshape([2] * n)
            sv_r = np.swapaxes(sv_r, i, n - i - 1)
            sv = sv_r.reshape(-1)
        return sv

    def simulate_rqc(self, n: int,
                     angles: List[float], depth: int) -> np.ndarray:
        sv = np.zeros(2 ** n, dtype=np.complex128)
        sv[0] = 1.0
        for d in range(depth):
            for i in range(n):
                base = (d * n + i) * 3
                sv = self._apply1(sv, self._rx(angles[base]),     i, n)
                sv = self._apply1(sv, self._ry(angles[base + 1]), i, n)
                sv = self._apply1(sv, self._rz(angles[base + 2]), i, n)
            for i in range(0, n - 1, 2):
                sv = self._apply_cx(sv, i, i + 1, n)
        return sv

    def simulate_vqe(self, n: int,
                     angles: List[float], layers: int) -> np.ndarray:
        sv = np.zeros(2 ** n, dtype=np.complex128)
        sv[0] = 1.0
        for l in range(layers):
            for i in range(n):
                base = (l * n + i) * 2
                sv = self._apply1(sv, self._ry(angles[base]),     i, n)
                sv = self._apply1(sv, self._rz(angles[base + 1]), i, n)
            for i in range(n - 1):
                sv = self._apply_cx(sv, i, i + 1, n)
        return sv

    def run(self, family: str, n: int, **kwargs) -> tuple:
        """Return (statevector, elapsed_seconds)."""
        t0 = time.perf_counter()
        if family == 'ghz':
            sv = self.simulate_ghz(n)
        elif family == 'qft':
            sv = self.simulate_qft(n)
        elif family == 'rqc':
            sv = self.simulate_rqc(n, kwargs['angles'], kwargs['depth'])
        elif family == 'vqe':
            sv = self.simulate_vqe(n, kwargs['angles'], kwargs['layers'])
        else:
            raise ValueError(f"Unknown family: {family}")
        elapsed = time.perf_counter() - t0
        return sv, elapsed


# ─────────────────────────────────────────────────────────────────────────────
#  Analytic ideal states (where known)
# ─────────────────────────────────────────────────────────────────────────────

def ideal_ghz(n: int) -> np.ndarray:
    """(|0…0⟩ + |1…1⟩) / √2"""
    sv = np.zeros(2 ** n, dtype=np.complex128)
    sv[0]  = 1.0 / math.sqrt(2)
    sv[-1] = 1.0 / math.sqrt(2)
    return sv


def ideal_qft_zero(n: int) -> np.ndarray:
    """QFT|0…0⟩ = uniform superposition = (1/√N) Σ|k⟩"""
    dim = 2 ** n
    return np.ones(dim, dtype=np.complex128) / math.sqrt(dim)


def fidelity(sv_sim: np.ndarray, sv_ref: np.ndarray) -> float:
    """State fidelity F = |⟨ψ_ref|ψ_sim⟩|²."""
    return float(np.abs(np.vdot(sv_ref, sv_sim)) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Noise model
# ─────────────────────────────────────────────────────────────────────────────

def build_noise_model(sq_err: float = 0.001,
                      tq_err: float = 0.002) -> 'cudaq.NoiseModel':
    """
    Depolarising noise on all injectable gates.

    Noise-injectable gate names in CUDA-Q (nvidia target):
      Single-qubit : h, r1, rx, ry, rz, x
      Two-qubit    : cx            ← cx only; 'swap' raises RuntimeError
      Note: swap decomposes internally to 3×CX; noise on cx covers those.
    """
    noise = cudaq.NoiseModel()
    for g in ('h', 'rx', 'ry', 'rz', 'r1', 'x'):
        noise.add_all_qubit_channel(g, cudaq.Depolarization1(sq_err))
    for g in ('cx', 'cr1'):
        noise.add_all_qubit_channel(g, cudaq.Depolarization2(tq_err))
    return noise


# ─────────────────────────────────────────────────────────────────────────────
#  Core benchmark function
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(family: str, n: int, cpu_sim: CPUSimulator,
                  max_cpu_q: int, trials: int,
                  use_noise: bool, sq_err: float, tq_err: float) -> dict:
    """
    Run one (family, n_qubits) benchmark point.
    Returns a dict of all measured metrics.
    """
    gc = gate_counts(family, n)
    rec = {
        'family':     family,
        'num_qubits': n,
        'gate_count': gc['total'],
        'gate_detail': gc,
    }

    # ── Pre-compute angles for parameterised kernels ──────────────────────
    extra_cudaq = {}   # extra positional args for cudaq.get_state()
    extra_cpu   = {}   # kwargs for CPUSimulator.run()

    if family == 'rqc':
        angles = rqc_angles(n, RQC_DEPTH)
        extra_cudaq = {'args': (n, RQC_DEPTH, angles)}
        extra_cpu   = {'angles': angles, 'depth': RQC_DEPTH}
    elif family == 'vqe':
        angles = vqe_angles(n, VQE_LAYERS)
        extra_cudaq = {'args': (n, VQE_LAYERS, angles)}
        extra_cpu   = {'angles': angles, 'layers': VQE_LAYERS}
    else:
        extra_cudaq = {'args': (n,)}
        extra_cpu   = {}

    kernel_map = {
        'ghz': ghz_kernel,
        'qft': qft_kernel,
        'rqc': rqc_kernel,
        'vqe': vqe_kernel,
    }
    kernel = kernel_map[family]

    # ── CUDA-Q GPU timing ─────────────────────────────────────────────────
    noise_obj = None
    if use_noise:
        noise_obj = build_noise_model(sq_err, tq_err)
        cudaq.set_noise(noise_obj)

    gpu_times = []
    last_sv   = None
    try:
        for t in range(WARMUP + trials):
            t0 = time.perf_counter()
            state = cudaq.get_state(kernel, *extra_cudaq['args'])
            gpu_times.append(time.perf_counter() - t0)
            if t == WARMUP:                   # keep first post-warmup state
                last_sv = np.array(state)
    finally:
        if noise_obj is not None:
            cudaq.unset_noise()

    gpu_times = gpu_times[WARMUP:]
    rec['gpu_time_s']    = float(np.median(gpu_times))
    rec['gpu_time_std_s'] = float(np.std(gpu_times))

    # ── CPU numpy timing ──────────────────────────────────────────────────
    rec['cpu_time_s'] = None
    cpu_sv = None
    if n <= max_cpu_q:
        cpu_times = []
        for t in range(WARMUP + trials):
            sv_t, dt = cpu_sim.run(family, n, **extra_cpu)
            if t >= WARMUP:
                cpu_times.append(dt)
                if t == WARMUP:
                    cpu_sv = sv_t
        rec['cpu_time_s'] = float(np.median(cpu_times))

    # ── Speedup ───────────────────────────────────────────────────────────
    if rec['cpu_time_s'] and rec['gpu_time_s'] > 0:
        rec['speedup'] = rec['cpu_time_s'] / rec['gpu_time_s']
    else:
        rec['speedup'] = None

    # ── Fidelity ──────────────────────────────────────────────────────────
    rec['fidelity'] = None
    rec['fidelity_ref'] = None

    if last_sv is not None:
        if family == 'ghz' and not use_noise:
            ideal = ideal_ghz(n)
            rec['fidelity'] = fidelity(last_sv, ideal)
            rec['fidelity_ref'] = 'analytic'
        elif family == 'qft' and not use_noise:
            ideal = ideal_qft_zero(n)
            rec['fidelity'] = fidelity(last_sv, ideal)
            rec['fidelity_ref'] = 'analytic'
        elif family == 'ghz' and use_noise:
            # Noisy: compare against ideal, degradation is expected
            ideal = ideal_ghz(n)
            rec['fidelity'] = fidelity(last_sv, ideal)
            rec['fidelity_ref'] = 'analytic_noisy'
        elif family == 'qft' and use_noise:
            ideal = ideal_qft_zero(n)
            rec['fidelity'] = fidelity(last_sv, ideal)
            rec['fidelity_ref'] = 'analytic_noisy'
        elif cpu_sv is not None:
            # RQC / VQE: use CPU simulation as reference
            rec['fidelity'] = fidelity(last_sv, cpu_sv)
            rec['fidelity_ref'] = 'cpu_numpy'

    # ── Energy ────────────────────────────────────────────────────────────
    rec['energy_j']  = GPU_TDP_W * GPU_UTIL_FRAC * rec['gpu_time_s']
    rec['energy_uj_per_gate'] = (
        rec['energy_j'] * 1e6 / gc['total'] if gc['total'] > 0 else None
    )

    # ── Memory footprint ──────────────────────────────────────────────────
    rec['sv_gb'] = (2 ** n * 16) / (1024 ** 3)

    return rec


# ─────────────────────────────────────────────────────────────────────────────
#  Console table printing
# ─────────────────────────────────────────────────────────────────────────────

def print_header(family: str):
    print(f"\n  ── {family.upper()} ──────────────────────────────────────────"
          f"───────────────────────────────")
    print(f"  {'Q':>3}  │ {'Gates':>6} │ {'CPU (s)':>9} │ "
          f"{'GPU (s)':>9} │ {'Speed':>6} │ "
          f"{'Fidelity':>12} │ {'Energy J':>9} │ {'µJ/gate':>8}")
    print("  " + "─" * 78)


def print_row(rec: dict):
    n    = rec['num_qubits']
    gc   = rec['gate_count']
    cpu  = rec.get('cpu_time_s')
    gpu  = rec.get('gpu_time_s')
    spd  = rec.get('speedup')
    fid  = rec.get('fidelity')
    enj  = rec.get('energy_j', 0.0)
    eug  = rec.get('energy_uj_per_gate')

    cpu_s = f"{cpu:.5f}" if cpu  else "   ---  "
    gpu_s = f"{gpu:.5f}" if gpu  else "   ---  "
    spd_s = f"{spd:.1f}×"  if spd  else "  --- "
    fid_s = f"{fid:.8f}"   if fid is not None else "    ---     "
    eug_s = f"{eug:.3f}"   if eug is not None else "   ---  "

    print(f"  {n:>3}q │ {gc:>6} │ {cpu_s:>9} │ "
          f"{gpu_s:>9} │ {spd_s:>6} │ "
          f"{fid_s:>12} │ {enj:>9.4f} │ {eug_s:>8}")


# ─────────────────────────────────────────────────────────────────────────────
#  Scaling sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_scaling(family: str, qubit_range,
                cpu_sim: CPUSimulator, max_cpu_q: int,
                trials: int, use_noise: bool,
                sq_err: float, tq_err: float) -> list:
    records = []
    print_header(family)
    for n in qubit_range:
        rec = benchmark_one(family, n, cpu_sim, max_cpu_q,
                            trials, use_noise, sq_err, tq_err)
        records.append(rec)
        print_row(rec)
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Plot generation (6-panel, publication quality)
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(all_results: dict, output_path: str = 'cudaq_benchmark.png'):
    if not PLOT_OK:
        print("\n[WARN] matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'CUDA-Q GPU Statevector Benchmark  —  RTX A4000  —  nvidia target',
        fontsize=13, fontweight='bold')

    COLORS = {'ghz': '#1565C0', 'qft': '#B71C1C',
              'rqc': '#E65100', 'vqe': '#2E7D32'}
    MARKERS = {'ghz': 'o', 'qft': 's', 'rqc': '^', 'vqe': 'D'}

    # ── Panel 1: GPU execution time scaling ───────────────────────────────
    ax = axes[0, 0]
    for fam, recs in all_results.items():
        pts = [(r['num_qubits'], r['gpu_time_s'])
               for r in recs if r.get('gpu_time_s')]
        if pts:
            qs, ts = zip(*pts)
            ax.semilogy(qs, ts, f"{MARKERS[fam]}-",
                        label=fam.upper(), color=COLORS[fam],
                        linewidth=2, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('GPU Kernel Time (s)')
    ax.set_title('GPU Execution Time Scaling')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: CPU vs GPU (QFT — most gate-dense family) ────────────────
    ax = axes[0, 1]
    qft = all_results.get('qft', [])
    cpu_pts  = [(r['num_qubits'], r['cpu_time_s'])
                for r in qft if r.get('cpu_time_s')]
    gpu_pts  = [(r['num_qubits'], r['gpu_time_s'])
                for r in qft if r.get('gpu_time_s')]
    if cpu_pts:
        qs, ts = zip(*cpu_pts)
        ax.semilogy(qs, ts, 'o-', label='CPU (numpy)', color='#B71C1C', linewidth=2)
    if gpu_pts:
        qs, ts = zip(*gpu_pts)
        ax.semilogy(qs, ts, 's-', label='GPU (CUDA-Q)', color='#1565C0', linewidth=2)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Time (s)')
    ax.set_title('CPU vs GPU  —  QFT')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Speedup ──────────────────────────────────────────────────
    ax = axes[0, 2]
    for fam, recs in all_results.items():
        pts = [(r['num_qubits'], r['speedup'])
               for r in recs if r.get('speedup')]
        if pts:
            qs, spd = zip(*pts)
            ax.plot(qs, spd, f"{MARKERS[fam]}-",
                    label=fam.upper(), color=COLORS[fam],
                    linewidth=2, markersize=5)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Speedup  (CPU time / GPU time)')
    ax.set_title('GPU Speedup over CPU')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Energy per gate ───────────────────────────────────────────
    ax = axes[1, 0]
    for fam, recs in all_results.items():
        pts = [(r['num_qubits'], r['energy_uj_per_gate'])
               for r in recs if r.get('energy_uj_per_gate')]
        if pts:
            qs, en = zip(*pts)
            ax.semilogy(qs, en, f"{MARKERS[fam]}-",
                        label=fam.upper(), color=COLORS[fam],
                        linewidth=2, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Energy per Gate (µJ)')
    ax.set_title('Energy Efficiency')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Gate count scaling ───────────────────────────────────────
    ax = axes[1, 1]
    for fam, recs in all_results.items():
        pts = [(r['num_qubits'], r['gate_count']) for r in recs]
        if pts:
            qs, gc = zip(*pts)
            ax.semilogy(qs, gc, f"{MARKERS[fam]}-",
                        label=fam.upper(), color=COLORS[fam],
                        linewidth=2, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Total Gate Count')
    ax.set_title('Circuit Complexity Scaling')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Fidelity (1-F infidelity, log scale) ─────────────────────
    ax = axes[1, 2]
    for fam, recs in all_results.items():
        pts = [(r['num_qubits'], 1.0 - r['fidelity'])
               for r in recs
               if r.get('fidelity') is not None and (1.0 - r['fidelity']) > 1e-16]
        if pts:
            qs, inf = zip(*pts)
            ax.semilogy(qs, inf, f"{MARKERS[fam]}-",
                        label=fam.upper(), color=COLORS[fam],
                        linewidth=2, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Infidelity  (1 − F)')
    ax.set_title('State Fidelity Error')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    all_recs = [r for recs in all_results.values() for r in recs]
    gpu_recs = [r for r in all_recs if r.get('gpu_time_s')]
    spd_recs = [r for r in all_recs if r.get('speedup')]
    fid_recs = [r for r in all_recs if r.get('fidelity') is not None]

    print(f"\n{'═' * 80}")
    print("  SUMMARY")
    print(f"{'═' * 80}")

    if gpu_recs:
        max_q = max(r['num_qubits'] for r in gpu_recs)
        total_e = sum(r.get('energy_j', 0) for r in gpu_recs)
        max_gc = max(r['gate_count'] for r in gpu_recs)
        print(f"  GPU benchmark points  : {len(gpu_recs)}")
        print(f"  Max qubits simulated  : {max_q}")
        print(f"  Max gate count        : {max_gc:,}")
        print(f"  Total energy consumed : {total_e:.3f} J")

    if spd_recs:
        speedups = [r['speedup'] for r in spd_recs]
        print(f"  Avg GPU speedup       : {np.mean(speedups):.1f}×")
        print(f"  Max GPU speedup       : {np.max(speedups):.1f}×")
        best = max(spd_recs, key=lambda r: r['speedup'])
        print(f"  Best speedup at       : {best['num_qubits']}q "
              f"({best['family'].upper()}) — {best['speedup']:.1f}×")

    if fid_recs:
        min_fid = min(r['fidelity'] for r in fid_recs)
        max_fid = max(r['fidelity'] for r in fid_recs)
        print(f"  Fidelity range        : [{min_fid:.8f}, {max_fid:.8f}]")

    print(f"{'═' * 80}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CUDA-Q Multi-Circuit GPU Benchmark')
    parser.add_argument('--max-qubits',     type=int,   default=MAX_QUBITS_DEFAULT)
    parser.add_argument('--max-cpu-qubits', type=int,   default=MAX_CPU_Q_DEFAULT)
    parser.add_argument('--trials',         type=int,   default=TRIALS)
    parser.add_argument('--min-qubits',     type=int,   default=2)
    parser.add_argument('--no-noise',       action='store_true')
    parser.add_argument('--no-plots',       action='store_true')
    parser.add_argument('--sq-err',         type=float, default=0.001)
    parser.add_argument('--tq-err',         type=float, default=0.002)
    parser.add_argument('--output',         type=str,
                        default='cudaq_benchmark_results.json')
    args = parser.parse_args()

    use_noise = not args.no_noise

    print("═" * 80)
    print("  CUDA-Q MULTI-CIRCUIT GPU BENCHMARK")
    print("  GHZ │ QFT │ RQC │ VQE  —  nvidia statevector target")
    print("═" * 80)
    print(f"  Date           : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Backend        : nvidia (GPU statevector)")
    print(f"  Qubit range    : {args.min_qubits}–{args.max_qubits}")
    print(f"  CPU cap        : ≤{args.max_cpu_qubits}q")
    print(f"  Trials         : {args.trials} (+ {WARMUP} warmup)")
    print(f"  Noise          : {'ON  (sq_err=%.3f, tq_err=%.3f)' % (args.sq_err, args.tq_err) if use_noise else 'OFF'}")
    print(f"  RQC depth      : {RQC_DEPTH}")
    print(f"  VQE layers     : {VQE_LAYERS}")
    print(f"  GPU TDP        : {GPU_TDP_W} W × {int(GPU_UTIL_FRAC*100)}% utilisation")

    # Memory check
    sv_gb = (2 ** args.max_qubits * 16) / (1024 ** 3)
    headroom = VRAM_GB - sv_gb - 0.5
    print(f"  Statevector    : {sv_gb:.2f} GB  (headroom {headroom:+.2f} GB)")
    if headroom < 0:
        print(f"  [WARN] Insufficient VRAM — reduce --max-qubits or use nvidia-mgpu")
    print("═" * 80)

    cudaq.set_target("nvidia")
    cpu_sim = CPUSimulator()
    qubit_range = range(args.min_qubits, args.max_qubits + 1)

    bm_kwargs = dict(
        cpu_sim   = cpu_sim,
        max_cpu_q = args.max_cpu_qubits,
        trials    = args.trials,
        use_noise = use_noise,
        sq_err    = args.sq_err,
        tq_err    = args.tq_err,
    )

    all_results = {}

    # ── Phase 1: GHZ ──────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  PHASE 1 / 4 — GHZ State  (H + CX chain, full-register entanglement)")
    all_results['ghz'] = run_scaling('ghz', qubit_range, **bm_kwargs)

    # ── Phase 2: QFT ──────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  PHASE 2 / 4 — Quantum Fourier Transform  (O(n²) two-qubit gates)")
    all_results['qft'] = run_scaling('qft', qubit_range, **bm_kwargs)

    # ── Phase 3: RQC ──────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  PHASE 3 / 4 — Random Quantum Circuit  "
          f"(depth={RQC_DEPTH}, Rx/Ry/Rz + CX brickwork)")
    all_results['rqc'] = run_scaling('rqc', qubit_range, **bm_kwargs)

    # ── Phase 4: VQE ──────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  PHASE 4 / 4 — VQE Ansatz  "
          f"(layers={VQE_LAYERS}, Ry/Rz + full CX ladder)")
    all_results['vqe'] = run_scaling('vqe', qubit_range, **bm_kwargs)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(all_results)

    # ── JSON export ───────────────────────────────────────────────────────
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved : {args.output}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        generate_plots(all_results,
                       args.output.replace('.json', '.png'))

    print(f"\n{'═' * 80}")
    print("  BENCHMARK COMPLETE")
    print(f"{'═' * 80}\n")


if __name__ == '__main__':
    main()