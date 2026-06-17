#!/usr/bin/env python3
"""
qldpc_qec_pipeline.py
════════════════════════════════════════════════════════════════════════════════
Full QEC Pipeline — Bivariate Bicycle (BB) / QLDPC Codes

  Stage 1  Code Construction    Hx, Hz via bivariate bicycle polynomial ring
  Stage 2  Error Sampling       CUDA-Q GPU statevector (n≤20) / NumPy (n>20)
  Stage 3  Syndrome Extraction  H @ e mod 2, vectorised
  Stage 4  FPGA Decode          XRT kernel on Alveo U55C → software BP fallback
  Stage 5  LER + Threshold      crossing-point threshold estimator
  Stage 6  Export               JSON + 2×2 publication PNG

Hardware  : NVIDIA RTX A4000 16 GB  +  Xilinx Alveo U55C (HBM2, 8 GB)
Toolchain : CUDA-Q  +  XRT 2.15  +  NumPy / Matplotlib
Platform  : Qniverse  |  C-DAC National Quantum Mission  (MeitY)
════════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, time, warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── optional scipy ────────────────────────────────────────────────────────────
try:
    from scipy.optimize import curve_fit
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ── CUDA-Q ────────────────────────────────────────────────────────────────────
try:
    import cudaq
    CUDAQ_OK = True
    print("[INIT] cudaq available — GPU statevector sampling enabled")
except ImportError:
    CUDAQ_OK = False
    print("[INIT] cudaq not found — falling back to NumPy sampling")

# ── XRT (Alveo U55C) ─────────────────────────────────────────────────────────
XRT_PYTHON_PATH = '/opt/xilinx/xrt/python'
sys.path.insert(0, XRT_PYTHON_PATH)
try:
    import pyxrt
    XRT_OK = True
    print("[INIT] pyxrt available — FPGA XRT interface ready")
except ImportError:
    XRT_OK = False
    print("[INIT] pyxrt not importable — software BP decoder active")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("qldpc_qec_output")
XCLBIN_PATH  = "qldpc_bp_decoder.xclbin"   # compiled FPGA bitstream (set if available)
OUTPUT_DIR.mkdir(exist_ok=True)

GPU_SV_LIMIT = 30    # max n for CUDA-Q statevector simulation
GPU_TDP_W    = 140.0
GPU_UTIL     = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BB CODE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

# Instances from Panteleev & Kalachev (2022) "Asymptotically Good Quantum and
# Linear Codes" and follow-on characterisations.
# Entry format: (l, m, a_terms, b_terms)
# a_terms / b_terms : list of (exp_x, exp_y) monomials for x^i * y^j
BB_LIBRARY: Dict[str, Tuple] = {
    # [[20, 2, ?]]  — n≤20 → CUDA-Q GPU statevector sampling active on RTX A4000
    # l=2, m=5: a(x,y)=1+y, b(x,y)=1+x  (toric-like; CSS verified, k=2)
    "BB[20,2]": (2, 5,
                 [(0, 0), (0, 1)],
                 [(0, 0), (1, 0)]),
    # [[36, 4, ?]]  — small validated code for fast sweeping / threshold comparison
    # l=3, m=6: a(x,y)=1+y+y², b(x,y)=1+y+x  (CSS verified, k=4)
    "BB[36,4]": (3, 6,
                 [(0, 0), (0, 1), (0, 2)],
                 [(0, 0), (0, 1), (1, 0)]),
    # [[72, 12, 6]]  — canonical Panteleev-Kalachev benchmark
    # l=m=6: a(x,y)=x³+y+y², b(x,y)=y³+x+x²  (CSS verified, k=12)
    "BB[72,12,6]": (6, 6,
                    [(3, 0), (0, 1), (0, 2)],
                    [(0, 3), (1, 0), (2, 0)]),
    # [[90, 8, 10]]  — larger scaling code (use if sweep time allows)
    "BB[90,8,10]": (5, 9,
                    [(0, 0), (1, 0), (0, 4)],
                    [(0, 0), (4, 0), (0, 1)]),
}


def gf2_rank(M: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2) via in-place Gaussian elimination."""
    M = M.copy().astype(np.uint8)
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = next((r for r in range(rank, rows) if M[r, col]), None)
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(rows):
            if r != rank and M[r, col]:
                M[r] ^= M[rank]
        rank += 1
    return rank


def build_bb_code(name: str) -> Dict:
    """
    Build Hx, Hz parity check matrices for a named BB code.
    Hx = [A | B]  (N×2N)  — Z-type stabilisers; detects X errors
    Hz = [B^T | A^T]      — X-type stabilisers; detects Z errors
    CSS condition: Hx @ Hz^T = 0 mod 2  (verified internally)
    """
    if name not in BB_LIBRARY:
        raise ValueError(f"Unknown BB code '{name}'. Available: {list(BB_LIBRARY)}")
    l, m, a_terms, b_terms = BB_LIBRARY[name]
    N = l * m   # block size

    def poly_to_circulant(terms: List[Tuple[int, int]]) -> np.ndarray:
        """Convert bivariate monomial list to lm×lm block-circulant matrix over GF(2)."""
        C = np.zeros((N, N), dtype=np.uint8)
        for rx in range(l):
            for ry in range(m):
                r = rx * m + ry
                for (di, dj) in terms:
                    cx = (rx + di) % l
                    cy = (ry + dj) % m
                    c = cx * m + cy
                    C[r, c] ^= 1
        return C

    A = poly_to_circulant(a_terms)
    B = poly_to_circulant(b_terms)

    Hx = np.hstack([A, B])       # (N, 2N)
    Hz = np.hstack([B.T, A.T])   # (N, 2N)

    # CSS validity check
    css_residual = (Hx @ Hz.T) % 2
    css_ok = bool(np.all(css_residual == 0))

    n     = 2 * N
    rx    = gf2_rank(Hx)
    rz    = gf2_rank(Hz)
    k     = n - rx - rz          # CSS formula: k = n - rank(Hx) - rank(Hz)

    # Row / column weights (uniform for BB codes)
    dv = int(np.sum(Hx[0]))                        # variable-node degree
    dc = int(np.sum(Hx[:, 0]) + np.sum(Hx[:, N]))  # check-node degree (approx)

    info = {
        "name": name, "l": l, "m": m,
        "n": n, "k": k, "N": N,
        "Hx": Hx, "Hz": Hz,
        "rank_Hx": rx, "rank_Hz": rz,
        "css_valid": css_ok,
        "dv": dv,
    }
    return info


# ══════════════════════════════════════════════════════════════════════════════
# 2.  NOISY ERROR SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def _classical_depolarising(n: int, p: float, shots: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Sample independent depolarising errors: each qubit gets X/Y/Z with prob p/3 each.
    Returns uint8 array (shots, n): 0=I, 1=X, 2=Y, 3=Z.
    """
    r = rng.random((shots, n))
    E = np.zeros((shots, n), dtype=np.uint8)
    E[r <  p / 3]                     = 1   # X
    E[(r >= p / 3) & (r < 2 * p / 3)] = 2   # Y
    E[(r >= 2 * p / 3) & (r < p)]     = 3   # Z
    return E


def _cudaq_sample_x_errors(n: int, p: float, shots: int,
                            rng: np.random.Generator,
                            timing: Dict) -> np.ndarray:
    """
    CUDA-Q GPU statevector: apply depolarising channel to n qubits,
    sample X-error pattern from Z-basis measurements.
    Measurement outcome |1⟩ corresponds to an X-type (X or Y) error.
    """
    cudaq.set_target("nvidia")
    noise = cudaq.NoiseModel()
    dp = cudaq.DepolarizationChannel(p)
    for q_idx in range(n):
        noise.add_channel("x", [q_idx], dp)

    @cudaq.kernel
    def noisy_x_circuit(n_q: int):
        q = cudaq.qvector(n_q)
        for i in range(n_q):
            x(q[i])   # flip to |1⟩; depolarising noise applied here
        mz(q)         # measure — flip back to 0 means X or Y error occurred

    t0 = time.perf_counter()
    result = cudaq.sample(noisy_x_circuit, n,
                          shots_count=shots, noise_model=noise)
    timing['sample_gpu_s'] = time.perf_counter() - t0
    timing['sample_backend'] = 'cudaq-nvidia-sv'

    # Decode bitstrings → X-error array
    E_x = np.zeros((shots, n), dtype=np.uint8)
    idx = 0
    for bitstring, count in result.items():
        for _ in range(count):
            if idx >= shots:
                break
            # bit='0' means qubit stayed in |0⟩ after x + noise → X or Y error
            for q_idx, bit in enumerate(reversed(bitstring)):
                if bit == '0':
                    E_x[idx, q_idx] = 1
            idx += 1
    timing['cudaq_shots_collected'] = idx
    return E_x


def sample_pauli_errors(n: int, p: float, shots: int,
                        rng: np.random.Generator,
                        timing: Dict) -> np.ndarray:
    """
    Route error sampling: CUDA-Q GPU for n ≤ GPU_SV_LIMIT, NumPy otherwise.
    Returns (shots, n) uint8 Pauli array (0=I, 1=X, 2=Y, 3=Z).
    """
    if CUDAQ_OK and n <= GPU_SV_LIMIT:
        try:
            # GPU: get X-error component from CUDA-Q statevector
            E_x = _cudaq_sample_x_errors(n, p, shots, rng, timing)
            # Z-error component: classical (symmetric by depolarising model)
            r   = rng.random((shots, n))
            E_z = ((r < p / 3) | ((r >= p / 3) & (r < 2 * p / 3))).astype(np.uint8)
            # Reconstruct full Pauli: X=1,Y=2,Z=3 from X and Z components
            E = np.zeros((shots, n), dtype=np.uint8)
            E[E_x == 1]                        = 1  # X
            E[(E_x == 0) & (E_z == 1)]         = 3  # Z
            E[(E_x == 1) & (E_z == 1)]         = 2  # Y  (X and Z both set → Y)
            return E
        except Exception as exc:
            print(f"  [WARN] CUDA-Q error ({exc}) — falling back to NumPy")

    t0 = time.perf_counter()
    E  = _classical_depolarising(n, p, shots, rng)
    timing['sample_cpu_s']    = time.perf_counter() - t0
    timing['sample_backend']  = 'numpy-cpu'
    return E


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYNDROME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_syndromes(H: np.ndarray, Ex: np.ndarray, Ez: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute parity syndromes for X and Z error components.
    sx = Hx @ Ex mod 2   (Z stabilisers detect X errors)
    sz = Hz @ Ez mod 2   (X stabilisers detect Z errors)
    H shape: (n_checks, n_data).  Ex/Ez shape: (shots, n_data).
    Returns sx, sz each (shots, n_checks).
    """
    sx = (Ex @ H.T) % 2
    sz = (Ez @ H.T) % 2
    return sx.astype(np.uint8), sz.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BELIEF PROPAGATION DECODER  (software reference / FPGA fallback)
# ══════════════════════════════════════════════════════════════════════════════

class BPDecoder:
    """
    Flooding-schedule min-sum belief propagation decoder for binary LDPC.
    Used as the software reference and as fallback when no FPGA xclbin is loaded.

    Algorithm: Fossorier et al. normalised min-sum (scale factor α).
    """

    def __init__(self, H: np.ndarray, max_iter: int = 50,
                 ms_scale: float = 0.75):
        self.H        = H.astype(np.uint8)
        self.m, self.n = H.shape
        self.max_iter = max_iter
        self.ms_scale = ms_scale
        # Pre-compute Tanner graph neighbour lists
        self.cn_nbrs  = [np.where(H[i])[0] for i in range(self.m)]
        self.vn_nbrs  = [np.where(H[:, j])[0] for j in range(self.n)]

    def _decode_one(self, syndrome: np.ndarray, llr_ch: np.ndarray) -> np.ndarray:
        """Decode a single syndrome vector."""
        # Initialise variable→check messages from channel LLRs
        v2c = {}
        for i in range(self.m):
            for j in self.cn_nbrs[i]:
                v2c[(i, j)] = float(llr_ch[j])

        est = np.zeros(self.n, dtype=np.uint8)

        for _ in range(self.max_iter):
            # ── Check-node update (min-sum) ──────────────────────────
            c2v = {}
            for i in range(self.m):
                nbrs = self.cn_nbrs[i]
                parity_sign = (-1) ** int(syndrome[i])
                for j in nbrs:
                    others = [v2c[(i, k)] for k in nbrs if k != j]
                    if not others:
                        c2v[(i, j)] = 0.0
                        continue
                    mag = self.ms_scale * min(abs(x) for x in others)
                    sgn = parity_sign
                    for x in others:
                        sgn *= (1.0 if x > 0 else -1.0)
                    c2v[(i, j)] = sgn * mag

            # ── Variable-node update + hard decision ─────────────────
            for j in range(self.n):
                total = llr_ch[j] + sum(c2v[(i, j)] for i in self.vn_nbrs[j])
                est[j] = 0 if total >= 0 else 1
                for i in self.vn_nbrs[j]:
                    v2c[(i, j)] = total - c2v[(i, j)]

            # ── Convergence check ────────────────────────────────────
            if np.all((self.H @ est) % 2 == syndrome):
                return est

        return est   # return best hard decision on timeout

    def decode_batch(self, syndromes: np.ndarray, p: float,
                     timing: Dict) -> np.ndarray:
        shots    = syndromes.shape[0]
        llr_ch   = np.full(self.n, np.log((1 - p) / max(p, 1e-12)))
        corr     = np.zeros((shots, self.n), dtype=np.uint8)
        t0 = time.perf_counter()
        for s in range(shots):
            corr[s] = self._decode_one(syndromes[s], llr_ch.copy())
        timing['decode_s']       = time.perf_counter() - t0
        timing['decode_backend'] = 'software-BP-minsum'
        return corr


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FPGA XRT DECODER INTERFACE  (Alveo U55C)
# ══════════════════════════════════════════════════════════════════════════════

class FPGADecoder:
    """
    XRT interface to the FPGA QLDPC-BP decoder kernel on Alveo U55C.

    Protocol
    --------
    1.  Load xclbin via pyxrt.device / pyxrt.uuid / pyxrt.kernel
    2.  Allocate HBM-mapped buffers for syndrome input and correction output
    3.  For each syndrome: write → sync to device → run kernel → sync from device → read

    If no xclbin is present or pyxrt is unavailable, falls back transparently
    to software BP with identical API so the caller is unaware.

    Future work: batch multiple syndromes per kernel invocation (HLS DATAFLOW)
    to saturate HBM bandwidth and reduce PCIe round-trip overhead.
    """

    def __init__(self, H: np.ndarray,
                 xclbin: str = XCLBIN_PATH,
                 bp_kwargs: Optional[Dict] = None):
        self.H            = H
        self.m, self.n    = H.shape
        self.fpga_active  = False
        self.xclbin_path  = xclbin

        if XRT_OK and os.path.exists(xclbin):
            try:
                self._init_fpga(xclbin)
            except Exception as exc:
                print(f"  [FPGA] xclbin load failed: {exc}")
                print(  "  [FPGA] Falling back to software BP")
        else:
            reason = "xclbin not found" if not os.path.exists(xclbin) else "pyxrt unavailable"
            print(f"  [FPGA] {reason} — software BP decoder active")
            print(f"         (place compiled bitstream at '{xclbin}' to enable FPGA decode)")

        kw = bp_kwargs or {}
        self.bp = BPDecoder(H, **kw)

    def _init_fpga(self, xclbin: str):
        self.device   = pyxrt.device(0)
        self.uuid     = self.device.load_xclbin(xclbin)
        self.kernel   = pyxrt.kernel(self.device, self.uuid,
                                     "qldpc_bp_decoder:{qldpc_bp_decoder_1}")
        # Allocate HBM-mapped buffers (U55C: 8 × 1 GB pseudo-channels)
        syn_bytes = self.m * np.dtype(np.uint8).itemsize
        cor_bytes = self.n * np.dtype(np.uint8).itemsize
        self.buf_syn = pyxrt.bo(self.device, syn_bytes,
                                pyxrt.bo.flags.normal, self.kernel.group_id(0))
        self.buf_cor = pyxrt.bo(self.device, cor_bytes,
                                pyxrt.bo.flags.normal, self.kernel.group_id(1))
        self.fpga_active = True
        print(f"  [FPGA] ✓ Loaded {xclbin} on Alveo U55C — hardware decode active")

    def _decode_fpga_batch(self, syndromes: np.ndarray, p: float,
                           timing: Dict) -> np.ndarray:
        """Stream syndromes to U55C HBM, invoke kernel, read corrections."""
        shots      = syndromes.shape[0]
        corrections = np.zeros((shots, self.n), dtype=np.uint8)
        # Fixed-point channel LLR in Q8.8 format for the FPGA kernel
        llr_fp = int(np.log((1 - p) / max(p, 1e-12)) * 256)

        t0 = time.perf_counter()
        for s in range(shots):
            # Write syndrome → DMA to device
            self.buf_syn.write(syndromes[s].tobytes())
            self.buf_syn.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            # Run FPGA kernel
            run = self.kernel(self.buf_syn, self.buf_cor,
                              np.uint32(self.m), np.uint32(self.n),
                              np.int32(llr_fp))
            run.wait()
            # DMA back → read correction
            self.buf_cor.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            corrections[s] = np.frombuffer(
                self.buf_cor.read(self.n), dtype=np.uint8)

        timing['decode_s']       = time.perf_counter() - t0
        timing['decode_backend'] = f'fpga-xrt-U55C'
        return corrections

    def decode_batch(self, syndromes: np.ndarray, p: float,
                     timing: Dict) -> np.ndarray:
        if self.fpga_active:
            return self._decode_fpga_batch(syndromes, p, timing)
        return self.bp.decode_batch(syndromes, p, timing)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  LOGICAL ERROR DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def count_logical_errors(Hz: np.ndarray,
                         Ex:   np.ndarray, cx: np.ndarray,
                         Ez:   np.ndarray, cz: np.ndarray) -> int:
    """
    Count shots where the residual error (true ⊕ correction) is a logical operator.

    A residual R_x is a logical X error iff:
      (a) Hz @ R_x = 0 mod 2  — undetected by Z stabilisers (in ker Hz)
      (b) R_x ≠ 0             — non-trivial

    Note: this counts stabiliser elements as logical errors (conservative upper
    bound on LER, acceptable for threshold estimation).  Full treatment requires
    knowing the actual logical operator representatives via Smith Normal Form.
    """
    Rx = (Ex ^ cx).astype(np.uint8)   # (shots, n)
    Rz = (Ez ^ cz).astype(np.uint8)

    # Syndromes of residuals — should be 0 if decoder succeeded
    syn_x = (Rx @ Hz.T) % 2   # (shots, m)
    syn_z = (Rz @ Hz.T) % 2   # reuse Hz for Z errors (symmetric CSS code)

    # Logical error: syndrome-free but non-trivial residual
    log_x = np.sum(
        (np.sum(syn_x, axis=1) == 0) & (np.sum(Rx, axis=1) > 0)
    )
    log_z = np.sum(
        (np.sum(syn_z, axis=1) == 0) & (np.sum(Rz, axis=1) > 0)
    )
    # Return worst-case (X or Z) per shot
    return int(max(log_x, log_z))


# ══════════════════════════════════════════════════════════════════════════════
# 7.  LER EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    code_name:      str
    p_phys:         float
    shots:          int
    logical_errors: int
    ler:            float
    ler_ci_lo:      float
    ler_ci_hi:      float
    timing:         Dict = field(default_factory=dict)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat   = k / n
    denom   = 1 + z ** 2 / n
    centre  = (p_hat + z ** 2 / (2 * n)) / denom
    margin  = z * (p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) ** 0.5 / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def run_experiment(code: Dict, decoder: FPGADecoder,
                   p_phys: float, shots: int,
                   rng: np.random.Generator) -> ExperimentResult:
    """Run one LER data point: sample → syndrome → decode → logical-error check."""
    n  = code['n']
    Hx = code['Hx']
    Hz = code['Hz']
    timing: Dict = {}

    # ── Stage 2: Sample errors ──────────────────────────────────────────────
    E  = sample_pauli_errors(n, p_phys, shots, rng, timing)
    Ex = ((E == 1) | (E == 2)).astype(np.uint8)   # X or Y component
    Ez = ((E == 2) | (E == 3)).astype(np.uint8)   # Y or Z component

    # ── Stage 3: Syndrome extraction ────────────────────────────────────────
    sx, sz = extract_syndromes(Hx, Ex, Ez)

    # ── Stage 4: Decode X errors (using Z syndromes via Hx decoder) ─────────
    timing_x: Dict = {}
    timing_z: Dict = {}
    cx = decoder.decode_batch(sx, p_phys, timing_x)
    cz = decoder.decode_batch(sz, p_phys, timing_z)

    timing['decode_x_s']       = timing_x.get('decode_s', 0)
    timing['decode_z_s']       = timing_z.get('decode_s', 0)
    timing['decode_backend']   = timing_x.get('decode_backend', '?')
    timing['sample_backend']   = timing.get('sample_backend',
                                  timing.get('sample_backend', 'numpy-cpu'))

    # ── Stage 5: Logical error count ────────────────────────────────────────
    n_logical = count_logical_errors(Hz, Ex, cx, Ez, cz)
    ler        = n_logical / shots
    lo, hi     = wilson_ci(n_logical, shots)

    return ExperimentResult(
        code_name=code['name'], p_phys=p_phys,
        shots=shots, logical_errors=n_logical,
        ler=ler, ler_ci_lo=lo, ler_ci_hi=hi,
        timing=timing,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  THRESHOLD ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

def estimate_threshold(p_vals: np.ndarray,
                       ler_map: Dict[str, np.ndarray]) -> Optional[float]:
    """
    Estimate decoder threshold p_th as the crossing of LER curves for two
    different code sizes.  Uses linear interpolation at the first zero-crossing
    of (LER_small - LER_large).
    """
    names = list(ler_map.keys())
    if len(names) < 2:
        return None
    ler1, ler2 = ler_map[names[0]], ler_map[names[1]]
    diff        = ler1 - ler2
    crossings   = np.where(np.diff(np.sign(diff)))[0]
    if len(crossings) == 0:
        return None
    i = crossings[0]
    if i + 1 >= len(p_vals):
        return float(p_vals[i])
    x0, x1 = p_vals[i], p_vals[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0 + 1e-30))


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PLOTTING  (publication-quality 2×2 figure)
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = ['#58a6ff', '#ff7c43', '#a29bfe', '#00b894', '#fd79a8', '#e17055']


def _style_ax(ax):
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#f0f6fc')
    ax.grid(True, color='#21262d', linewidth=0.5, alpha=0.9)


def plot_results(all_results:  Dict[str, List[ExperimentResult]],
                 p_vals:       np.ndarray,
                 threshold:    Optional[float],
                 timing_table: List[Dict],
                 save_path:    Path):
    """Build and save 2×2 publication figure."""
    fig = plt.figure(figsize=(14, 11), dpi=150)
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_ler   = fig.add_subplot(gs[0, 0])
    ax_thr   = fig.add_subplot(gs[0, 1])
    ax_time  = fig.add_subplot(gs[1, 0])
    ax_ener  = fig.add_subplot(gs[1, 1])
    for ax in [ax_ler, ax_thr, ax_time, ax_ener]:
        _style_ax(ax)

    # ── Panel A: LER vs p_phys ─────────────────────────────────────────────
    for idx, (cname, res_list) in enumerate(all_results.items()):
        col   = PALETTE[idx % len(PALETTE)]
        lers  = np.array([r.ler       for r in res_list])
        lo    = np.array([r.ler_ci_lo for r in res_list])
        hi    = np.array([r.ler_ci_hi for r in res_list])
        label = cname.replace("BB", "BB-code")
        ax_ler.plot(p_vals * 100, np.where(lers > 0, lers, np.nan),
                    'o-', color=col, linewidth=1.6, markersize=4, label=label)
        ax_ler.fill_between(p_vals * 100, lo, hi,
                            color=col, alpha=0.12)
    if threshold and 0 < threshold < 1:
        ax_ler.axvline(threshold * 100, color='#ff4444', linestyle='--',
                       linewidth=1.5, alpha=0.85,
                       label=f"p_th ≈ {threshold*100:.2f}%")
    ax_ler.set_yscale('log')
    ax_ler.set_xlabel("Physical Error Rate  p (%)")
    ax_ler.set_ylabel("Logical Error Rate (LER)")
    ax_ler.set_title("A — LER vs Physical Error Rate")
    ax_ler.legend(fontsize=7, framealpha=0.3,
                  labelcolor='#c9d1d9', facecolor='#0d1117')

    # ── Panel B: Threshold zoom ────────────────────────────────────────────
    for idx, (cname, res_list) in enumerate(all_results.items()):
        col  = PALETTE[idx % len(PALETTE)]
        lers = [r.ler for r in res_list]
        ax_thr.plot(p_vals * 100, lers, 'o-', color=col,
                    linewidth=1.6, markersize=4,
                    label=cname.replace("BB", ""))
    if threshold and 0 < threshold < 1:
        ax_thr.axvline(threshold * 100, color='#ff4444',
                       linestyle='--', linewidth=1.5, alpha=0.85)
        ax_thr.set_title(f"B — Threshold  p_th ≈ {threshold*100:.2f}%")
        win = 0.05
        ax_thr.set_xlim(max(0, (threshold - win) * 100),
                        min(100, (threshold + win) * 100))
    else:
        ax_thr.set_title("B — Threshold Region")
    ax_thr.set_xlabel("Physical Error Rate  p (%)")
    ax_thr.set_ylabel("Logical Error Rate (LER)")
    ax_thr.legend(fontsize=7, framealpha=0.3,
                  labelcolor='#c9d1d9', facecolor='#0d1117')

    # ── Panel C: Decode latency ────────────────────────────────────────────
    if timing_table:
        by_code: Dict[str, List] = {}
        for t in timing_table:
            by_code.setdefault(t['code'], []).append(t)
        for idx, (cname, pts) in enumerate(by_code.items()):
            pts.sort(key=lambda x: x['p'])
            ps   = [t['p'] * 100 for t in pts]
            ts   = [t['decode_ms_per_shot'] for t in pts]
            col  = PALETTE[idx % len(PALETTE)]
            ax_time.plot(ps, ts, 'o-', color=col, linewidth=1.4,
                         markersize=4, label=cname.replace("BB", ""))
    ax_time.set_xlabel("Physical Error Rate  p (%)")
    ax_time.set_ylabel("Decode Latency / Shot  (ms)")
    ax_time.set_yscale('log')
    ax_time.set_title("C — Decode Latency  (FPGA→BP fallback)")
    ax_time.legend(fontsize=7, framealpha=0.3,
                   labelcolor='#c9d1d9', facecolor='#0d1117')

    # ── Panel D: Energy per shot ───────────────────────────────────────────
    if timing_table:
        for idx, (cname, pts) in enumerate(by_code.items()):
            pts.sort(key=lambda x: x['p'])
            ps   = [t['p'] * 100 for t in pts]
            ejs  = [t['energy_uj_per_shot'] for t in pts]
            col  = PALETTE[idx % len(PALETTE)]
            ax_ener.plot(ps, ejs, 's--', color=col, linewidth=1.2,
                         markersize=4, label=cname.replace("BB", ""))
    ax_ener.set_xlabel("Physical Error Rate  p (%)")
    ax_ener.set_ylabel("Energy / Shot  (µJ)")
    ax_ener.set_title("D — GPU Energy per Decoded Shot")
    ax_ener.legend(fontsize=7, framealpha=0.3,
                   labelcolor='#c9d1d9', facecolor='#0d1117')

    fig.suptitle(
        "BB/QLDPC QEC Pipeline  ·  CUDA-Q (RTX A4000)  +  Alveo U55C FPGA  ·  C-DAC Qniverse",
        fontsize=9.5, color='#f0f6fc', y=0.98
    )
    plt.savefig(save_path, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PLOT] Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SUMMARY TABLE PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_table(all_results: Dict, timing_table: List[Dict]):
    w = 104
    hdr = ("  {:<20s}  {:>6s}  {:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>12s}  {:>12s}"
           .format("Code", "n", "p(%)", "LER", "CI_lo", "CI_hi",
                   "Dec_ms/shot", "Energy_µJ"))
    print("\n" + "─" * w)
    print(hdr)
    print("─" * w)
    for cname, res_list in all_results.items():
        for res in res_list:
            t_entry = next((t for t in timing_table
                            if t['code'] == cname and
                            abs(t['p'] - res.p_phys) < 1e-9), {})
            dec_ms = t_entry.get('decode_ms_per_shot', 0)
            ej     = t_entry.get('energy_uj_per_shot', 0)
            print(("  {:<20s}  {:>6d}  {:>6.2f}  {:>10.5f}  {:>10.5f}  {:>10.5f}"
                   "  {:>12.3f}  {:>12.2f}").format(
                cname, res_list[0].shots // len(res_list) if res_list else 0,
                res.p_phys * 100, res.ler, res.ler_ci_lo, res.ler_ci_hi,
                dec_ms, ej))
    print("─" * w)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 72)
    print("  QLDPC QEC PIPELINE  ·  BB Codes  ·  GPU + FPGA")
    print("  C-DAC National Quantum Mission  ·  Qniverse Platform")
    print(f"  CUDA-Q : {'✓' if CUDAQ_OK else '✗'}   XRT/pyxrt : {'✓' if XRT_OK else '✗'}   SciPy : {'✓' if SCIPY_OK else '✗'}")
    print("═" * 72)

    rng = np.random.default_rng(42)

    # ── Configuration ────────────────────────────────────────────────────────
    CODES_TO_RUN = ["BB[36,4]", "BB[72,12,6]"]
    P_VALS       = np.linspace(0.01, 0.20, 16)
    SHOTS        = 400    # raise to 2000+ for publication-quality CI bars
    BP_MAX_ITER  = 50
    MS_SCALE     = 0.75

    # ── Stage 1: Build codes ─────────────────────────────────────────────────
    print("\n[1] Building BB parity check matrices …")
    codes: Dict[str, Dict] = {}
    for name in CODES_TO_RUN:
        c = build_bb_code(name)
        codes[name] = c
        css_sym = "✓" if c['css_valid'] else "✗ VIOLATED"
        print(f"    {name:<20s}  n={c['n']:4d}  k={c['k']:3d}  "
              f"rank(Hx)={c['rank_Hx']:4d}  CSS={css_sym}  dv≈{c['dv']}")

    # ── Stage 4 setup: decoders ──────────────────────────────────────────────
    print("\n[4] Initialising decoders …")
    decoders: Dict[str, FPGADecoder] = {}
    for name, code in codes.items():
        decoders[name] = FPGADecoder(
            code['Hx'],
            xclbin=XCLBIN_PATH,
            bp_kwargs={'max_iter': BP_MAX_ITER, 'ms_scale': MS_SCALE},
        )

    # ── CUDA-Q GPU demo (n≤20 → statevector sampling on RTX A4000) ────────────
    if CUDAQ_OK:
        print("\n[CUDAQ] GPU statevector sampling demo — BB[20,2] (n=20, k=2)")
        cudaq_demo_code = build_bb_code("BB[20,2]")
        cudaq_demo_dec  = FPGADecoder(cudaq_demo_code['Hx'],
                                      bp_kwargs={'max_iter': 40, 'ms_scale': 0.75})
        cudaq_demo_p    = 0.05
        t_demo = time.perf_counter()
        r_demo = run_experiment(cudaq_demo_code, cudaq_demo_dec,
                                cudaq_demo_p, 100, rng)
        t_demo = time.perf_counter() - t_demo
        print(f"    n=20  p={cudaq_demo_p*100:.0f}%  "
              f"LER={r_demo.ler:.4f}  "
              f"backend={r_demo.timing.get('sample_backend','?')}  "
              f"wall={t_demo:.2f}s")
    else:
        print("\n[CUDAQ] Not available — install via: pip install cuda-quantum")
        print("         (GPU statevector path activates for n ≤ GPU_SV_LIMIT=20)")

    print(f"\n[5] LER sweep: {len(P_VALS)} error rates × {len(CODES_TO_RUN)} codes "
          f"× {SHOTS} shots …")

    all_results:  Dict[str, List[ExperimentResult]] = {n: [] for n in CODES_TO_RUN}
    timing_table: List[Dict]                         = []
    wall_start = time.perf_counter()

    for name in CODES_TO_RUN:
        code = codes[name]
        dec  = decoders[name]
        print(f"\n  ── {name}  (n={code['n']}, k={code['k']}) ──")
        print(f"  {'p(%)':>6}  {'LER':>8}  {'CI_lo':>8}  {'CI_hi':>8}"
              f"  {'Sample':>10}  {'Decode':>12}  {'E(µJ/sh)':>10}")
        print("  " + "─" * 76)

        for p in P_VALS:
            res       = run_experiment(code, dec, p, SHOTS, rng)
            all_results[name].append(res)

            t_samp    = res.timing.get('sample_gpu_s',
                        res.timing.get('sample_cpu_s', 0))
            t_dec     = res.timing.get('decode_x_s', 0) + \
                        res.timing.get('decode_z_s', 0)
            total_s   = t_samp + t_dec
            # Energy: GPU at 140 W × 70% utilisation
            energy_j  = total_s * GPU_TDP_W * GPU_UTIL
            energy_uj = energy_j / SHOTS * 1e6   # µJ per shot
            dec_ms    = t_dec / SHOTS * 1e3       # ms per shot

            samp_b    = res.timing.get('sample_backend', 'numpy-cpu')
            dec_b     = res.timing.get('decode_backend', '?')

            print(f"  {p*100:>6.2f}  {res.ler:>8.5f}  {res.ler_ci_lo:>8.5f}"
                  f"  {res.ler_ci_hi:>8.5f}  {samp_b:>10s}  {dec_b:>12s}"
                  f"  {energy_uj:>10.2f}")

            timing_table.append({
                'code':               name,
                'n':                  code['n'],
                'p':                  float(p),
                'decode_ms_per_shot': float(dec_ms),
                'energy_uj_per_shot': float(energy_uj),
                'sample_backend':     samp_b,
                'decode_backend':     dec_b,
            })

    wall_total = time.perf_counter() - wall_start
    print(f"\n  [TIMING] Total wall time: {wall_total:.1f} s")

    # ── Threshold ────────────────────────────────────────────────────────────
    ler_map = {
        name: np.array([r.ler for r in res_list])
        for name, res_list in all_results.items()
    }
    threshold = estimate_threshold(P_VALS, ler_map)
    print("\n" + "═" * 72)
    if threshold and 0 < threshold < 1:
        print(f"  THRESHOLD   p_th ≈ {threshold * 100:.3f}%")
    else:
        print("  THRESHOLD   Not detected in this p range — extend sweep or increase shots")

    # ── Timing + energy table ─────────────────────────────────────────────────
    print_table(all_results, timing_table)

    # ── Export JSON ──────────────────────────────────────────────────────────
    json_path = OUTPUT_DIR / "qldpc_qec_results.json"
    export = {
        "metadata": {
            "codes":         CODES_TO_RUN,
            "p_vals":        P_VALS.tolist(),
            "shots":         SHOTS,
            "bp_max_iter":   BP_MAX_ITER,
            "ms_scale":      MS_SCALE,
            "gpu_tdp_w":     GPU_TDP_W,
            "gpu_util":      GPU_UTIL,
            "cudaq_ok":      CUDAQ_OK,
            "xrt_ok":        XRT_OK,
            "gpu_sv_limit":  GPU_SV_LIMIT,
            "xclbin":        XCLBIN_PATH,
        },
        "threshold_pct": threshold * 100 if threshold else None,
        "results": {
            name: [
                {k: v for k, v in asdict(r).items() if k != 'timing'}
                for r in res_list
            ]
            for name, res_list in all_results.items()
        },
        "timing_table": timing_table,
    }
    with open(json_path, 'w') as f:
        json.dump(export, f, indent=2, default=str)
    print(f"\n  [JSON] Saved → {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = OUTPUT_DIR / "qldpc_qec_results.png"
    plot_results(all_results, P_VALS, threshold, timing_table, plot_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  PIPELINE COMPLETE")
    print("═" * 72)
    for name, res_list in all_results.items():
        min_ler  = min(r.ler for r in res_list)
        best_p   = P_VALS[np.argmin([r.ler for r in res_list])]
        backends = set(r.timing.get('decode_backend', '?') for r in res_list)
        print(f"  {name:<20s}  min LER={min_ler:.5f}  best p={best_p*100:.2f}%"
              f"  decoder={', '.join(backends)}")
    if threshold:
        print(f"  {'Threshold':<20s}  p_th = {threshold*100:.3f}%")
    print(f"  Output → {OUTPUT_DIR.resolve()}")
    print("═" * 72)


if __name__ == "__main__":
    main()