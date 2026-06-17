#!/usr/bin/env python3
# python -u cudaqx_qec_v6.py 2>&1 | tee v6.log
"""
=============================================================================
  CUDA-QX QEC v6  —  cudaq 0.13.0 + cudaq-qec 0.5.0
  NVIDIA RTX A4000  |  C-DAC Qniverse / National Quantum Mission
  ─────────────────────────────────────────────────────────────────────────
  Fixes over v5
  ─────────────────────────────────────────────────────────────────────────
  1. LOGICAL ERROR DETECTION
     v5 used  np.sum(res.result) % 2  which is wrong (sums soft floats).
     v6 thresholds to a hard correction vector, then applies the DEM
     observable_flips_matrix for a proper GF2 logical-error check.
     Falls back to parity of correction if that attribute is missing.

  2. SAMPLE_MEMORY_CIRCUIT CALL
     v5 used positional args inconsistently.
     v6 uses keyword args (op=, numRounds=, numShots=, noise=) everywhere.

  3. DEM FUNCTION DETECTION
     Cleaner: tries both names once at startup, exits if neither found.

  4. LAYOUT DETECTION
     Single robust helper handles both known layouts without the verbose
     probe section.  Layout is determined per-call from array shape.

  5. CUSTOM CODE CLASS  (@qec.code  0.5.0 API)
     v5 used @staticmethod which breaks the base-class initialiser.
     v6 uses instance methods + __init__ calling qec.Code.__init__.
     Attribute names corrected:  pauli_observables (not observables).
     Stabilizer round kernel returns list[bool] and resets ancillas.

  6. nv-relaybp-decoder
     Handled with a specific string check so "decoder not found" prints
     a clear skip message instead of a full traceback.

  7. THRESHOLD SCAN extended to d ∈ {3, 5, 7}.

  8. ALL STEPS share a single run_memory_experiment() / decode_shots()
     pipeline — no duplicated logic.
=============================================================================
"""

import os, sys, time, traceback
import numpy as np

# ── stderr logger (unbuffered, visible when stdout is piped) ──────────────────
def log(msg=""):
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()

# ── Imports ───────────────────────────────────────────────────────────────────
import cudaq
import cudaq_qec as qec
from cudaq_qec import patch

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Banner ────────────────────────────────────────────────────────────────────
log("=" * 65)
log("  CUDA-QX QEC v6  |  cudaq 0.13.0 + cudaq-qec 0.5.0")
log("  C-DAC Qniverse  |  National Quantum Mission  |  RTX A4000")
log("=" * 65)

cudaq.set_target("stim")

# ── DEM function: try both API names ─────────────────────────────────────────
_DEM_CANDIDATES = ["dem_from_memory_circuit", "z_dem_from_memory_circuit"]
dem_fn = None
for _name in _DEM_CANDIDATES:
    if hasattr(qec, _name):
        dem_fn = getattr(qec, _name)
        log(f"[INFO]  DEM function  →  qec.{_name}")
        break
if dem_fn is None:
    log("[FATAL] Neither dem_from_memory_circuit nor z_dem_from_memory_circuit "
        "found in cudaq_qec. Exiting.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def to_detector_events(syns_raw: np.ndarray,
                       n_shots:  int,
                       n_rounds: int,
                       n_det:    int) -> np.ndarray:
    """
    Convert the raw syndrome array returned by sample_memory_circuit
    into detector events expected by the decoder.

    Two layouts are handled automatically:

      Layout A  (n_shots * n_rounds,  n_anc)
        sample_memory_circuit stacks rounds sequentially.
        → reshape to (n_shots, n_rounds, n_anc)
        → detector[r=0] = raw[0]          (vs ideal |0⟩ reference)
        → detector[r>0] = raw[r] XOR raw[r-1]
        → flatten and trim to n_det columns.

      Layout B  (n_shots,  any_cols)
        Rounds already folded by the library.
        → slice to first n_det columns and return.

    Returns
    -------
    float64 ndarray of shape (n_shots, n_det)
    """
    if syns_raw.ndim == 1:
        syns_raw = syns_raw.reshape(1, -1)

    total_rows, n_cols = syns_raw.shape

    if total_rows == n_shots * n_rounds:          # Layout A
        n_anc = n_cols
        s = syns_raw.reshape(n_shots, n_rounds, n_anc)
        dets = np.zeros((n_shots, n_rounds * n_anc), dtype=np.uint8)
        dets[:, :n_anc] = s[:, 0, :]             # round 0 vs |0⟩
        for r in range(1, n_rounds):
            dets[:, r * n_anc:(r + 1) * n_anc] = s[:, r, :] ^ s[:, r - 1, :]
        return dets[:, :n_det].astype(np.float64)

    else:                                          # Layout B
        return syns_raw[:, :n_det].astype(np.float64)


def count_logical_errors(decoder,
                         det_events: np.ndarray,
                         datas:      np.ndarray,
                         dem,
                         n_shots:    int) -> int:
    """
    Decode every shot and count logical errors.

    Logical error check (GF2):
      correction  = threshold(result.result, 0.5)       binary error vector
      pred_flip   = (obs_matrix @ correction) mod 2     predicted observable flip
      actual      = data[0] mod 2                       measured logical
      error if pred_flip[0] != actual

    If the DEM object lacks observable_flips_matrix (older build), fall back
    to parity of the correction vector (valid for repetition / Steane codes
    with a single logical qubit measured in Z).
    """
    # Fetch observable flips matrix once
    try:
        obs_matrix = np.array(dem.observable_flips_matrix, dtype=np.uint8)
        use_obs    = True
    except AttributeError:
        obs_matrix = None
        use_obs    = False

    errors = 0
    for s in range(n_shots):
        res = decoder.decode(det_events[s].tolist())
        if not res.converged:
            continue

        # Hard threshold → binary correction vector
        correction = (np.array(res.result) > 0.5).astype(np.uint8)

        if use_obs:
            pred_flip = int((obs_matrix @ correction).sum() % 2)
        else:
            pred_flip = int(correction.sum() % 2)

        actual = int(datas[s, 0]) % 2
        if pred_flip != actual:
            errors += 1

    return errors


def run_memory_experiment(code, dem, decoder,
                          n_shots:  int,
                          n_rounds: int,
                          noise):
    """
    Full pipeline for one (code, decoder, noise) configuration:
      sample → convert → decode → count logical errors.

    Returns
    -------
    n_logical_errors : int
    elapsed_s        : float   (decode time only)
    syns_shape       : tuple
    """
    syns, datas = qec.sample_memory_circuit(
        code,
        op       = qec.operation.prep0,
        numRounds= n_rounds,
        numShots = n_shots,
        noise    = noise,
    )
    syns  = np.array(syns,  dtype=np.uint8)
    datas = np.array(datas, dtype=np.uint8)

    n_det      = dem.detector_error_matrix.shape[0]
    det_events = to_detector_events(syns, n_shots, n_rounds, n_det)

    t0     = time.perf_counter()
    n_errs = count_logical_errors(decoder, det_events, datas, dem, n_shots)
    elapsed = time.perf_counter() - t0

    return n_errs, elapsed, syns.shape


def make_qldpc_decoder(dem, max_iter=50, osd=True):
    """Convenience wrapper for nv-qldpc-decoder with BP-OSD."""
    return qec.get_decoder(
        "nv-qldpc-decoder",
        dem.detector_error_matrix,
        error_rate_vec = np.array(dem.error_rates),
        max_iterations = max_iter,
        bp_method      = 0,      # sum-product BP
        use_osd        = osd,
    )


def make_noise(p: float) -> cudaq.NoiseModel:
    """Depolarising noise on all two-qubit X gates at rate p."""
    nm = cudaq.NoiseModel()
    nm.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    return nm


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Surface Code d=3  +  nv-qldpc-decoder  (BP-OSD, GPU)
# ══════════════════════════════════════════════════════════════════════════════
log()
log("[STEP 1]  Surface Code d=3  +  nv-qldpc-decoder  (BP-OSD, GPU)")
log("-" * 60)

SHOTS   = 2000
P_PHYS  = 0.005

# Variables that STEP 2 reuses — defined here so the scope is visible even if
# STEP 1 raises an exception (they stay None and STEP 2 skips gracefully).
code3 = noise3 = dem_s1 = dec_s1 = None
p_L3  = None

try:
    D3, ROUNDS_D3 = 3, 5
    code3  = qec.get_code("surface_code", distance=D3)
    noise3 = make_noise(P_PHYS)
    dem_s1 = dem_fn(code3, qec.operation.prep0, ROUNDS_D3, noise3)

    n_det3  = dem_s1.detector_error_matrix.shape[0]
    n_mech3 = dem_s1.detector_error_matrix.shape[1]
    log(f"   DEM shape   : ({n_det3}, {n_mech3})")

    dec_s1 = make_qldpc_decoder(dem_s1)

    n_err3, t3, shape3 = run_memory_experiment(
        code3, dem_s1, dec_s1, SHOTS, ROUNDS_D3, noise3)
    p_L3 = n_err3 / SHOTS

    log(f"   syns shape  : {shape3}")
    log(f"   p_logical   : {p_L3:.4f}  ({n_err3}/{SHOTS})")
    log(f"   Throughput  : {SHOTS / t3:,.0f} shots/s  ({t3 * 1e3:.1f} ms)")

except Exception:
    log("   ERROR:"); traceback.print_exc(file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Surface Code d=3  +  nv-relaybp-decoder  (new in 0.5.0)
# ══════════════════════════════════════════════════════════════════════════════
log()
log("[STEP 2]  Surface Code d=3  +  nv-relaybp-decoder  (0.5.0)")
log("-" * 60)

try:
    if dem_s1 is None:
        raise RuntimeError("STEP 1 failed — no DEM available, skipping STEP 2")

    dec_rbp = qec.get_decoder(
        "nv-relaybp-decoder",
        dem_s1.detector_error_matrix,
        error_rate_vec = np.array(dem_s1.error_rates),
        max_iterations = 50,
    )
    n_err_rbp, t_rbp, _ = run_memory_experiment(
        code3, dem_s1, dec_rbp, SHOTS, ROUNDS_D3, noise3)

    log(f"   p_logical   : {n_err_rbp / SHOTS:.4f}  ({n_err_rbp}/{SHOTS})")
    log(f"   Throughput  : {SHOTS / t_rbp:,.0f} shots/s  ({t_rbp * 1e3:.1f} ms)")

except Exception as exc:
    msg = str(exc).lower()
    if any(kw in msg for kw in ("relaybp", "unknown decoder", "not found", "invalid")):
        log("   nv-relaybp-decoder not available in this build — skipping.")
        log("   (Requires cudaq-qec built with Sequential Relay BP support.)")
    else:
        log("   ERROR:"); traceback.print_exc(file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Surface Code d=5  (distance scaling check)
# ══════════════════════════════════════════════════════════════════════════════
log()
log("[STEP 3]  Surface Code d=5  +  nv-qldpc-decoder  (distance scaling)")
log("-" * 60)

p_L5 = None
try:
    D5, ROUNDS_D5 = 5, 5
    code5  = qec.get_code("surface_code", distance=D5)
    noise5 = make_noise(P_PHYS)
    dem5   = dem_fn(code5, qec.operation.prep0, ROUNDS_D5, noise5)
    dec5   = make_qldpc_decoder(dem5)

    n_err5, t5, shape5 = run_memory_experiment(
        code5, dem5, dec5, SHOTS, ROUNDS_D5, noise5)
    p_L5 = n_err5 / SHOTS

    log(f"   DEM shape   : {dem5.detector_error_matrix.shape}  "
        f"syns: {shape5}")
    log(f"   p_logical   : {p_L5:.4f}  ({n_err5}/{SHOTS})")
    log(f"   Throughput  : {SHOTS / t5:,.0f} shots/s")

    if p_L3 is not None and p_L5 is not None:
        denom = max(p_L5, 0.5 / SHOTS)          # avoid divide-by-zero
        log(f"   d=3 → d=5 suppression : ×{p_L3 / denom:.2f}")

except Exception:
    log("   ERROR:"); traceback.print_exc(file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Custom Repetition Code d=5  (0.5.0 instance-method API)
# ══════════════════════════════════════════════════════════════════════════════
log()
log("[STEP 4]  Custom Repetition Code d=5  (@qec.code, 0.5.0 API)")
log("-" * 60)

try:
    DR = 5   # repetition code distance

    # ── Quantum kernels ──────────────────────────────────────────────────────

    @cudaq.kernel
    def rep_prep0(lq: patch):
        """Prepare logical |0⟩:  reset all data and ancilla qubits."""
        reset(lq.data)
        reset(lq.ancz)

    @cudaq.kernel
    def rep_x_logical(lq: patch):
        """Apply logical X: flip all data qubits."""
        for i in range(len(lq.data)):
            x(lq.data[i])

    @cudaq.kernel
    def rep_prep1(lq: patch):
        """Prepare logical |1⟩ = prep0 then logical X."""
        reset(lq.data)
        reset(lq.ancz)
        for i in range(len(lq.data)):
            x(lq.data[i])

    @cudaq.kernel
    def rep_stabilizer_round(lq: patch) -> list[bool]:
        """
        One round of ZZ stabiliser measurements.
        Stabilisers:  ZZ_01, ZZ_12, ..., ZZ_{d-2,d-1}
        Each ancilla[i] measures data[i] XOR data[i+1].
        """
        for i in range(len(lq.ancz)):
            reset(lq.ancz[i])
            x.ctrl(lq.data[i],     lq.ancz[i])
            x.ctrl(lq.data[i + 1], lq.ancz[i])
        results = mz(lq.ancz)
        reset(lq.ancz)
        return results

    @cudaq.kernel
    def rep_z_logical(lq: patch):
        """Logical Z measurement: measure first data qubit in Z basis."""
        mz(lq.data[0])

    # ── Spin-word helper (handles API name variation) ─────────────────────────
    def spin_word(s: str):
        try:
            return cudaq.SpinOperator.from_word(s)
        except AttributeError:
            return cudaq.spin.from_word(s)

    # ── Code class — 0.5.0 style: instance methods + __init__ ────────────────
    @qec.code("NQM-RepetitionCode-d5")
    class RepetitionCodeD5:
        """
        [[5, 1, 5]] repetition code protecting against bit-flip (X) errors.
        Stabilisers:  ZZI II,  IZZ II,  IIZ ZI,  III ZZ
        Logical obs:  ZZZZZ
        """

        def __init__(self, **kwargs):
            qec.Code.__init__(self, **kwargs)

            # ZZ stabilisers for neighbouring pairs
            self.stabilizers = [
                spin_word("I" * i + "ZZ" + "I" * (DR - i - 2))
                for i in range(DR - 1)
            ]

            # Logical Z observable
            self.pauli_observables = [spin_word("Z" * DR)]

            # Operation kernel registry
            self.operation_encodings = {
                qec.operation.prep0:             rep_prep0,
                qec.operation.prep1:             rep_prep1,
                qec.operation.x:                 rep_x_logical,
                qec.operation.stabilizer_round:  rep_stabilizer_round,
                qec.operation.z:                 rep_z_logical,
            }

        def get_num_data_qubits(self):      return DR
        def get_num_ancilla_x_qubits(self): return 0
        def get_num_ancilla_z_qubits(self): return DR - 1
        def get_num_ancilla_qubits(self):   return DR - 1
        def get_num_x_stabilizers(self):    return 0
        def get_num_z_stabilizers(self):    return DR - 1

    # ── Run experiment ────────────────────────────────────────────────────────
    rep_code  = qec.get_code("NQM-RepetitionCode-d5")
    rep_noise = make_noise(0.01)

    rep_dem = dem_fn(rep_code, qec.operation.prep0, 6, rep_noise)
    rep_dec = qec.get_decoder(
        "nv-qldpc-decoder",
        rep_dem.detector_error_matrix,
        error_rate_vec = np.array(rep_dem.error_rates),
        max_iterations = 30,
        use_osd        = True,
    )

    r_err, r_t, r_shape = run_memory_experiment(
        rep_code, rep_dem, rep_dec, 1000, 6, rep_noise)

    log(f"   syns shape  : {r_shape}")
    log(f"   d={DR} RepCode  p_phys=0.01  →  p_logical={r_err / 1000:.4f}")
    log(f"   Throughput  : {1000 / r_t:,.0f} shots/s")

except Exception:
    log("   ERROR:"); traceback.print_exc(file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Threshold scan  d ∈ {3, 5, 7}
# ══════════════════════════════════════════════════════════════════════════════
log()
log("[STEP 5]  Threshold scan  d ∈ {3, 5, 7}")
log("-" * 60)

scan_results = {}

try:
    distances   = [3, 5, 7]
    phys_errors = [0.001, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020]
    SC          = 500   # shots per scan point

    for d in distances:
        code_d = qec.get_code("surface_code", distance=d)
        scan_results[d] = []

        for p in phys_errors:
            try:
                noise_d = make_noise(p)
                dem_d   = dem_fn(code_d, qec.operation.prep0, d, noise_d)
                dec_d   = make_qldpc_decoder(dem_d, max_iter=30)

                n_e, _, _ = run_memory_experiment(
                    code_d, dem_d, dec_d, SC, d, noise_d)
                p_L = n_e / SC
                scan_results[d].append(p_L)
                log(f"   d={d}  p={p:.3f}  →  p_L={p_L:.4f}")

            except Exception as exc_inner:
                log(f"   d={d}  p={p:.3f}  →  SKIP ({exc_inner})")
                scan_results[d].append(float("nan"))

    # ── Plot ─────────────────────────────────────────────────────────────────
    if HAS_MPL and scan_results:
        colors = {3: "#1f77b4", 5: "#ff7f0e", 7: "#2ca02c"}
        fig, ax = plt.subplots(figsize=(8, 5))

        for d in distances:
            if d not in scan_results:
                continue
            ys = [max(v, 0.5 / SC) if not np.isnan(v) else np.nan
                  for v in scan_results[d]]
            ax.semilogy(phys_errors, ys,
                        "o-", color=colors.get(d, "black"),
                        label=f"d={d}", lw=2, ms=7)

        ax.axvline(0.01, color="gray", ls="--", alpha=0.7,
                   label="~threshold ≈ 1 %")
        ax.set_xlabel("Physical error rate  $p$",  fontsize=12)
        ax.set_ylabel("Logical error rate  $p_L$", fontsize=12)
        ax.set_title(
            "Surface Code Threshold — cudaq-qec 0.5.0\n"
            "nv-qldpc-decoder (BP-OSD, GPU)  |  C-DAC NQM  |  RTX A4000",
            fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        out_png = "cudaqx_qec_v6_threshold.png"
        plt.savefig(out_png, dpi=150)
        log(f"   Plot saved → {out_png}")

except Exception:
    log("   ERROR:"); traceback.print_exc(file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
log()
log("=" * 65)
log("  ALL STEPS COMPLETE  —  CUDA-QX QEC v6")
log("=" * 65)