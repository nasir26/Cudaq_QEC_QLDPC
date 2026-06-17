#!/usr/bin/env python3
# python -u cudaqx_qec_final.py 2>&1 | tee final.log
"""
=============================================================================
  CUDA-QX QEC Final  —  cudaq 0.13.0 + cudaq-qec 0.5.0
  NVIDIA RTX A4000  |  C-DAC Qniverse / NQM
  ─────────────────────────────────────────────────────────────────────────
  All API fixes confirmed from qec_minimal.py smoke-test:
    ① All args to dem_from_memory_circuit / sample_memory_circuit → POSITIONAL
    ② get_decoder args → POSITIONAL
    ③ Syndrome vector must be SLICED to n_detectors before decode()
       (sample_memory_circuit appends logical observable bits at the end)
  ─────────────────────────────────────────────────────────────────────────
  STEP 1 — Surface Code d=3 + nv-qldpc-decoder  (BP-OSD, GPU)
  STEP 2 — Surface Code d=3 + nv-relaybp-decoder (RelayBP, GPU, new 0.5.0)
  STEP 3 — Steane Code  [7,1,3] + nv-qldpc-decoder
  STEP 4 — Custom Repetition Code d=5 via @qec.code
  STEP 5 — Threshold scan d=3 vs d=5 → PNG
=============================================================================
"""

import os, sys, traceback, time
import numpy as np

def w(msg=""):
    os.write(2, (str(msg) + "\n").encode())

def decode_shots(decoder, syns, datas, n_det, n_shots):
    """
    Decode all shots and return logical error count + throughput.

    ★ KEY FIX: syns from sample_memory_circuit has shape
      (n_shots, n_detectors + n_logicals).
      The decoder only accepts a vector of length n_detectors.
      Slice syns[s, :n_det] before every decode() call.
    """
    t0 = time.perf_counter()
    errs = 0
    for s in range(n_shots):
        syndrome_vec = syns[s, :n_det].astype(np.float64)   # ← critical slice
        res = decoder.decode(syndrome_vec)
        predicted = int(np.sum(res.result) % 2)
        actual    = int(datas[s, 0])
        if predicted != actual:
            errs += 1
    elapsed = time.perf_counter() - t0
    return errs, elapsed

# ── imports ───────────────────────────────────────────────────────────────────
w("── importing cudaq …")
import cudaq
w(f"   cudaq      : {cudaq.__version__}")

w("── importing cudaq_qec …")
import cudaq_qec as qec
from cudaq_qec import patch
w(f"   cudaq_qec  : {qec.__version__}")

w("── matplotlib …")
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True; w("   OK")
except ImportError:
    HAS_MPL = False
    w("   not found — run: pip install matplotlib")

w()
w("=" * 65)
w("  CUDA-QX QEC Final  |  cudaq 0.13.0 + cudaq-qec 0.5.0")
w("  C-DAC Qniverse  |  National Quantum Mission  |  RTX A4000")
w("=" * 65)

cudaq.set_target("stim")
w("   target : stim\n")

dem_fn = getattr(qec, "dem_from_memory_circuit",
                 getattr(qec, "z_dem_from_memory_circuit", None))
w(f"   DEM fn : {dem_fn.__name__}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Surface Code  d=3  +  nv-qldpc-decoder  (BP-OSD)
# ══════════════════════════════════════════════════════════════════════════════
w("[STEP 1]  Surface Code  d=3  +  nv-qldpc-decoder  (BP-OSD, GPU)")
w("-" * 60)
try:
    D, ROUNDS, SHOTS, P = 3, 5, 2000, 0.005
    code  = qec.get_code("surface_code", distance=D)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(P), 1)

    dem    = dem_fn(code, qec.operation.prep0, ROUNDS, noise)
    n_det  = dem.detector_error_matrix.shape[0]   # ← store detector count
    n_mech = dem.detector_error_matrix.shape[1]
    w(f"   DEM : {n_det} detectors × {n_mech} error mechanisms")

    dec = qec.get_decoder("nv-qldpc-decoder",
                          dem.detector_error_matrix,
                          np.array(dem.error_rates), 50)

    syns, datas = qec.sample_memory_circuit(
        code, qec.operation.prep0, ROUNDS, SHOTS, noise)
    w(f"   syns raw shape : {syns.shape}  "
      f"(last {syns.shape[1]-n_det} col(s) = logical obs)")

    errs, t = decode_shots(dec, syns, datas, n_det, SHOTS)
    w(f"   p_logical = {errs/SHOTS:.4f}  ({errs}/{SHOTS})")
    w(f"   decode    : {t*1e3:.1f} ms  →  {SHOTS/t:.0f} shots/s")

    # keep for STEP 2
    dem_s1, syns_s1, datas_s1, ndet_s1 = dem, syns, datas, n_det

except Exception:
    w("   ERROR:"); traceback.print_exc(file=sys.stderr)
    dem_s1 = None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Surface Code  d=3  +  nv-relaybp-decoder  (new in 0.5.0)
# ══════════════════════════════════════════════════════════════════════════════
w()
w("[STEP 2]  Surface Code  d=3  +  nv-relaybp-decoder  (GPU, new 0.5.0)")
w("-" * 60)
try:
    if dem_s1 is None:
        raise RuntimeError("skipped — STEP 1 failed")

    dec_rbp = qec.get_decoder("nv-relaybp-decoder",
                               dem_s1.detector_error_matrix,
                               np.array(dem_s1.error_rates), 50)

    errs_rbp, t_rbp = decode_shots(
        dec_rbp, syns_s1, datas_s1, ndet_s1, SHOTS)
    w(f"   p_logical = {errs_rbp/SHOTS:.4f}  ({errs_rbp}/{SHOTS})")
    w(f"   decode    : {t_rbp*1e3:.1f} ms  →  {SHOTS/t_rbp:.0f} shots/s")

    # comparison
    w(f"   BP-OSD vs RelayBP  Δp_L = "
      f"{abs(errs/SHOTS - errs_rbp/SHOTS):.4f}")

except Exception:
    w("   ERROR:"); traceback.print_exc(file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Steane Code  [[7,1,3]]
# ══════════════════════════════════════════════════════════════════════════════
w()
w("[STEP 3]  Steane Code  [[7,1,3]]  +  nv-qldpc-decoder")
w("-" * 60)
try:
    steane = qec.get_code("steane_code")
    s_noise = cudaq.NoiseModel()
    s_noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.005), 1)

    s_dem   = dem_fn(steane, qec.operation.prep0, 3, s_noise)
    s_ndet  = s_dem.detector_error_matrix.shape[0]
    w(f"   DEM : {s_ndet} detectors × "
      f"{s_dem.detector_error_matrix.shape[1]} mechanisms")

    s_dec   = qec.get_decoder("nv-qldpc-decoder",
                               s_dem.detector_error_matrix,
                               np.array(s_dem.error_rates), 30)

    s_syns, s_data = qec.sample_memory_circuit(
        steane, qec.operation.prep0, 3, 1000, s_noise)

    s_errs, s_t = decode_shots(s_dec, s_syns, s_data, s_ndet, 1000)
    w(f"   p_logical = {s_errs/1000:.4f}  ({s_errs}/1000)")
    w(f"   decode    : {s_t*1e3:.1f} ms")

except Exception:
    w("   ERROR:"); traceback.print_exc(file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Custom Repetition Code  d=5  (@qec.code)
# ══════════════════════════════════════════════════════════════════════════════
w()
w("[STEP 4]  Custom Repetition Code  d=5  (@qec.code)")
w("-" * 60)
try:
    DR = 5

    @cudaq.kernel
    def rep_prep0(lq: patch):
        reset(lq.data); reset(lq.ancz)

    @cudaq.kernel
    def rep_x_logical(lq: patch):
        for i in range(len(lq.data)): x(lq.data[i])

    @cudaq.kernel
    def rep_prep1(lq: patch):
        rep_prep0(lq); rep_x_logical(lq)

    @cudaq.kernel
    def rep_stabilizer_round(lq: patch):
        for i in range(len(lq.ancz)):
            reset(lq.ancz[i])
            x.ctrl(lq.data[i],   lq.ancz[i])
            x.ctrl(lq.data[i+1], lq.ancz[i])
            mz(lq.ancz[i])

    @cudaq.kernel
    def rep_z_logical(lq: patch):
        mz(lq.data[0])

    def spin_word(s):
        try:    return cudaq.spin.from_word(s)
        except: return cudaq.SpinOperator.from_word(s)

    @qec.code
    class RepetitionCode:
        num_data_qubits = DR
        num_ancz_qubits = DR - 1
        num_ancx_qubits = 0
        operation_encodings = {
            qec.operation.x               : rep_x_logical,
            qec.operation.prep0           : rep_prep0,
            qec.operation.prep1           : rep_prep1,
            qec.operation.stabilizer_round: rep_stabilizer_round,
            qec.operation.z               : rep_z_logical,
        }
        stabilizers = [spin_word("I"*i + "ZZ" + "I"*(DR-i-2))
                       for i in range(DR-1)]
        observables = [spin_word("Z"*DR)]

    rep_code  = qec.get_code("RepetitionCode")
    rep_noise = cudaq.NoiseModel()
    rep_noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)

    r_dem  = dem_fn(rep_code, qec.operation.prep0, 6, rep_noise)
    r_ndet = r_dem.detector_error_matrix.shape[0]
    r_dec  = qec.get_decoder("nv-qldpc-decoder",
                              r_dem.detector_error_matrix,
                              np.array(r_dem.error_rates), 30)
    r_syns, r_data = qec.sample_memory_circuit(
        rep_code, qec.operation.prep0, 6, 1000, rep_noise)

    r_errs, _ = decode_shots(r_dec, r_syns, r_data, r_ndet, 1000)
    w(f"   RepCode d={DR}  p_phys=0.01  →  p_logical={r_errs/1000:.4f}")

except Exception:
    w("   ERROR:"); traceback.print_exc(file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Threshold scan  d=3 vs d=5
# ══════════════════════════════════════════════════════════════════════════════
w()
w("[STEP 5]  Threshold scan  d=3 vs d=5  (surface code)")
w("-" * 60)
scan = {}
try:
    distances   = [3, 5]
    phys_errors = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
    SC          = 500

    for d in distances:
        cd = qec.get_code("surface_code", distance=d)
        scan[d] = []
        for p in phys_errors:
            np_ = cudaq.NoiseModel()
            np_.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
            dp   = dem_fn(cd, qec.operation.prep0, d, np_)
            ndet = dp.detector_error_matrix.shape[0]
            decp = qec.get_decoder("nv-qldpc-decoder",
                                   dp.detector_error_matrix,
                                   np.array(dp.error_rates), 30)
            sp, dp2 = qec.sample_memory_circuit(
                cd, qec.operation.prep0, d, SC, np_)
            e, _ = decode_shots(decp, sp, dp2, ndet, SC)
            scan[d].append(e / SC)
            w(f"   d={d}  p={p:.3f}  →  p_L={e/SC:.4f}")

    if HAS_MPL and scan:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for d, col in zip([3,5], ["#1f77b4","#ff7f0e"]):
            ax.semilogy(phys_errors, scan[d], "o-", color=col,
                        label=f"d={d} surface code", lw=2, ms=6)
        ax.axvline(0.01, color="gray", ls="--", alpha=0.7, label="~threshold")
        ax.set_xlabel("Physical error rate  p", fontsize=12)
        ax.set_ylabel("Logical error rate  p_L", fontsize=12)
        ax.set_title(
            "Surface Code Threshold\n"
            "cudaq-qec 0.5.0  |  nv-qldpc-decoder GPU  |  RTX A4000",
            fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("cudaqx_qec_threshold.png", dpi=150)
        w("   Plot saved → cudaqx_qec_threshold.png")

except Exception:
    w("   ERROR:"); traceback.print_exc(file=sys.stderr)

w()
w("=" * 65)
w("  ALL STEPS COMPLETE")
w("=" * 65)