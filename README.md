# CUDA-Q QEC & QLDPC Experiments

GPU-accelerated quantum computing experiments using [CUDA-Q](https://developer.nvidia.com/cuda-quantum) (NVIDIA), [Qiskit](https://qiskit.org/), and [CuPy](https://cupy.dev/), developed at **C-DAC Noida** for quantum error correction (QEC) and quantum LDPC (QLDPC) research.

> **Author:** Nasir Ali  
> **Affiliation:** C-DAC Noida / MSc Physics, Jawaharlal Nehru University  
> **Hardware target:** NVIDIA GPU (CUDA-Q `nvidia` target; CuPy fallback to NumPy on CPU)

---

## Overview

This repository spans three interconnected research themes:

| Theme | Scripts | Description |
|-------|---------|-------------|
| **CUDA-Q Quantum Circuits** | `Bell_State.py`, `cudaq01.py`, `deutsch_algorithm.py` | Native CUDA-Q kernels on GPU |
| **GPU Physics Simulations** | `heat_equation.py`, `heat.py`, `imaginary_time_evolution_gpu.py`, `gpu_at.py` | CuPy / PyTorch GPU-accelerated PDE + VarQITE |
| **Quantum Visualisation** | `atom.py`, `atom_gif.py`, `gr_neu.py`, `conc1.py` | Bloch sphere, EM fields, entanglement metrics, atomic dynamics |

---

## Repository Structure

```
cudaq/
├── Bell_State.py                    # GHZ/Bell-state preparation — CUDA-Q kernel, GPU sampling
├── cudaq01.py                       # Scalable GHZ circuit with CLI qubit-count argument
├── deutsch_algorithm.py             # Deutsch algorithm (constant vs balanced oracle) on GPU
│
├── heat_equation.py                 # 2-D heat diffusion PDE — CuPy GPU solver + animation
├── heat.py                          # 2-D heat diffusion baseline (CPU/GPU auto-detect)
├── imaginary_time_evolution_gpu.py  # VarQITE on GPU via Qiskit + CuPy (ImaginaryMcLachlan)
├── gpu_at.py                        # Atomic oscillation + FFT spectrum — PyTorch + Qiskit Aer
│
├── atom.py                          # Animated atomic oscillation (matplotlib)
├── atom_gif.py                      # GIF export of atomic dynamics
├── gr_neu.py                        # Entanglement of formation & mutual information (Qiskit)
├── conc1.py                         # Concurrence / entanglement metric exploration
│
├── cudaq.ipynb                      # Jupyter notebook: CUDA-Q interactive experiments
├── hello.lean                       # Lean 4 formal verification proof sketch
│
├── energy_evolution_GPU.csv         # Numerical output: imaginary-time energy vs iteration
├── energy_evolution_GPU.png         # Corresponding energy convergence plot
├── additional_analysis_GPU.png      # Additional GPU simulation analysis figure
├── 2D_heat_diffusion_GPU.gif        # 2-D heat diffusion GPU animation (root copy)
│
└── anims/                           # All generated animations and figures
    ├── 2D_diffusion_*.gif           # Heat diffusion at 8×8 → 128×128 grid resolutions
    ├── atomic_oscillation*.gif      # Atomic oscillation dynamics
    ├── em_field_animation.gif       # Electromagnetic field evolution
    ├── e8_petrie_*.png              # E8 Lie algebra Petrie projection (adjacency + Coxeter)
    ├── wilson_rg_analysis.png       # Wilson renormalisation group analysis
    ├── spectrum_plot.png            # FFT frequency spectrum
    └── energy_evolution_GPU.png     # GPU energy convergence (anims copy)
```

---

## Key Experiments

### Bell / GHZ State Preparation — `Bell_State.py`

```python
@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector[0])
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    mz(qvector)
```

Runs on NVIDIA GPU via `cudaq.set_target("nvidia")`. Prints circuit drawing and samples the multi-qubit GHZ state.

### Deutsch Algorithm — `deutsch_algorithm.py`

Demonstrates quantum advantage: determines in a single query whether a black-box function `f: {0,1} → {0,1}` is constant or balanced. Targets NVIDIA GPU when available, falls back to CPU simulator.

### 2-D Heat Diffusion (GPU) — `heat_equation.py`

Solves the 2-D heat equation on an N×N grid using CuPy for GPU-accelerated finite-difference time stepping. Generates animated GIFs at multiple grid resolutions (8×8 → 128×128).

### Variational Imaginary Time Evolution — `imaginary_time_evolution_gpu.py`

Implements VarQITE (Variational Quantum Imaginary Time Evolution) using:
- `EfficientSU2` ansatz (Qiskit)
- `ImaginaryMcLachlanPrinciple` + `ReverseQGT`
- CuPy GPU backend via Qiskit Aer

Tracks energy convergence to the ground state and exports results to `energy_evolution_GPU.csv`.

### Atomic Oscillation & FFT — `gpu_at.py`

Simulates a driven quantum harmonic oscillator on GPU (PyTorch CUDA), computes the FFT frequency spectrum, and overlays a single-qubit Qiskit Aer circuit for gate-level comparison.

---

## Setup

### Dependencies

```bash
pip install cuda-quantum cupy-cuda12x torch qiskit qiskit-aer qiskit-algorithms matplotlib numpy pandas
```

Or use the Nix environment if available (see `flake.nix` in the companion `Artiq-Sinara_Replica` repo for a reference template).

### GPU requirement

- CUDA-Q experiments require an NVIDIA GPU with CUDA 11+.
- CuPy experiments fall back to NumPy automatically when no GPU is detected.
- PyTorch experiments use `torch.device('cuda')` with CPU fallback.

### Run an experiment

```bash
# Bell state (2-qubit GHZ)
python Bell_State.py

# Scalable GHZ with n qubits
python cudaq01.py 5

# Deutsch algorithm
python deutsch_algorithm.py

# 2-D heat diffusion GPU animation
python heat_equation.py

# Imaginary time evolution
python imaginary_time_evolution_gpu.py
```

---

## Output Gallery

| File | Description |
|------|-------------|
| `anims/2D_diffusion_128x128.gif` | Heat diffusion on 128×128 grid |
| `anims/atomic_oscillations.gif` | Driven quantum harmonic oscillator |
| `anims/em_field_animation.gif` | Electromagnetic field time evolution |
| `anims/e8_petrie_adjacency_edges.png` | E8 root system — adjacency projection |
| `anims/wilson_rg_analysis.png` | Wilson renormalisation group flow |
| `energy_evolution_GPU.png` | VarQITE energy convergence on GPU |

---

## Relation to QEC / QLDPC Research

This codebase serves as the GPU simulation and circuit-primitive layer for quantum error correction experiments targeting LDPC codes:

- **Bell / GHZ circuits** provide the entangled ancilla resource states used in syndrome extraction for surface and LDPC codes.
- **VarQITE** is used to find low-energy code states and study logical qubit fidelity under noise.
- **Heat diffusion simulations** benchmark GPU throughput and memory bandwidth prior to scaling the syndrome decoder to large parity-check matrices.

---

## License

Research code developed at C-DAC Noida. CUDA-Q is © NVIDIA Corporation (Apache-2.0). Qiskit is © IBM (Apache-2.0).
