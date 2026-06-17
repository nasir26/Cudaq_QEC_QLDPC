import os
import time
import cupy as cp  # Replace NumPy with CuPy for GPU support
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from PIL import Image
from collections import Counter
from IPython.display import HTML
from concurrent.futures import ThreadPoolExecutor

# Qiskit imports
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit_algorithms import TimeEvolutionProblem, VarQITE
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, ReverseQGT
from qiskit_algorithms.gradients import DerivativeType
from qiskit.primitives import Estimator as EstimatorP
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler, Batch
from qiskit.transpiler import preset_passmanagers

# Set environment variables for parallel processing
n_vcpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_vcpus)
os.environ["MKL_NUM_THREADS"] = str(n_vcpus)
print("Number of vCPUs detected:", n_vcpus)

# Check if GPU is available
gpu_available = cp.cuda.is_available()
print("GPU available:", gpu_available)

# ---------------------------
# Step 1. Build the Hamiltonian (on GPU)
# ---------------------------
def build_2d_dirichlet_laplacian_hamiltonian(nx: int, ny: int, alpha: float = 1.0) -> SparsePauliOp:
    n = nx * ny
    L2d = cp.zeros((n, n), dtype=float)  # Use CuPy for GPU array
    for y in range(ny):
        for x in range(nx):
            i = x + nx * y
            L2d[i, i] = -4.0
            if x > 0:
                L2d[i, i-1] = 1.0
            if x < nx - 1:
                L2d[i, i+1] = 1.0
            if y > 0:
                L2d[i, i - nx] = 1.0
            if y < ny - 1:
                L2d[i, i + nx] = 1.0
    H_mat = -alpha * L2d
    # Convert CuPy array to NumPy for Operator (Qiskit doesn't natively support CuPy yet)
    op = Operator(cp.asnumpy(H_mat))
    return SparsePauliOp.from_operator(op)

nx, ny = 16, 16
hamiltonian = build_2d_dirichlet_laplacian_hamiltonian(nx, ny)
print("Number of qubits:", hamiltonian.num_qubits)
print("Number of Pauli terms:", len(hamiltonian))

# ---------------------------
# Step 2. Define the ansatz
# ---------------------------
ansatz = EfficientSU2(hamiltonian.num_qubits, entanglement="linear", reps=1)

# ---------------------------
# Step 3. Transpile the circuit (GPU backend)
# ---------------------------
backend = AerSimulator(method='statevector', device='GPU' if gpu_available else 'CPU',
                       max_parallel_threads=n_vcpus, max_parallel_experiments=n_vcpus)
pm = preset_passmanagers.generate_preset_pass_manager(optimization_level=3, backend=backend)
circuit = pm.run(ansatz)
observable = hamiltonian.apply_layout(circuit.layout)
circuit_params = list(circuit.parameters)
n_params = len(circuit_params)
print(f"Number of circuit parameters: {n_params}")

# Initialize parameters on GPU
init_param_values = cp.zeros(n_params)
random_indices = cp.random.choice(n_params, size=10, replace=True)
init_param_values[random_indices] = cp.pi / 4

# ---------------------------
# Step 4. Instantiate Estimators (GPU backend)
# ---------------------------
estimator = Estimator(mode=backend)
estimator.options.default_shots = 1000
estimatorp = EstimatorP()

# -----------------------------------------------------------------------------
# Step 5. Define functions (use GPU where possible)
# -----------------------------------------------------------------------------
def energy(params):
    params_np = cp.asnumpy(params)  # Convert to NumPy for Qiskit
    pub = (circuit, observable, [params_np])
    job = estimator.run([pub])
    return job.result()[0].data.evs[0]

def compute_gradient(params):
    grad_calculator = ParamShiftEstimatorGradient(estimatorp, derivative_type=DerivativeType.IMAG)
    params_np = cp.asnumpy(params)
    result = grad_calculator.run([circuit], [observable], [params_np]).result()
    return cp.array(result.gradients[0])  # Convert back to CuPy

def compute_qgt(params):
    qgt_instance = ReverseQGT(phase_fix=True, derivative_type=DerivativeType.REAL)
    params_np = cp.asnumpy(params)
    result = qgt_instance.run([circuit], [params_np]).result().qgts
    return cp.array(result[0])  # Convert back to CuPy

# -----------------------------------------------------------------------------
# Step 6. VarQITE loop (GPU-accelerated)
# -----------------------------------------------------------------------------
total_time = 35
dt = 0.1
n_steps = int(total_time / dt)
params = init_param_values.copy()

energy_history = []
params_history = [cp.asnumpy(params).copy()]  # Store as NumPy for compatibility

print("Starting variational imaginary time evolution...")
start_time = time.time()

for step in range(n_steps):
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_energy = executor.submit(energy, params)
        future_grad = executor.submit(compute_gradient, params)
        future_qgt = executor.submit(compute_qgt, params)
        current_energy = future_energy.result()
        g = future_grad.result()
        S = future_qgt.result()

    energy_history.append(current_energy)
    print(f"Step {step:3d}: Energy = {current_energy:.6f}")

    # GPU-accelerated linear algebra
    lambda_reg = 1e-3
    S_reg = S + lambda_reg * cp.eye(S.shape[0])
    try:
        delta = cp.linalg.solve(S_reg, -g * dt)
    except cp.linalg.LinAlgError:
        delta = -cp.dot(cp.linalg.pinv(S_reg), g) * dt

    params = params + delta
    params_history.append(cp.asnumpy(params).copy())

end_time = time.time()
print(f"Imaginary time evolution completed in {end_time - start_time:.2f} seconds")
print("Final energy:", energy(params))
print("Final parameters:", cp.asnumpy(params))

# ---------------------------
# Step 7. Final circuit measurement (GPU backend)
# ---------------------------
ansatz.measure_all()
circuit = pm.run(ansatz)
sampler = Sampler(mode=backend)

param_values = cp.array(params_history)
num_parameter_sets = param_values.shape[0]
print(f"Total parameter sets: {num_parameter_sets}")

batch_circuits = [(circuit, cp.asnumpy(param_values[i, :]).tolist()) for i in range(num_parameter_sets)]

with Batch(backend=backend) as batch:
    batch_job = sampler.run(batch_circuits)
result_samp = batch_job.result()
all_counts = [res.data.meas.get_counts() for res in result_samp]

# ---------------------------
# Step 8. Process results (on CPU for simplicity)
# ---------------------------
all_probabilities = []
for counts_dict in all_counts:
    shots = sum(counts_dict.values())
    prob_dict = {key: val / shots for key, val in counts_dict.items()}
    all_probabilities.append(prob_dict)

# ---------------------------
# Step 9. Create heatmap (on CPU due to matplotlib)
# ---------------------------
def bitstring_to_grid_coords(bstr, nx, ny):
    i = int(bstr, 2)
    y = i // nx
    x = i % nx
    return x, y

def probabilities_to_2d(prob_dict, nx, ny):
    dist2d = cp.zeros((ny, nx)).get()  # Convert to NumPy for matplotlib
    for bstr, prob in prob_dict.items():
        x, y = bitstring_to_grid_coords(bstr, nx, ny)
        if y < ny and x < nx:
            dist2d[y, x] = prob
    return dist2d

distributions_2d = [probabilities_to_2d(prob_dict, nx, ny) for prob_dict in all_probabilities]

# ---------------------------
# Step 10. Animation (CPU-based due to matplotlib)
# ---------------------------
mpl.rcParams['animation.embed_limit'] = 30
fig, ax = plt.subplots(figsize=(6, 4))
norm = mpl.colors.LogNorm(vmin=1e-4, vmax=1)
im = ax.imshow(distributions_2d[0] + 1e-6, origin='lower', cmap='cool', norm=norm, aspect='auto', animated=True)
plt.colorbar(im, ax=ax)
plt.title("Heat Flow Animation (32x32 Grid)")
plt.xlabel("x index")
plt.ylabel("y index")

def update(frame):
    im.set_array(distributions_2d[frame] + 1e-6)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(distributions_2d), interval=50, blit=True)
ani.save("2D_diffu1.gif", writer="pillow")
HTML(ani.to_jshtml())

# ---------------------------
# Step 11. Plot energy (CPU-based)
# ---------------------------
times = cp.linspace(0, total_time, len(energy_history)).get()
h_exp_val = cp.array(energy_history).get()
assert len(times) == len(h_exp_val), "Mismatch in times vs. energy measurements!"

plt.figure()
plt.plot(times, h_exp_val, label="VarQITE")
plt.xlabel("Time")
plt.ylabel(r"$\langle H \rangle$ (energy)")
plt.legend(loc="upper right")
plt.show()
