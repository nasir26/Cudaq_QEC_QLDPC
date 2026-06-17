import os
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from PIL import Image
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# CUDA-Q imports
import cudaq

# Set environment variables for parallel processing
n_vcpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_vcpus)
os.environ["MKL_NUM_THREADS"] = str(n_vcpus)
print("Number of vCPUs detected:", n_vcpus)

# Check if GPU is available
gpu_available = cp.cuda.is_available()
print("GPU available:", gpu_available)

# Set cudaq target to use GPU if available
if gpu_available:
    cudaq.set_target("nvidia")
else:
    cudaq.set_target("qpp-cpu")

# ---------------------------
# Step 1. Build the Hamiltonian - Updated for correct CUDA-Q API
# ---------------------------
def build_2d_dirichlet_laplacian_hamiltonian(nx: int, ny: int, alpha: float = 1.0):
    """Build the 2D Laplacian Hamiltonian using the appropriate CUDA-Q format"""
    n = nx * ny
    
    # Initialize an empty Hamiltonian - direct approach
    hamiltonian = cudaq.Operator(n)
    
    # Build the 2D Laplacian using spin operators
    for y in range(ny):
        for x in range(nx):
            i = x + nx * y
            
            # Diagonal term (-4)
            # Create Z operator for site i
            z_term = cudaq.Operator(n)
            z_term.z(i)
            hamiltonian += -4.0 * alpha * z_term
            
            # Off-diagonal terms
            if x > 0:
                # X-X interaction with left neighbor
                xx_term = cudaq.Operator(n)
                xx_term.x(i).x(i-1)
                hamiltonian += alpha * xx_term
                
                # Y-Y interaction with left neighbor
                yy_term = cudaq.Operator(n)
                yy_term.y(i).y(i-1)
                hamiltonian += alpha * yy_term
                
            if x < nx - 1:
                # X-X interaction with right neighbor
                xx_term = cudaq.Operator(n)
                xx_term.x(i).x(i+1)
                hamiltonian += alpha * xx_term
                
                # Y-Y interaction with right neighbor
                yy_term = cudaq.Operator(n)
                yy_term.y(i).y(i+1)
                hamiltonian += alpha * yy_term
                
            if y > 0:
                # X-X interaction with upper neighbor
                xx_term = cudaq.Operator(n)
                xx_term.x(i).x(i-nx)
                hamiltonian += alpha * xx_term
                
                # Y-Y interaction with upper neighbor
                yy_term = cudaq.Operator(n)
                yy_term.y(i).y(i-nx)
                hamiltonian += alpha * yy_term
                
            if y < ny - 1:
                # X-X interaction with lower neighbor
                xx_term = cudaq.Operator(n)
                xx_term.x(i).x(i+nx)
                hamiltonian += alpha * xx_term
                
                # Y-Y interaction with lower neighbor
                yy_term = cudaq.Operator(n)
                yy_term.y(i).y(i+nx)
                hamiltonian += alpha * yy_term
    
    return hamiltonian

# Use a more reasonable size for simulation
nx, ny = 4, 4  # Reduced from 128x128 to make it tractable
num_qubits = nx * ny
hamiltonian = build_2d_dirichlet_laplacian_hamiltonian(nx, ny)
print("Number of qubits:", num_qubits)
print("Number of Hamiltonian terms:", len(hamiltonian.get_terms()))

# ---------------------------
# Step 2. Define the ansatz
# ---------------------------
def create_ansatz(num_qubits, num_params):
    # Define a kernel function that will create our ansatz
    @cudaq.kernel
    def ansatz(params: list[float]):
        # Allocate qubits
        q = cudaq.qvector(num_qubits)
        
        # Initialize with a layer of Hadamard gates
        for i in range(num_qubits):
            h(q[i])
        
        # Add parameterized gates
        param_idx = 0
        # Rotation layer
        for i in range(num_qubits):
            rx(params[param_idx], q[i])
            param_idx += 1
            rz(params[param_idx], q[i])
            param_idx += 1
        
        # Entanglement layer - linear connectivity
        for i in range(num_qubits - 1):
            cx(q[i], q[i + 1])
        
        # Another rotation layer
        for i in range(num_qubits):
            rx(params[param_idx], q[i])
            param_idx += 1
            rz(params[param_idx], q[i])
            param_idx += 1
            
        # Measure all qubits
        mz(q)
    
    return ansatz

# Calculate number of parameters: 4 params per qubit (2 rotations * 2 layers)
n_params = num_qubits * 4
print(f"Number of circuit parameters: {n_params}")

# Initialize parameters on GPU
init_param_values = cp.zeros(n_params)
random_indices = cp.random.choice(n_params, size=10, replace=True)
init_param_values[random_indices] = cp.pi / 4

# Create the ansatz
ansatz = create_ansatz(num_qubits, n_params)

# ---------------------------
# Step 3. Define optimizer - use CUDA-Q's built-in optimizer
# ---------------------------
optimizer = cudaq.optimizers.LBFGS()
optimizer.max_iterations = 1

# -----------------------------------------------------------------------------
# Step 4. Define functions for energy, gradient, and QGT
# -----------------------------------------------------------------------------
def energy(params):
    params_np = cp.asnumpy(params)
    expectation = cudaq.observe(ansatz, hamiltonian, params_np)
    return expectation.expectation()

def compute_gradient(params):
    params_np = cp.asnumpy(params)
    # Use parameter shift rule for gradients
    epsilon = 1e-4
    grad = np.zeros_like(params_np)
    
    for i in range(len(params_np)):
        params_plus = params_np.copy()
        params_plus[i] += epsilon
        energy_plus = cudaq.observe(ansatz, hamiltonian, params_plus).expectation()
        
        params_minus = params_np.copy()
        params_minus[i] -= epsilon
        energy_minus = cudaq.observe(ansatz, hamiltonian, params_minus).expectation()
        
        grad[i] = (energy_plus - energy_minus) / (2 * epsilon)
    
    return cp.array(grad)

def compute_qgt(params):
    params_np = cp.asnumpy(params)
    # Approximate QGT using finite differences on the gradients
    qgt = np.zeros((len(params_np), len(params_np)))
    epsilon = 1e-4
    
    # Compute the QGT approximately by perturbing parameters
    base_grad = compute_gradient(params)
    
    for i in range(len(params_np)):
        params_plus = params_np.copy()
        params_plus[i] += epsilon
        grad_plus = compute_gradient(cp.array(params_plus))
        
        qgt[:, i] = (grad_plus.get() - base_grad.get()) / epsilon
    
    return cp.array(qgt)

# -----------------------------------------------------------------------------
# Step 5. VarQITE loop (GPU-accelerated)
# -----------------------------------------------------------------------------
total_time = 10  # Reduced from 35 to make it faster
dt = 0.1
n_steps = int(total_time / dt)
params = init_param_values.copy()

energy_history = []
params_history = [cp.asnumpy(params).copy()]  # Store as NumPy for compatibility

print("Starting variational imaginary time evolution...")
start_time = time.time()

for step in range(n_steps):
    current_energy = energy(params)
    g = compute_gradient(params)
    S = compute_qgt(params)

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
# Step 6. Run final simulations with optimized parameters
# ---------------------------
all_counts = []
sample_count = 1000  # Number of shots per simulation

for param_set in params_history:
    # Run the ansatz with the evolved parameters
    result = cudaq.sample(ansatz, param_set)
    counts_dict = dict(result.counts())
    all_counts.append(counts_dict)

# ---------------------------
# Step 7. Process results
# ---------------------------
all_probabilities = []
for counts_dict in all_counts:
    shots = sum(counts_dict.values())
    prob_dict = {key: val / shots for key, val in counts_dict.items()}
    all_probabilities.append(prob_dict)

# ---------------------------
# Step 8. Create heatmap
# ---------------------------
def bitstring_to_grid_coords(bstr, nx, ny):
    i = int(bstr, 2)
    y = i // nx
    x = i % nx
    return x, y

def probabilities_to_2d(prob_dict, nx, ny):
    dist2d = np.zeros((ny, nx))
    for bstr, prob in prob_dict.items():
        # Make sure the bitstring is properly formatted
        padded_bstr = bstr.zfill(num_qubits)
        try:
            x, y = bitstring_to_grid_coords(padded_bstr, nx, ny)
            if y < ny and x < nx:
                dist2d[y, x] = prob
        except ValueError:
            # Skip bitstrings that can't be converted
            continue
    return dist2d

distributions_2d = []
for prob_dict in all_probabilities:
    try:
        dist = probabilities_to_2d(prob_dict, nx, ny)
        distributions_2d.append(dist)
    except:
        # If there's an error, add a placeholder
        distributions_2d.append(np.zeros((ny, nx)))

# ---------------------------
# Step 9. Animation
# ---------------------------
if distributions_2d:
    mpl.rcParams['animation.embed_limit'] = 30
    fig, ax = plt.subplots(figsize=(6, 4))
    norm = mpl.colors.LogNorm(vmin=1e-4, vmax=1)
    im = ax.imshow(distributions_2d[0] + 1e-6, origin='lower', cmap='cool', norm=norm, aspect='auto', animated=True)
    plt.colorbar(im, ax=ax)
    plt.title(f"Heat Flow Animation ({nx}x{ny} Grid)")
    plt.xlabel("x index")
    plt.ylabel("y index")

    def update(frame):
        im.set_array(distributions_2d[frame % len(distributions_2d)] + 1e-6)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(distributions_2d), interval=50, blit=True)
    ani.save("2D_diffusion_cudaq.gif", writer="pillow")

# ---------------------------
# Step 10. Plot energy
# ---------------------------
times = np.linspace(0, total_time, len(energy_history))
h_exp_val = np.array(energy_history)

plt.figure()
plt.plot(times, h_exp_val, label="VarQITE")
plt.xlabel("Time")
plt.ylabel(r"$\langle H \rangle$ (energy)")
plt.legend(loc="upper right")
plt.savefig("energy_evolution_cudaq.png")
plt.show()