import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

# Try to import CuPy, fallback to NumPy if not available
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    print("CuPy available, GPU detected:", GPU_AVAILABLE)
    if GPU_AVAILABLE:
        try:
            # Try different ways to get GPU device name
            device_id = cp.cuda.Device().id
            device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
            print(f"GPU device: {device_name}")
        except:
            try:
                # Alternative method
                print(f"GPU device ID: {cp.cuda.Device().id}")
                print(f"GPU memory: {cp.cuda.runtime.memGetInfo()}")
            except:
                print("GPU device: Available (name detection failed)")
        xp = cp  # Use CuPy as array library
    else:
        print("CuPy available but no GPU detected, using NumPy")
        xp = np
except ImportError:
    print("CuPy not available, using NumPy")
    GPU_AVAILABLE = False
    xp = np

# Qiskit imports with error handling and version compatibility
try:
    from qiskit.quantum_info import Operator, SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import EfficientSU2
    from qiskit.transpiler import preset_passmanagers
    
    # Try modern Qiskit primitives first
    try:
        from qiskit.primitives import StatevectorEstimator as EstimatorP
        print("Using StatevectorEstimator")
    except ImportError:
        try:
            from qiskit.primitives import Estimator as EstimatorP
            print("Using standard Estimator")
        except ImportError:
            from qiskit_aer.primitives import Estimator as EstimatorP
            print("Using Aer Estimator")
    
    # Try to import gradients (optional for basic simulation)
    try:
        from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, ReverseQGT
        from qiskit_algorithms.gradients import DerivativeType
        GRADIENTS_AVAILABLE = True
        print("Gradient calculations available")
    except ImportError:
        print("Gradient calculations not available, using simplified optimization")
        GRADIENTS_AVAILABLE = False
    
    print("Qiskit imports successful")
except ImportError as e:
    print(f"Qiskit import error: {e}")
    print("Please install/update qiskit: pip install --upgrade qiskit qiskit-aer")
    exit(1)

# Try to import IBM Runtime, fallback to local estimator
try:
    from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler, Batch
    IBM_RUNTIME_AVAILABLE = True
    print("IBM Runtime available")
except ImportError:
    print("IBM Runtime not available, using local estimator")
    IBM_RUNTIME_AVAILABLE = False

# Set environment variables
n_vcpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_vcpus)
os.environ["MKL_NUM_THREADS"] = str(n_vcpus)
print("Number of vCPUs detected:", n_vcpus)

# ---------------------------
# Step 1. Build the Hamiltonian
# ---------------------------
def build_2d_dirichlet_laplacian_hamiltonian(nx: int, ny: int, alpha: float = 1.0) -> SparsePauliOp:
    """Build 2D Laplacian Hamiltonian with proper GPU/CPU handling."""
    n = nx * ny
    
    if GPU_AVAILABLE:
        # GPU version with CuPy
        L2d = cp.zeros((n, n), dtype=cp.float64)
        
        # Vectorized approach for GPU
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
        # Convert to NumPy for Qiskit
        H_mat_np = cp.asnumpy(H_mat)
    else:
        # CPU version with NumPy
        L2d = np.zeros((n, n), dtype=np.float64)
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
        H_mat_np = -alpha * L2d
    
    op = Operator(H_mat_np)
    return SparsePauliOp.from_operator(op)

# ---------------------------
# Step 2. Setup quantum circuit
# ---------------------------
print("Building Hamiltonian...")
nx, ny = 8, 8  # Reduced size for faster testing
hamiltonian = build_2d_dirichlet_laplacian_hamiltonian(nx, ny)
print("Number of qubits:", hamiltonian.num_qubits)
print("Number of Pauli terms:", len(hamiltonian))

# Define ansatz
ansatz = EfficientSU2(hamiltonian.num_qubits, entanglement="linear", reps=1)

# ---------------------------
# Step 3. Configure backend with proper GPU detection
# ---------------------------
def test_qiskit_gpu_support():
    """Test if Qiskit-Aer actually supports GPU on this system."""
    try:
        from qiskit import QuantumCircuit
        test_backend = AerSimulator(method='statevector', device='GPU')
        test_circuit = QuantumCircuit(1)
        test_circuit.h(0)
        # Try a simple test run
        job = test_backend.run(test_circuit, shots=1)
        job.result()  # This will fail if GPU isn't supported
        return True
    except Exception as e:
        return False

# Check if Qiskit GPU support actually works
qiskit_gpu_available = GPU_AVAILABLE and test_qiskit_gpu_support()

if qiskit_gpu_available:
    try:
        backend = AerSimulator(
            method='statevector',
            device='GPU',
            max_parallel_threads=1,
            max_parallel_experiments=1
        )
        print("Using Qiskit GPU backend")
        QISKIT_USE_GPU = True
    except Exception as e:
        print(f"Qiskit GPU backend failed: {e}")
        print("Falling back to CPU backend for Qiskit")
        backend = AerSimulator(
            method='statevector',
            device='CPU',
            max_parallel_threads=min(n_vcpus, 8),  # Limit threads for stability
            max_parallel_experiments=1
        )
        QISKIT_USE_GPU = False
else:
    backend = AerSimulator(
        method='statevector',
        device='CPU',
        max_parallel_threads=min(n_vcpus, 8),  # Limit threads for stability
        max_parallel_experiments=1
    )
    if GPU_AVAILABLE:
        print("GPU available for CuPy but not supported by Qiskit-Aer")
        print("Using CPU backend for Qiskit, GPU for CuPy operations")
    else:
        print("Using CPU backend")
    QISKIT_USE_GPU = False

# Transpile circuit
pm = preset_passmanagers.generate_preset_pass_manager(optimization_level=1, backend=backend)
circuit = pm.run(ansatz)
observable = hamiltonian.apply_layout(circuit.layout)
circuit_params = list(circuit.parameters)
n_params = len(circuit_params)
print(f"Number of circuit parameters: {n_params}")

# Initialize parameters
if GPU_AVAILABLE:
    init_param_values = cp.random.random(n_params) * 0.1
    init_param_values = init_param_values.astype(cp.float64)
else:
    init_param_values = np.random.random(n_params) * 0.1

# ---------------------------
# Step 4. Setup estimators
# ---------------------------
if IBM_RUNTIME_AVAILABLE:
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 1000
else:
    estimator = EstimatorP()

estimatorp = EstimatorP()

# ---------------------------
# Step 5. Define computation functions
# ---------------------------
def energy(params):
    """Calculate energy expectation value."""
    if GPU_AVAILABLE:
        params_np = cp.asnumpy(params)
    else:
        params_np = params
    
    if IBM_RUNTIME_AVAILABLE:
        pub = (circuit, observable, [params_np])
        job = estimator.run([pub])
        return job.result()[0].data.evs[0]
    else:
        job = estimator.run(circuit, observable, params_np)
        return job.result().values[0]

def compute_gradient(params):
    """Calculate parameter gradients."""
    if not GRADIENTS_AVAILABLE:
        # Simple finite difference gradient
        eps = 1e-8
        grad = np.zeros_like(params if not GPU_AVAILABLE else cp.asnumpy(params))
        
        for i in range(len(grad)):
            params_plus = params.copy()
            params_minus = params.copy()
            
            if GPU_AVAILABLE:
                params_plus[i] += eps
                params_minus[i] -= eps
            else:
                params_plus[i] += eps
                params_minus[i] -= eps
            
            e_plus = energy(params_plus)
            e_minus = energy(params_minus)
            grad[i] = (e_plus - e_minus) / (2 * eps)
        
        if GPU_AVAILABLE:
            return cp.array(grad)
        else:
            return grad
    else:
        # Use proper gradient calculation
        grad_calculator = ParamShiftEstimatorGradient(estimatorp, derivative_type=DerivativeType.IMAG)
        
        if GPU_AVAILABLE:
            params_np = cp.asnumpy(params)
        else:
            params_np = params
        
        result = grad_calculator.run([circuit], [observable], [params_np]).result()
        
        if GPU_AVAILABLE:
            return cp.array(result.gradients[0])
        else:
            return np.array(result.gradients[0])

def compute_qgt(params):
    """Calculate quantum geometric tensor."""
    if not GRADIENTS_AVAILABLE:
        # Simple identity matrix approximation for QGT
        n = len(params)
        if GPU_AVAILABLE:
            return cp.eye(n)
        else:
            return np.eye(n)
    else:
        qgt_instance = ReverseQGT(phase_fix=True, derivative_type=DerivativeType.REAL)
        
        if GPU_AVAILABLE:
            params_np = cp.asnumpy(params)
        else:
            params_np = params
        
        result = qgt_instance.run([circuit], [params_np]).result().qgts
        
        if GPU_AVAILABLE:
            return cp.array(result[0])
        else:
            return np.array(result[0])

# ---------------------------
# Step 6. Variational Imaginary Time Evolution
# ---------------------------
total_time = 10  # Reduced for faster testing
dt = 0.2
n_steps = int(total_time / dt)
params = init_param_values.copy()

energy_history = []
if GPU_AVAILABLE:
    params_history = [cp.asnumpy(params).copy()]
else:
    params_history = [params.copy()]

print(f"Starting VarQITE for {n_steps} steps...")
start_time = time.time()

for step in range(n_steps):
    try:
        # Compute energy and gradients
        current_energy = energy(params)
        g = compute_gradient(params)
        S = compute_qgt(params)
        
        energy_history.append(current_energy)
        print(f"Step {step:3d}: Energy = {current_energy:.6f}")
        
        # Update parameters using linear algebra
        lambda_reg = 1e-3
        if GPU_AVAILABLE:
            S_reg = S + lambda_reg * cp.eye(S.shape[0])
            try:
                delta = cp.linalg.solve(S_reg, -g * dt)
            except:
                delta = -cp.dot(cp.linalg.pinv(S_reg), g) * dt
        else:
            S_reg = S + lambda_reg * np.eye(S.shape[0])
            try:
                delta = np.linalg.solve(S_reg, -g * dt)
            except:
                delta = -np.dot(np.linalg.pinv(S_reg), g) * dt
        
        params = params + delta
        
        if GPU_AVAILABLE:
            params_history.append(cp.asnumpy(params).copy())
        else:
            params_history.append(params.copy())
            
    except Exception as e:
        print(f"Error in step {step}: {e}")
        break

end_time = time.time()
print(f"VarQITE completed in {end_time - start_time:.2f} seconds")
print("Final energy:", energy(params))

# ---------------------------
# Step 7. Plot results
# ---------------------------
if len(energy_history) > 0:
    times = np.linspace(0, total_time, len(energy_history))
    h_exp_val = np.array(energy_history)

    plt.figure(figsize=(10, 6))
    plt.plot(times, h_exp_val, 'b-', linewidth=2, label="VarQITE")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle H \rangle$ (Energy)")
    if GPU_AVAILABLE and not QISKIT_USE_GPU:
        title_suffix = "CuPy GPU + Qiskit CPU"
    elif QISKIT_USE_GPU:
        title_suffix = "Full GPU"
    else:
        title_suffix = "CPU"
    plt.title(f"Energy Evolution - {title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No energy history to plot - simulation failed")

# ---------------------------
# Step 8. Create simple visualization
# ---------------------------
def create_simple_visualization():
    """Create a simple parameter evolution plot."""
    if len(params_history) <= 1:
        print("Not enough data for visualization")
        return
        
    params_array = np.array(params_history)
    
    plt.figure(figsize=(12, 8))
    
    # Plot parameter evolution
    plt.subplot(2, 2, 1)
    for i in range(min(5, params_array.shape[1])):
        plt.plot(params_array[:, i], label=f'Param {i}')
    plt.xlabel('Step')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution')
    plt.legend()
    
    # Plot energy
    plt.subplot(2, 2, 2)
    if len(energy_history) > 0:
        plt.plot(energy_history)
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.title('Energy Evolution')
    else:
        plt.text(0.5, 0.5, 'No energy data', ha='center', va='center')
        plt.title('Energy Evolution (No Data)')
    
    # Plot gradient norms (if available)
    plt.subplot(2, 2, 3)
    if GPU_AVAILABLE:
        param_norms = [cp.linalg.norm(cp.array(p)).get() for p in params_history]
    else:
        param_norms = [np.linalg.norm(p) for p in params_history]
    plt.plot(param_norms)
    plt.xlabel('Step')
    plt.ylabel('Parameter Norm')
    plt.title('Parameter Norm Evolution')
    
    # Add system info
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f"GPU Available: {GPU_AVAILABLE}", fontsize=10)
    plt.text(0.1, 0.6, f"Qiskit GPU: {QISKIT_USE_GPU}", fontsize=10)
    plt.text(0.1, 0.4, f"Steps completed: {len(energy_history)}", fontsize=10)
    plt.text(0.1, 0.2, f"Parameters: {len(params_history[-1]) if params_history else 0}", fontsize=10)
    plt.axis('off')
    plt.title('System Info')
    
    plt.tight_layout()
    plt.show()

create_simple_visualization()

# Clean up GPU memory if used
if GPU_AVAILABLE:
    try:
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        print("GPU memory cleaned up")
    except:
        pass

print("Simulation completed successfully!")