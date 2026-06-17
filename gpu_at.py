import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.fft import fft, fftfreq
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
omega0 = 5.0  # natural frequency
gamma = 0.0   # damping coefficient (set to 0 for undamped, statevector simulation)
c = 1.0       # speed of light (normalized)
b = gamma / 2
w = np.sqrt(omega0**2 - b**2)

grid_size = 400
x = np.linspace(-10, 10, grid_size)
y = np.linspace(-10, 10, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

num_frames = 300
t_max = 60.0
times = np.linspace(0, t_max, num_frames)

# Fine time grid for quantum simulation
num_fine = 10000
t_fine = np.linspace(0, t_max, num_fine)

# Qiskit simulation using GPU statevector
sim = AerSimulator(method='statevector', device='GPU')
circuits = []
for t in t_fine:
    qc = QuantumCircuit(1)
    qc.h(0)  # Initialize in superposition
    qc.rz(omega0 * t, 0)  # Time evolution with omega0
    qc.save_statevector()  # Explicitly save statevector
    circuits.append(qc)

# Run simulation with statevector saving
result = sim.run(circuits, shots=1).result()

# Extract expectation value of Pauli X
x_data = np.zeros(num_fine)
for i in range(num_fine):
    state = result.get_statevector(i)
    x_data[i] = np.real(state.expectation_value(Pauli('X')))

# Acceleration ax = -omega0**2 * x (undamped)
ax_data = -omega0**2 * x_data

# Transfer to torch
t_fine_torch = torch.tensor(t_fine, device=device)
ax_fine = torch.tensor(ax_data, device=device)
X = torch.tensor(X, device=device)
Y = torch.tensor(Y, device=device)
R = torch.sqrt(X**2 + Y**2)
times = torch.tensor(times, device=device)

# Precompute fields on GPU
fields = torch.zeros(num_frames, grid_size, grid_size, device=device)
for i in range(num_frames):
    ret_t = (times[i] - R / c).clamp(min=0).cpu().numpy()  # Move to CPU for np.interp
    ax_ret = np.interp(ret_t, t_fine, ax_data)  # Use numpy for interpolation
    fields[i] = torch.tensor(ax_ret, device=device) / (R + 1e-6)

# Transfer to CPU for plotting
fields_cpu = fields.cpu().numpy()

# Animation setup
fig, ax = plt.subplots()
im = ax.imshow(fields_cpu[0], cmap='RdBu', extent=(-10, 10, -10, 10), origin='lower', vmin=-0.1, vmax=0.1)
ax.set_title('Quantum Electromagnetic Field Oscillation')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame):
    im.set_array(fields_cpu[frame])
    return [im]

ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# Spectrum at a far point
r0 = 8.0
t_spec = np.linspace(0, t_max, 10000)
ret_t_spec = (t_spec - r0 / c).clip(min=0)
ax_spec = np.interp(ret_t_spec, t_fine, ax_data)

# FFT
N = len(ax_spec)
freq = fftfreq(N, d=t_spec[1] - t_spec[0])[:N//2]
spectrum = (2.0 / N) * np.abs(fft(torch.tensor(ax_spec, device=device)).cpu().numpy()[:N//2])

fig_spec, ax_spec = plt.subplots()
ax_spec.plot(freq, spectrum)
ax_spec.set_title('Frequency Spectrum')
ax_spec.set_xlabel('Frequency')
ax_spec.set_ylabel('Amplitude')
ax_spec.set_xlim(0, 10)

# Save outputs
ani.save('em_field_animation.gif', writer='pillow', fps=20)
fig_spec.savefig('spectrum_plot.png')
plt.close(fig_spec)
plt.close(fig)