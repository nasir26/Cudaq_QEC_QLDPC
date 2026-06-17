import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import cupy as cp  # GPU acceleration
from numba import cuda
import math

class AtomicOscillationVisualizer:
    def __init__(self, grid_size=512, dt=0.01, c=1.0):
        self.grid_size = grid_size
        self.dt = dt
        self.c = c  # speed of light (normalized)
        
        # GPU arrays
        self.Ex = cp.zeros((grid_size, grid_size), dtype=cp.float32)
        self.Ey = cp.zeros((grid_size, grid_size), dtype=cp.float32)
        self.Bz = cp.zeros((grid_size, grid_size), dtype=cp.float32)
        
        # Atom positions and properties
        self.n_atoms = 5
        self.atom_positions = cp.array([
            [grid_size//2 + 50, grid_size//2],
            [grid_size//2 - 50, grid_size//2 + 30],
            [grid_size//2, grid_size//2 - 50],
            [grid_size//2 + 30, grid_size//2 + 40],
            [grid_size//2 - 30, grid_size//2 - 30]
        ], dtype=cp.int32)
        
        # Each atom has different natural frequency
        self.frequencies = cp.array([0.8, 1.2, 1.5, 2.0, 0.6], dtype=cp.float32)
        self.amplitudes = cp.array([1.0, 0.8, 1.2, 0.9, 1.1], dtype=cp.float32)
        self.phases = cp.zeros(self.n_atoms, dtype=cp.float32)
        self.damping = cp.array([0.995, 0.996, 0.994, 0.997, 0.993], dtype=cp.float32)
        
        # Disturbance parameters
        self.disturbed = cp.ones(self.n_atoms, dtype=cp.bool_)
        self.time = 0
        
        # Coordinate arrays
        x = cp.arange(grid_size)
        y = cp.arange(grid_size)
        self.X, self.Y = cp.meshgrid(x, y)
        
        # Spectroscope data
        self.spectral_data = cp.zeros(1000, dtype=cp.complex64)
        self.frequency_bins = cp.linspace(0, 5, 1000)
        
    @staticmethod
    @cuda.jit
    def update_fields_kernel(Ex, Ey, Bz, atom_pos, frequencies, amplitudes, 
                           phases, damping, time, dt, c, X, Y):
        i, j = cuda.grid(2)
        if i >= Ex.shape[0] or j >= Ex.shape[1]:
            return
            
        # Reset fields
        Ex[i, j] = 0
        Ey[i, j] = 0
        Bz[i, j] = 0
        
        # Calculate field contributions from each oscillating atom
        for atom_idx in range(atom_pos.shape[0]):
            ax, ay = atom_pos[atom_idx, 0], atom_pos[atom_idx, 1]
            freq = frequencies[atom_idx]
            amp = amplitudes[atom_idx]
            phase = phases[atom_idx]
            damp = damping[atom_idx]
            
            # Distance from atom
            dx = X[i, j] - ax
            dy = Y[i, j] - ay
            r = math.sqrt(dx*dx + dy*dy) + 1e-6
            
            # Retarded time
            t_ret = time - r/c
            
            # Oscillating dipole field (near and far field approximation)
            if t_ret > 0:
                omega_t = freq * t_ret + phase
                oscillation = amp * math.sin(omega_t) * math.pow(damp, t_ret)
                
                # Electric field components (dipole radiation pattern)
                sin_theta = abs(dy) / r
                field_strength = oscillation / (r + 1) * sin_theta
                
                Ex[i, j] += field_strength * (dx / r)
                Ey[i, j] += field_strength * (dy / r)
                
                # Magnetic field (perpendicular to E and propagation)
                Bz[i, j] += field_strength / c
    
    def update_simulation(self):
        # Update time and phases
        self.time += self.dt
        
        # Apply damping to amplitudes
        self.amplitudes *= self.damping
        
        # Launch GPU kernel
        threads_per_block = (16, 16)
        blocks_per_grid_x = (self.grid_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self.grid_size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        self.update_fields_kernel[blocks_per_grid, threads_per_block](
            self.Ex, self.Ey, self.Bz,
            self.atom_positions, self.frequencies, self.amplitudes,
            self.phases, self.damping, self.time, self.dt, self.c,
            self.X, self.Y
        )
        
        # Update spectral analysis
        center_idx = self.grid_size // 2
        field_sample = self.Ex[center_idx-10:center_idx+10, center_idx-10:center_idx+10]
        field_intensity = cp.mean(cp.abs(field_sample))
        
        # Rolling spectral data
        self.spectral_data[:-1] = self.spectral_data[1:]
        self.spectral_data[-1] = field_intensity * cp.exp(1j * self.time)
    
    def get_spectrum(self):
        # FFT for frequency analysis
        fft_data = cp.fft.fft(self.spectral_data)
        power_spectrum = cp.abs(fft_data[:len(self.frequency_bins)])
        return cp.asnumpy(power_spectrum)
    
    def get_field_data(self):
        # Transfer data back to CPU for visualization
        Ex_cpu = cp.asnumpy(self.Ex)
        Ey_cpu = cp.asnumpy(self.Ey)
        field_magnitude = np.sqrt(Ex_cpu**2 + Ey_cpu**2)
        return field_magnitude, Ex_cpu, Ey_cpu
    
    def add_disturbance(self, strength=2.0):
        # Add random disturbance to atom amplitudes
        disturbance = cp.random.uniform(-strength, strength, self.n_atoms)
        self.amplitudes += disturbance
        self.phases += cp.random.uniform(0, 2*np.pi, self.n_atoms)

def main():
    # Initialize simulation
    sim = AtomicOscillationVisualizer()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Field visualization
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Electromagnetic Field from Oscillating Atoms')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Position')
    
    # Spectroscope
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Spectroscope - Observed Frequencies')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Intensity')
    ax2.set_xlim(0, 5)
    
    # Time series
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Field Oscillation at Center')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Field Amplitude')
    
    # Initialize plots
    field_magnitude, _, _ = sim.get_field_data()
    im = ax1.imshow(field_magnitude, animated=True, cmap='plasma', 
                    vmin=0, vmax=0.1, origin='lower')
    
    # Add atom positions
    atom_pos_cpu = cp.asnumpy(sim.atom_positions)
    for i, (x, y) in enumerate(atom_pos_cpu):
        ax1.plot(x, y, 'wo', markersize=8, markeredgecolor='black', 
                markeredgewidth=2, label=f'Atom {i+1}' if i < 3 else '')
    
    ax1.legend(loc='upper right')
    plt.colorbar(im, ax=ax1, label='Field Magnitude')
    
    spectrum_line, = ax2.plot([], [], 'r-', linewidth=2)
    time_series_data = []
    time_data = []
    time_line, = ax3.plot([], [], 'b-', linewidth=1)
    
    def animate(frame):
        # Add disturbance every 200 frames
        if frame % 200 == 0 and frame > 0:
            sim.add_disturbance(1.5)
        
        # Update simulation
        sim.update_simulation()
        
        # Get field data
        field_magnitude, _, _ = sim.get_field_data()
        im.set_array(field_magnitude)
        im.set_clim(vmin=0, vmax=np.max(field_magnitude) * 1.1 + 1e-6)
        
        # Update spectroscope
        spectrum = sim.get_spectrum()
        spectrum_line.set_data(cp.asnumpy(sim.frequency_bins[:len(spectrum)]), spectrum)
        ax2.set_ylim(0, np.max(spectrum) * 1.1 + 1e-6)
        
        # Update time series
        center_field = field_magnitude[sim.grid_size//2, sim.grid_size//2]
        time_series_data.append(center_field)
        time_data.append(sim.time)
        
        if len(time_series_data) > 500:
            time_series_data.pop(0)
            time_data.pop(0)
            
        time_line.set_data(time_data, time_series_data)
        ax3.set_xlim(max(0, sim.time - 50), sim.time + 5)
        if time_series_data:
            ax3.set_ylim(min(time_series_data) * 1.1 - 1e-6, 
                        max(time_series_data) * 1.1 + 1e-6)
        
        return [im, spectrum_line, time_line]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=2000, 
                                 interval=50, blit=False, repeat=True)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    
    return anim

if __name__ == "__main__":
    animation_obj = main()