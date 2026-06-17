import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # Grid setup
    size = 200
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # Atom positions and properties
    atoms = [
        {'pos': [2, 0], 'freq': 2.0, 'amp': 1.0, 'phase': 0},
        {'pos': [-2, 1], 'freq': 3.0, 'amp': 0.8, 'phase': np.pi/4},
        {'pos': [0, -2], 'freq': 4.0, 'amp': 1.2, 'phase': np.pi/2},
    ]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Main field plot
    ax1.set_title('Electromagnetic Field from Oscillating Atoms')
    im = ax1.imshow(np.zeros((size, size)), extent=[-10, 10, -10, 10], 
                   cmap='plasma', vmin=0, vmax=2, origin='lower')
    
    # Mark atoms
    for i, atom in enumerate(atoms):
        ax1.plot(atom['pos'][0], atom['pos'][1], 'wo', markersize=8, 
                markeredgecolor='black', label=f"Atom {i+1}")
    ax1.legend()
    
    # Spectroscope
    ax2.set_title('Frequency Spectrum')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 1)
    spectrum_bars = ax2.bar([2, 3, 4], [0, 0, 0], width=0.2, 
                          color=['red', 'green', 'blue'], alpha=0.7)
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Intensity')
    
    # Time series
    ax3.set_title('Field Oscillation at Origin')
    ax3.set_xlim(0, 20)
    ax3.set_ylim(-3, 3)
    time_data = []
    field_data = []
    line, = ax3.plot([], [], 'b-')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Field Amplitude')
    
    # System info
    ax4.set_title('System Parameters')
    ax4.axis('off')
    info_text = ax4.text(0.1, 0.8, '', fontfamily='monospace', fontsize=10)
    
    def animate(frame):
        t = frame * 0.1
        field = np.zeros((size, size))
        
        # Add disturbance every 100 frames
        disturbance_factor = 1.0
        if frame % 100 < 10:
            disturbance_factor = 2.0
        
        # Calculate field from each atom
        for atom in atoms:
            ax_pos, ay_pos = atom['pos']
            freq = atom['freq']
            amp = atom['amp'] * disturbance_factor
            phase = atom['phase']
            
            # Distance from atom
            dx = X - ax_pos
            dy = Y - ay_pos
            r = np.sqrt(dx**2 + dy**2) + 0.1
            
            # Oscillating field with damping
            damping = np.exp(-t * 0.05)
            oscillation = amp * np.sin(freq * t + phase) * damping
            
            # Field contribution (simplified dipole)
            field += oscillation / r * np.exp(-r * 0.1)
        
        # Update main plot
        field_magnitude = np.abs(field)
        im.set_array(field_magnitude)
        im.set_clim(0, np.max(field_magnitude) + 0.1)
        
        # Update spectrum (show current intensities)
        for i, (atom, bar) in enumerate(zip(atoms, spectrum_bars)):
            intensity = atom['amp'] * np.exp(-t * 0.05)
            bar.set_height(intensity)
        
        # Update time series
        center_field = field[size//2, size//2]
        time_data.append(t)
        field_data.append(center_field)
        
        # Keep last 200 points
        if len(time_data) > 200:
            time_data.pop(0)
            field_data.pop(0)
        
        line.set_data(time_data, field_data)
        if time_data:
            ax3.set_xlim(max(0, time_data[-1] - 20), time_data[-1] + 2)
        
        # Update info
        status = "DISTURBED" if frame % 100 < 10 else "OSCILLATING"
        info = f"""Time: {t:.1f}

Status: {status}

Atom Frequencies:
  Atom 1: 2.0 Hz
  Atom 2: 3.0 Hz  
  Atom 3: 4.0 Hz

Damping: e^(-0.05t)

Observable: EM radiation
at atom frequencies"""
        
        info_text.set_text(info)
        
        return [im, line, info_text] + list(spectrum_bars)
    
    anim = animation.FuncAnimation(fig, animate, frames=600, 
                                 interval=100, blit=False)
    
    plt.tight_layout()
    
    # Save animation
    try:
        anim.save('atomic_oscillation.gif', writer='pillow', fps=10)
        print("Animation saved as 'atomic_oscillation.gif'")
    except:
        print("Could not save animation")
    
    # Show plot
    plt.show()
    
    return anim

# Keep animation alive
anim = None
if __name__ == "__main__":
    anim = main()