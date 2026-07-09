import numpy as np
import matplotlib.pyplot as plt

def calculate_band_structure(N, alpha, k0=1.0, m=1.0, num_q=200):
    """Calculates the band structure (dispersion relations) for the 1D lattice."""
    n_index = np.arange(1, N + 1)
    k_array = k0 * (1.0 + alpha * np.sin(2.0 * np.pi * n_index / N))
    qa_array = np.linspace(0, np.pi / N, num_q)
    frequencies = np.zeros((num_q, N))
    
    for i, qa in enumerate(qa_array):
        D = np.zeros((N, N), dtype=complex)
        for row in range(N):
            k_right = k_array[row]
            k_left  = k_array[row - 1] 
            D[row, row] = (k_right + k_left) / m
            if row < N - 1:
                D[row, row + 1] = -k_right / m
                D[row + 1, row] = -k_right / m
        if N > 1:
            phase_shift = np.exp(1j * qa * N)
            D[0, N - 1] = (-k_array[-1] / m) * np.conj(phase_shift)
            D[N - 1, 0] = (-k_array[-1] / m) * phase_shift
            
        eigenvalues = np.linalg.eigvalsh(D)
        frequencies[i, :] = np.sqrt(np.abs(eigenvalues))
        
    return qa_array, frequencies

def simulate_transmission(N, alpha, omega_array, total_cells=10, k0=1.0, m=1.0):
    """
    Simulates a forced harmonic excitation at the left boundary and computes 
    the resultant output amplitude at the right boundary as a function of omega.
    """
    total_masses = N * total_cells
    n_index = np.arange(1, total_masses + 1)
    
    # Expand the modulated spring profile across the full chain length
    k_global = k0 * (1.0 + alpha * np.sin(2.0 * np.pi * n_index / N))
    
    amplitudes = np.zeros(len(omega_array))
    
    for idx, omega in enumerate(omega_array):
        # Construct the global mass-displacement governing matrix: (K - w^2 * M) * U = F
        A = np.zeros((total_masses, total_masses), dtype=complex)
        
        for i in range(total_masses):
            k_right = k_global[i]
            k_left  = k_global[i - 1] if i > 0 else k0
            
            A[i, i] = (k_right + k_left) - m * (omega**2)
            
            if i > 0:
                A[i, i - 1] = -k_left
            if i < total_masses - 1:
                A[i, i + 1] = -k_right
                
        # Define a unit force vector applied at the first node (Input Amplitude = 1)
        F = np.zeros(total_masses)
        F[0] = 1.0
        
        try:
            # Solve the linear system to get the steady-state displacement of all masses
            U = np.linalg.solve(A, F)
            # Fetch the absolute amplitude at the absolute right boundary (last mass)
            amplitudes[idx] = np.abs(U[-1])
        except np.linalg.LinAlgError:
            amplitudes[idx] = 0.0
            
    # Normalize the output response so that maximum unattenuated transmission equals 1
    if np.max(amplitudes) > 0:
        amplitudes = amplitudes / np.max(amplitudes)
        
    return amplitudes

# ==============================================================================
# Simulation Execution and Academic Visualization
# ==============================================================================

# Model Configurations
modulation_period = 5       # N = 5
modulation_amplitude = 0.6  # alpha = 0.6
omega_scan = np.linspace(0.0, 2.4, 500) # Frequency sweep range for transmission

# Execute Solver Engines
wavevectors, omega_branches = calculate_band_structure(N=modulation_period, alpha=modulation_amplitude)
output_amplitudes = simulate_transmission(N=modulation_period, alpha=modulation_amplitude, omega_array=omega_scan)

# Initialize a subplotted layout for professional comparative review
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharey=True)

# ------------------------------------------------------------------------------
# Figure A: Phononic Band Structure (Left Panel)
# ------------------------------------------------------------------------------
for branch in range(modulation_period):
    ax1.plot(wavevectors, omega_branches[:, branch], color='#2980b9', linewidth=2.2, label='Dispersion Curves' if branch==0 else "")
ax1.set_title('Figure A: Phononic Band Structure', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Dimensionless Wavevector ($qa$)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Angular Frequency ($\\omega$)', fontsize=11, fontweight='bold')
ax1.set_xlim(0, np.pi / modulation_period)
ax1.set_ylim(0, 2.4)
ax1.grid(True, linestyle='--', alpha=0.5)

# ------------------------------------------------------------------------------
# Figure B: Resultant Wave Amplitude Response (Right Panel)
# ------------------------------------------------------------------------------
ax2.plot(output_amplitudes, omega_scan, color='#e74c3c', linewidth=2.0, label='Resultant Amplitude')
ax2.set_title('Figure B: Transmitted Wave Amplitude', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Resultant Wave Amplitude ($U_{out} / U_{in}$)', fontsize=11, fontweight='bold')
ax2.set_xlim(0, 1.1)
ax2.axvline(x=1.0, color='#7f8c8d', linestyle=':', alpha=0.7)
ax2.grid(True, linestyle='--', alpha=0.5)

# Perfect Layout Presentation
plt.tight_layout()
plt.show()