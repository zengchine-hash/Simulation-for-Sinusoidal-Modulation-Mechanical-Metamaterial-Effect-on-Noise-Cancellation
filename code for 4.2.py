import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. CORE PHYSICS ENGINE: BANDS CALCULATOR
# =============================================================================
def compute_lattice_eigenvalues(N, alpha, q_vectors, k0=1.0, m=1.0):
    """
    Computes the dispersion eigenvalues (omega) for a given N and alpha 
    across a defined vector of wavevectors (q).
    """
    # Generate stiffness array for n = 1 to N+1 to cover cell boundary coupling
    n_indices = np.arange(1, N + 2)
    k = k0 * (1.0 + alpha * np.sin(2.0 * np.pi * n_indices / N))
    
    omega_records = []
    for q in q_vectors:
        # Initialize the complex Hermitian Dynamical Matrix
        D = np.zeros((N, N), dtype=complex)
        
        # Populate main diagonal and localized nearest-neighbor coupling
        for i in range(N):
            D[i, i] = (k[i] + k[i+1]) / m
            if i > 0:
                D[i, i-1] = -k[i] / m
            if i < N - 1:
                D[i, i+1] = -k[i+1] / m
                
        # Inject Bloch periodic boundary conditions into the extreme corners
        D[0, N-1] = -k[0] * np.exp(-1j * q * N) / m
        D[N-1, 0] = -k[0] * np.exp(1j * q * N) / m
        
        # Diagonalize matrix and extract roots (frequencies omega)
        eigvals = np.linalg.eigvalsh(D)
        eigvals = np.maximum(eigvals, 0.0) # Numerical safety filter against tiny negative roots
        omega_records.append(np.sqrt(eigvals))
        
    return np.array(omega_records) # Shape: (len(q_vectors), N)

# =============================================================================
# 2. DATA GENERATION PIPELINE FOR PARAMETRIC SWEEPS
# =============================================================================
print("Generating data for Plot A (Sweeping Alpha)...")
# Setup for Plot A: Fix N, continuously sweep alpha
N_fixed_A = 5
alpha_sweep = np.linspace(0.0, 0.8, 120)
q_vec_A = np.linspace(0, np.pi / N_fixed_A, 40)

plot_A_alpha = []
plot_A_omega = []

for alpha in alpha_sweep:
    omegas = compute_lattice_eigenvalues(N_fixed_A, alpha, q_vec_A)
    # Flatten the matrix to log individual points for the scatter plot
    for om in omegas.flatten():
        plot_A_alpha.append(alpha)
        plot_A_omega.append(om)

print("Generating data for Plot B (Sweeping N)...")
# Setup for Plot B: Fix alpha, discretely sweep N from 3 to 10
alpha_fixed_B = 0.6
N_sweep = np.arange(3, 11)

plot_B_N = []
plot_B_omega = []

for N in N_sweep:
    q_vec_B = np.linspace(0, np.pi / N, 60)
    omegas = compute_lattice_eigenvalues(N, alpha_fixed_B, q_vec_B)
    for om in omegas.flatten():
        # Introduce a tiny horizontal jitter to turn discrete lines into clean blocks
        jittered_N = N + np.random.uniform(-0.15, 0.15)
        plot_B_N.append(jittered_N)
        plot_B_omega.append(om)

# =============================================================================
# 3. HIGH-QUALITY PUBLICATION VISUALIZATION
# =============================================================================
plt.rcParams['font.family'] = 'serif'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300)

# --- Subplot A: Frequency vs. Modulation Amplitude (Alpha) ---
ax1.scatter(plot_A_alpha, plot_A_omega, s=0.4, color='#1f77b4', alpha=0.7, label='Passband States')
ax1.set_title("A: Bandgap Evolution vs. Modulation Amplitude", fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel("Modulation Amplitude ($\\alpha$)", fontsize=11)
ax1.set_ylabel("Dimensionless Frequency ($\\omega$)", fontsize=11)
ax1.set_xlim(0.0, 0.8)
ax1.set_ylim(0.0, 2.3)
ax1.grid(True, linestyle=':', alpha=0.5)



# --- Subplot B: Frequency vs. Modulation Period (N) ---
ax2.scatter(plot_B_N, plot_B_omega, s=0.3, color='#2c3e50', alpha=0.7)
ax2.set_title("B: Bandgap Topography vs. Modulation Period", fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel("Modulation Period ($N$)", fontsize=11)
ax2.set_ylabel("Dimensionless Frequency ($\\omega$)", fontsize=11)
ax2.set_xticks(N_sweep)
ax2.set_xlim(2.5, 10.5)
ax2.set_ylim(0.0, 2.3)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('parametric_sensitivity_maps.png', bbox_inches='tight', dpi=300)
plt.show()
print("Parametric figures generated and exported successfully.")