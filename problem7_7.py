import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# Use LaTeX and serif font
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
plt.rcParams['text.usetex'] = True

def getCnCoefficients(alpha, r, theta, nMax):

    # 1. Calculate Gamma as defined by user
    gamma = alpha * np.cosh(r) + np.conj(alpha) * np.exp(1j * theta) * np.sinh(r)
    
    cosh_r = np.cosh(r)
    tanh_r = np.tanh(r)
    sinh_2r = np.sinh(2 * r)
    
    # 2. Calculate the c_n prefix
    prefix = (1.0 / np.sqrt(cosh_r)) * np.exp(-0.5 * np.abs(gamma)**2 +  0.5 * (gamma**2) * np.exp(-1j * theta) * tanh_r)
    
    # 3. Use Recurrence for numerical stability of Hermite polynomials with complex args
    # c_n = prefix * (1/sqrt(n!)) * K^n * H_n(z)
    K = np.sqrt(0.5 * np.exp(1j * theta) * tanh_r)
    z = gamma / np.sqrt(np.exp(1j * theta) * sinh_2r) if r > 1e-10 else 0

    f = np.zeros(nMax, dtype=complex)
    if nMax > 0: f[0] = 1.0
    if nMax > 1: f[1] = 2.0 * z * K # H_1(z) = 2z
        
    for n in range(2, nMax):
        # Normalized recurrence: f_n = (K*2z/sqrt(n)) * f_{n-1} - (K^2 * sqrt(4*(n-1)/n)) * f_{n-2}
        f[n] = (2.0 * z * K / np.sqrt(n)) * f[n-1] - (K**2 * np.sqrt(4.0 * (n - 1) / n)) * f[n-2]
        
    return prefix * f

def getInversion(alpha, r, theta, nMax, tVals, lam=1.0):
    c_n = getCnCoefficients(alpha, r, theta, nMax)
    prob_n = np.abs(c_n)**2
    if np.sum(prob_n) > 0: prob_n /= np.sum(prob_n) # Normalization
    
    sqrt_n_plus_1 = np.sqrt(np.arange(nMax) + 1)
    return np.dot(np.cos(2 * lam * tVals[:, None] * sqrt_n_plus_1), prob_n)

# --- Dynamics Study ---
t = np.linspace(0, 80, 1000)
nMax = 100

L = 2
alpha = 5
r = 3
theta = np.pi/4

W_displacedSqueezed = getInversion(alpha, r, theta, nMax, t, L)
W_coherent = getInversion(alpha, 1e-5, theta, nMax, t, L)
W_squeezed = getInversion(1e-5, r, theta, nMax, t, L)

ax1: Axes
ax2: Axes
ax3: Axes

fig, (ax1,ax2,ax3) = plt.subplots(3, 1, layout='constrained', sharex=True) 

fig.set_size_inches(6,8)

ax1.plot(t, W_displacedSqueezed, lw=0.5)
ax1.set_ylabel(r'$W(t)$ for a displaced squeezed state')

ax2.plot(t, W_coherent, lw=0.5)
ax2.set_ylabel(r'$W(t)$ for a coherent state')

ax3.plot(t, W_squeezed, lw=0.5)
ax3.set_ylabel(r'$W(t)$ for a squeezed vacuum state')
ax3.set_xlabel(r'Time')

plt.savefig('./Portfolio/figures/problem7_7.pdf')





# plt.figure(figsize=(10, 12))

# # Case 1: Coherent Limit
# W_coh = getInversion(3.0, 1e-5, 0, nMax, t)
# plt.subplot(3, 1, 1)
# plt.plot(t, W_coh, 'b-')
# plt.title(r"Coherent Limit ($r \to 0, \alpha=3$)")
# plt.ylabel(r"$W(t)$")

# # Case 2: Squeezed Vacuum Limit
# W_sqv = getInversion(1e-5, 1.0, 0, nMax, t)
# plt.subplot(3, 1, 2)
# plt.plot(t, W_sqv, 'r-')
# plt.title(r"Squeezed Vacuum Limit ($\alpha \to 0, r=1.0$")