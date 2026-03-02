import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Use LaTeX and serif font
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
plt.rcParams['text.usetex'] = True


def C(theta, phi):
    return np.cos(2*(theta - phi))

def S(theta, phi, thetaPrime, phiPrime):
    return C(theta, phi) + C(theta, phiPrime) + C(thetaPrime, phi) - C(thetaPrime, phiPrime)


thetaPrime = np.pi/4
phiPrime = -np.pi/8

thetaList = np.linspace(0,2*np.pi, 200)
phiList = np.linspace(0, 2*np.pi, 200)

X, Y = np.meshgrid(thetaList, phiList)
Z = S(X, Y, thetaPrime, phiPrime)

fig, ax = plt.subplots()

cPlot = ax.contourf(X, Y, Z, levels=100, cmap='Blues')
cPlot.set_rasterized(True)
fig.colorbar(cPlot)

levels = [-2, 2]
fmt = {-2: r'$|S|=2$', 2: r'$|S|=2$'}
CS = ax.contour(X, Y, Z, levels=levels, colors='red', linewidths=3, zorder=4, linestyles='solid')
ax.clabel(CS, fmt=fmt, inline=True, fontsize=12, colors='red', zorder=5)


positions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
labels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r"$\pi$", r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$']
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_yticks(positions)
ax.set_yticklabels(labels)

ax.set_aspect('equal')
ax.set_xlabel(r'$\vartheta$')
ax.set_ylabel(r'$\varphi$')

plt.savefig('./Portfolio/figures/problem9_3.pdf')