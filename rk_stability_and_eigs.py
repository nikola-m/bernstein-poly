import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, inv, eig
from bernstein import bpoly_cgl_nst, bpoly_cgl, bpoly, interpolant
from cheb import cheb
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif'})
rc('font',**{'family':'serif','serif':['Asana Math'], 'size':'14'})


# Define the stability function R(z)
def stability_function(z):
    return 0.004478*z**5 + 0.04167*z**4 + 0.1667*z**3 + 0.5*z**2 + z + 1

# Generate a grid of complex numbers
real_vals = np.linspace(-10, 5, 400)  # Real axis values
imag_vals = np.linspace(-10, 10, 400)  # Imaginary axis values
X, Y = np.meshgrid(real_vals, imag_vals)
Z = X + 1j*Y

# Compute the magnitude of the stability function
R = stability_function(Z)
abs_R = np.abs(R)

# Plot the absolute stability region
plt.figure(figsize=(8, 8))
plt.contour(X, Y, abs_R, levels=[1])
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.title("Absolute Stability Region of SSP RK 5(4)")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
#plt.show()


c=1.#3.5; 
gama=0.#0.022

n = 20 # order of B-polynomial
D2,D,B,x=bpoly_cgl_nst(n) 

D=D[1:,1:]
D2=D2[1:,1:]


# D=2*D; D2=4*D2 # [-1,1] => [0,1]
# x=(x+1)/2      # [-1,1] => [0,1]
  

Bi=inv(B[1:,1:])

A=gama*np.dot(Bi,D2)-c*np.dot(Bi,D)

# Plots eigenvals of A
lam,V=eig(A)
plt.title('n = %d, c=%g, gama=%g, max($\lambda_{real}$) = %e' % (n, c, gama, np.max(lam.real)))
plt.scatter(0.08*lam.real, 0.08*lam.imag, s=110, facecolors='none', edgecolors='c', label="Bernstein")
#plt.xlim(-700,200)
plt.xlim(-6,1)
plt.ylim(-5,5)
plt.grid('True')


# Chebyshev
# Diff mat
D,x = cheb(n)
D2 = np.dot(D,D)

#D=2*D; D2=4*D2 # [-1,1] => [0,1]
#x=(x+1)/2      # [-1,1] => [0,1]

D=D[1:,1:]
D2=D2[1:,1:]

# RHS Operator
A=gama*D2-c*D

# Plots eigenvals of A
lam,V=eig(A)
#plt.title('n = %d, c=%g, gama=%g, max($\lambda_{real}$) = %e' % (n, c, gama, np.max(lam.real)))
plt.scatter(0.08*lam.real, 0.08*lam.imag, facecolors='none', edgecolors='m', label="Chebyshev")
plt.legend()

plt.show()

