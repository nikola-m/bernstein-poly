import numpy as np
from scipy.linalg import solve
from bernstein import bpoly_cgl_nst
import matplotlib.pyplot as plt

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif'})
rc('font',**{'family':'serif','serif':['Asana Math'], 'size':18})




# Function f(x) and its analytical derivative f'(x)
def f(x):
    return np.sin(8*x) / (x + 1.1)**(3/2)

def f_prime_analytical(x):
    return (8 * np.cos(8*x) * (x + 1.1) - (3/2) * np.sin(8*x)) / (x + 1.1)**(5/2)

# Test the accuracy for different orders
orders = [5, 10, 12, 14, 16, 18, 20, 21, 22, 24, 26, 32, 36, 40]
errors = []

for N in orders:

    #### With symmetry and NST for Bernstein Polynomial Differentiation Matrices: ###
    D2,D,B,x=bpoly_cgl_nst(N)
    # Transformacija
    D=2*D            # [-1,1] => [0,1]
    D2=4*D2          # [-1,1] => [0,1]
    x=(x+1.)/2.      # [-1,1] => [0,1]

    f_values = f(x)
    coefs = solve(B,f_values)
    f_prime_numerical = np.dot(D, coefs)
    f_prime_exact = f_prime_analytical(x)
    
    # Calculate the maximum absolute error
    error = np.max(np.abs(f_prime_numerical - f_prime_exact))
    errors.append(error)

    # Plotting - enable to see indivudial polynomial order match with the analytical solution
    #plt.figure(figsize=(8, 6))
    #plt.plot(x, f_prime_exact, label="Analytical Derivative", color='r')
    #plt.plot(x, f_prime_numerical, label="Numerical Derivative", linestyle='--', color='b')
    #plt.title(f"Bernstein Polynomial Order {N}")
    #plt.xlabel("x")
    #plt.ylabel("f'(x)")
    #plt.legend()
    #plt.grid(True)
    #plt.show()

# Plot error vs polynomial order
plt.figure(figsize=(10, 6))
plt.plot(orders, errors, marker='o', mfc='none', color='mediumvioletred')
plt.title("Error for the numerical derivative of $y = \sin(8 x) / (x + 1.1)^{3/2}$")
plt.xlabel("Bernstein Polynomial Order")
plt.ylabel("Maximum Absolute Error")
plt.yscale('log')
plt.grid(True)
plt.show()
