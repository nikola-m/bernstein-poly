import numpy as np
from scipy.linalg import solve, inv#, eig
from cheb import cheb, cheb_trefethen
from rk import ssprk54
from bernstein import bpoly_cgl_nst, bpoly_cgl
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Asana Math'], 'size':'14'})

# Function to compute Chebyshev differentiation matrix
def chebyshev_differentiation_matrix(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    D = np.zeros((N + 1, N + 1))
    c = np.ones(N + 1)
    c[0] = 2
    c[N] = 2
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = c[i] * (-1)**(i + j) / (c[j] * (x[i] - x[j]))
        if i == 0:
            D[i, i] = (2 * N**2 + 1) / 6
        elif i == N:
            D[i, i] = -(2 * N**2 + 1) / 6
        else:
            D[i, i] = -x[i] / (2 * (1 - x[i]**2))
    return D, x


def chebyshev_solution(t, dt, tfinal, N):

    # Diff mat
    # D,x = cheb(N)

    # Compute first and second Chebyshev differentiation matrices
    D, x = chebyshev_differentiation_matrix(N)

    x = (x + 1)/2
    D = 2*D
    D2 = np.dot(D, D)

    # Parameter for the problem:
    mu=0.02

    # Set initial value of function
    u = np.exp( (np.cos(np.pi*x)-1) / (2*np.pi*mu) )
    plt.plot(x, u, 'b-', label='w-initial')

    # D2[0, :] = D[0, :]            # Neumann BC
    # D2[N, :] = D[N, :]            # Neumann BC

    A = mu * D2

    # Time-stepping loop with explicit Neumann boundary condition handling
    while t < tfinal:
        # u_new = u + dt * mu * np.dot(D2, u); t+=dt
        #  Five stage, forth order SSP Runge-Kutta
        t, u_new = ssprk54(t, dt, A, u)
        # Explicitly enforce Neumann boundary conditions
        # u_new[0] = u_new[1]  # Neumann BC at x = -1
        # u_new[-1] = u_new[-2]  # Neumann BC at x = 1
        u_new[0]   = u_new[0]-np.dot(D[0,:],u_new[:])/D[0,0]
        u_new[N]   = u_new[N]-np.dot(D[N,:],u_new[:])/D[N,N]
        u = u_new

    # Inverse Cole-Hopf transformation
    # uu = -2*mu*(np.dot(D,u)/u)
    uu = -2*mu*np.dot(D,np.log(u))

    # Plot the final solution
    plt.plot(x, u, '--', label='w-final')
    plt.plot(x, uu, 'o-', label='u-final')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Convection-diffusion using Cole-Hopf transformation')
    plt.legend()
    plt.grid()
    plt.show()

    return x, u, uu 
    
def bernstein_solution(t, dt, tfinal, n, nst=True):

    # Diff mat
    if(nst):
        D2,D,B,x=bpoly_cgl_nst(n) 
    else:
        D2,D,B,x=bpoly_cgl(n)

    # Domain transformation
    #D=2*D; D2=4*D2 # [-1,1] => [0,1]
    #x=(x+1)/2      # [-1,1] => [0,1]
    D=2*D            # [-1,1] => [0,1]
    D2=4*D2          # [-1,1] => [0,1]
    x=(x+1.)/2.      # [-1,1] => [0,1]   

    # Parameter for the problem:
    mu=0.02

    # Set initial value of function
    win = np.exp( (np.cos(np.pi*x)-1) / (2*np.pi*mu)) 

    # Set bn from initial solution win 
    bn=solve(B,win)

    # Eliminisi zbog zero Dirichlet BC
    #D2=D2[1:n,1:n]
    #Bi=inv(B[1:n,1:n])
    Bi=inv(B)
    DBi = D @ Bi

    # Modify D2 for Neumann boundary conditions
    #D2[0, :] = D[0, :]
    #D2[n, :] = D[n, :]

    A = mu * (Bi @ D2)

    w = np.zeros(n+1)

    # Do Runge-Kutta integration till final time
    while t < tfinal:        
        #   Forward Euler 1st order:
        # bn = bn + dt * np.dot(A,bn); t += dt
        #  Five stage, forth order SSP Runge-Kutta
        t, bn = ssprk54(t, dt, A, bn)
      
        # Reconstruct the solution w = B*bn
        w = np.dot(B,bn)
        # w[0] = w[1]  # Neumann BC at x = -1
        # w[-1] = w[-2]  # Neumann BC at x = 1
        w[0] = w[0]-np.dot(DBi[0,:],w[:])/DBi[0,0] 
        w[n] = w[n]-np.dot(DBi[n,:],w[:])/DBi[n,n] 
        bn = solve(B,w)


    # Inverse Cole-Hopf transformation 
    u = -2*mu*(np.dot(D,bn) / w)

    ### Plot solutions 
    plt.plot(x, win,'m-')
    plt.plot(x, w,'c*-')
    plt.plot(x, u, 'o-', mfc='none', color='mediumvioletred', label='u_final')
    plt.title('Convection-diffusion using Cole-Hopf transformation')
    #plt.title('rk4, n=%d, dt=%e, max err = %e' % (n,dt,maxerr))
    plt.legend(('initial','final'))
    plt.grid('True')
    plt.axis('tight')
    plt.show()

    return x, w, u
    
    

# Polynomial order of the collocation solution
n = 30  

t = 0.        # start time
dt = 1e-4     # timestep-size
tfinal = 2.0  #t + nsteps*dt # final time

# Solution domain: -> We are going to set this as known in "solution" functions
# a = 0.
# b = 1.


xb, wb, ub = bernstein_solution(t, dt, tfinal, n, nst=False)
xc, wc, uc = chebyshev_solution(t, dt, tfinal, n)

diff = np.abs(uc[::-1]- ub)
print(diff)

plt.figure(figsize=(10, 6))
# plt.plot(xb, ub, 'o-', mfc='none', color='darkslateblue', label='Chebyshev', markersize=10, markeredgewidth=1.5)
# plt.plot(xc, uc, '+-', mfc='none', color='mediumvioletred', label='Bernstein', markersize=15, markeredgewidth=1.5)
plt.plot(xb, diff, 'o-', mfc='none', color='mediumvioletred', label='Difference', markersize=10, markeredgewidth=1.0)
plt.title("Difference of numerical solutions using Cole-Hopf transformation")
plt.xlabel("x")
plt.ylabel("Absolute Difference")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

# orders = [8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# errors_chebyshev = []
# errors_bernstein = []
# errors_bernstein_nst = []

# for n in orders:

#     # Calculate the maximum absolute error Chebyshev
#     error = chebyshev_solution(t, dt, tfinal, n)
#     errors_chebyshev.append(error)

#     # Calculate the maximum absolute error Bernstein
#     error = bernstein_solution(t, dt, tfinal, n)
#     errors_bernstein.append(error)

#     # Calculate the maximum absolute error Bernstein
#     error = bernstein_solution(t, dt, tfinal, n, nst=True)
#     errors_bernstein_nst.append(error)

# # Plot error vs polynomial order
# plt.figure(figsize=(10, 6))
# plt.plot(orders, errors_chebyshev, 'o', mfc='none', color='darkslateblue', label='Chebyshev', markersize=10, markeredgewidth=1.5)
# plt.plot(orders, errors_bernstein, '+', mfc='none', color='mediumvioletred', label='Bernstein', markersize=15, markeredgewidth=1.5)
# plt.plot(orders, errors_bernstein_nst, 'D', mfc='none', color='mediumvioletred', label='Bernstein+Symmetry+NST', markersize=10, markeredgewidth=1.0)
# plt.title("Error for the numerical solution of advection equation")
# plt.xlabel("Polynomial Order")
# plt.ylabel("Maximum Absolute Error")
# plt.yscale('log')
# plt.grid(True)
# plt.legend()
# plt.show()