import numpy as np
# from rk import ssprk54
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Source:
# https://scicomp.stackexchange.com/questions/43544/where-am-i-making-a-mistake-in-solving-the-heat-equation-using-the-spectral-meth/43546#43546

def ssprk54(t,dt,L,u):
    """SSPRK(5,4) The five-stage fourth-order SSPRK.
       SSPRK - Strong Stability Preserving Runge-Kutta
    """
    u1 = u + 0.391752226571890*dt*(L @ u)
    u1[0] = u1[0]-np.dot(D[0,:],u1[:])/D[0,0]
    u2 = 0.444370493651235*u + 0.555629506348765*u1+0.368410593050371*dt*(L @ u1)
    u2[0] = u2[0]-np.dot(D[0,:],u2[:])/D[0,0]
    u3 = 0.620101851488403*u + 0.379898148511597*u2+0.251891774271694*dt*(L @ u2)
    u3[0] = u3[0]-np.dot(D[0,:],u3[:])/D[0,0]
    u4 = 0.178079954393132*u + 0.821920045606868*u3+0.544974750228521*dt*(L @ u3)
    u4[0] = u4[0]-np.dot(D[0,:],u4[:])/D[0,0]
    u  = 0.517231671970585*u2 +0.096059710526147*u3 + 0.063692468666290*dt*(L @ u3) \
                              +0.386708617503269*u4 + 0.226007483236906*dt*(L @ u4)
    u[0] = u[0]-np.dot(D[0,:],u[:])/D[0,0]
    return t+dt,u

# def rhs(A,u):
#    """  Linear operator acting on RHS vector and applying BC
#    """ 
#    rhs = np.dot(A,u)
#    # Enforce Neumann BC                 
#    u_n[N, i]   = u_n[N, i]-np.dot(D[N,:],u_n[:,i])/D[N,N]
#    return rhs

# Function to compute Chebyshev differentiation matrix and grid points
def cheb(N):
    if N == 0:
        D = 0
        x = 1
    else:
        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.array([2] + [1] * (N - 1) + [2]) * (-1) ** np.arange(N + 1)
        X = np.tile(x, (N + 1, 1))
        dX = X - X.T
        D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D, axis=1))
    return D, x

# # Chebyshev differentiation matrix and grid
# def chebyshev_differentiation_matrix(N):
#     x = np.cos(np.pi * np.arange(N + 1) / N)
#     c = np.ones(N + 1)
#     c[0] = 2
#     c[N] = 2
#     c = c * (-1) ** np.arange(N + 1)
#     X = np.tile(x, (N + 1, 1))
#     dX = X - X.T
#     D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))  # Chebyshev differentiation matrix
#     D -= np.diag(np.sum(D, axis=1))  # Enforce correct diagonal terms
#     return D, x

def chebyshev_differentiation_matrix(N):
    '''Chebyshev polynomial differentiation matrix.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    '''
    x = np.cos(np.pi*np.linspace(N,0,N+1)/N)
    c=np.zeros(N+1)
    c[0]=2.
    c[1:N]=1.
    c[N]=2.
    c = c * (-1)**np.linspace(0,N,N+1)
    X = np.tile(x, (N+1,1))
    dX = X - X.T
    D = np.dot(c.reshape(N+1,1),(1./c).reshape(1,N+1))
    D = D / (dX+np.eye(N+1))
    D = D - np.diag( D.T.sum(axis=0) )
    return D,x

# Simulation time parameters
nt = 200001                  # Number of time points
t_max = 1                     # Simulation time
t = np.linspace(0, t_max, nt)
dt = t[1] - t[0]              # Time step

# Constants
a = 1                         # Heat constant
A = 100                       # Amplitude of the boundary condition signal
l = a / np.sqrt(np.pi)        # Rod length

# Numerical solution
N = 8
# D, x = cheb(N)                # D is the Chebyshev's differentiation matrix
D, x = chebyshev_differentiation_matrix(N)
nx = len(x)                   # Number of spatial points

print(f'Chebyshev, dt: {dt}, N: {N}')

D2 = np.dot(D, D)
D2[0, :] = D[0, :]            # Neumann BC

L = (2 * a / l) ** 2 * D2     # Linear operator: u_t = L(u)
dL = dt * L                   # Scaling for the finite difference loop

u_n = np.zeros((nx, nt))      # Numerical heat function

u_0 = A * np.sin((a ** 2 / l ** 2) * t)  # Boundary condition

t1 = 0.
for i in range(nt - 1):

    # Enforce Dirichlet BC
    u_n[N, i] = u_0[i]
    # Enforce Neumann BC                 
    # u_n[0, i]   = u_n[1, i]
    # Enforce Neumann BC                 
    u_n[0, i]   = u_n[0, i]-np.dot(D[0,:],u_n[:,i])/D[0,0]

    # Solve
    u_n[:, i + 1] = u_n[:, i] + np.dot(dL, u_n[:, i])
    # t1, u_n[:, i + 1] = ssprk54(t1, dt, L, u_n[:, i])

    # Dirichlet
    # u_n[N, i+1] = u_0[i+1]
    # Enforce Neumann BC                 
    u_n[0, i+1]   = u_n[0, i+1]-np.dot(D[0,:],u_n[:,i+1])/D[0,0]    

# Analytical solution
u_a = np.zeros((nx, nt))      # Analytical heat function

for n in range(101):
    w_n = (0.5 + n) * np.pi
    u_a += (np.sin(w_n * 0.5 * (1 - x)).reshape(-1, 1) * 
           (4 * (np.sin(0.5 * w_n)) ** 2 / (w_n + w_n ** 5)) *
           (w_n ** 2 * (np.exp(-(w_n ** 2) * (a ** 2 / l ** 2) * t) - 
           np.cos((a ** 2 / l ** 2) * t)) - np.sin((a ** 2 / l ** 2) * t)))

u_a = A * np.ones((nx, 1)) * np.sin((a ** 2 / l ** 2) * t) + A * u_a

u_a[N, :] = A * np.sin((a ** 2 / l ** 2) * t)

# Absolute error
error = np.zeros(nx)
time = int((nt - 1) / 4) # 0.25
error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) 
print(f'Chebyshev, dt= {dt}, N= {N}, time= {t[time]:.2f}, maxerr= {error_norm:g}')
time = int((nt - 1) / 2 ) # 0.5
error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) 
print(f'Chebyshev, dt= {dt}, N= {N}, time= {t[time]:.2f}, maxerr= {error_norm:g}')
time = int((nt - 1) / 2 + (nt - 1) / 4) # 0.75
error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) 
print(f'Chebyshev, dt= {dt}, N= {N}, time= {t[time]:.2f}, maxerr= {error_norm:g}')
time = int(nt - 1) # 1
#error = np.abs(u_a[:, time] - u_n[:, time]) # Absolute Error
error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) 
print(f'Chebyshev, dt= {dt}, N= {N}, time= {t[time]:.2f}, maxerr= {error_norm:g}')

# Limits
u_a_max = np.max(u_a)
u_a_min = np.min(u_a)

u_n_max = np.max(u_n)
u_n_min = np.min(u_n)

u_max = max(u_a_max, u_n_max)
u_min = min(u_a_min, u_n_min)

# Graphs
plt.figure('Heat function')

k = [int((nt - 1) / 4), int((nt - 1) / 2), int((nt - 1) / 2 + (nt - 1) / 4), int(nt - 1)]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(x, u_a[:, k[i]], 'ro', label='u_a', linewidth=3)
    plt.plot(x, u_n[:, k[i]], 'b', label='u_n', linewidth=2)
    # plt.gca().invert_xaxis()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.ylim([u_min, u_max])
    plt.grid(True)
    plt.title(f't : {t[k[i]]:.2f}')
    plt.legend()

plt.show()


for i in range(4):
    plt.subplot(2, 2, i + 1)
    time = k[i]
    error = np.abs(u_a[:, time] - u_n[:, time]) # Absolute Error
    # error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) # Infinity Norm of Abs. Error - Max. Abs. Err
    plt.plot(x, error, 'o-', mfc='none', color='darkslateblue', linewidth=1)
    plt.title(f'Time t = {t[k[i]]:.2f}')
    plt.xlabel('x')
    plt.ylabel('Abs. Error')
    plt.yscale('log')
    plt.grid('True')
    plt.axis('tight')

plt.tight_layout()
plt.show()

# # Max abs error plot
# plt.plot(x, error, 'o-', color='darkslateblue', label='maxerr', linewidth=1)
# plt.title('ssprk(5,4), dt=%g, n = %d, $||err||_\infty$ = %e' % (dt,N,error_norm))
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('maxerr')
# plt.grid('True')
# plt.axis('tight')
# plt.show()

# # Visualization in 3D
# # Sample time at regular intervals of 0.1
# time_samples = np.arange(0, t_max + 0.1, 0.1)
# time_indices = (time_samples * (nt - 1) / t_max).astype(int)

# X, T = np.meshgrid(x, time_samples)
# U_n = u_n[:, time_indices].T  # Corresponding solution values

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, T, U_n, cmap='viridis')
# ax.set_xlabel('Spatial coordinate x')
# ax.set_ylabel('Time t')
# ax.set_zlabel('Temperature u_n')
# ax.set_title('3D Visualization of Numerical Solution u_n')
# ax.view_init(elev=30, azim=240)

# plt.show()

