import numpy as np
from scipy.linalg import solve, inv
from bernstein import bpoly_cgl_nst
# from rk import ssprk54, rk4
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif'})
rc('font',**{'family':'serif','serif':['Asana Math']})


# def ssprk54(t,dt,L,u):
#     """SSPRK(5,4) The five-stage fourth-order SSPRK.
#        SSPRK - Strong Stability Preserving Runge-Kutta
#     """
#     u1 = u + 0.391752226571890*dt*(L @ u)
#     u1[0] = u1[0]-np.dot(D[0,:],u1[:])/D[0,0]
#     u2 = 0.444370493651235*u + 0.555629506348765*u1+0.368410593050371*dt*(L @ u1)
#     u2[0] = u2[0]-np.dot(D[0,:],u2[:])/D[0,0]
#     u3 = 0.620101851488403*u + 0.379898148511597*u2+0.251891774271694*dt*(L @ u2)
#     u3[0] = u3[0]-np.dot(D[0,:],u3[:])/D[0,0]
#     u4 = 0.178079954393132*u + 0.821920045606868*u3+0.544974750228521*dt*(L @ u3)
#     u4[0] = u4[0]-np.dot(D[0,:],u4[:])/D[0,0]
#     u  = 0.517231671970585*u2 +0.096059710526147*u3 + 0.063692468666290*dt*(L @ u3) \
#                               +0.386708617503269*u4 + 0.226007483236906*dt*(L @ u4)
#     u[0] = u[0]-np.dot(D[0,:],u[:])/D[0,0]
#     return t+dt,u
    
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
N = 16
D2,D,B,x=bpoly_cgl_nst(N) # Bernstein's differentiation matrix and collocation nodes

print(f'Bernstein, dt: {dt}, N: {N}')

# Transformacija
#D=2*D; D2=4*D2 # [-1,1] => [0,1]
#x=(x+1)/2      # [-1,1] => [0,1]
#D=D*1./(b) #D2=D2*1./(b**2)  # Ovo je zbog trqansformacije domena [-1,1] => [a,b] 
#x=x*b # stretch x
nx = len(x)                   # Number of spatial points

Bi=inv(B)
D2[0, :] = D[0, :]            # Neumann BC
DBi = D @ Bi  

L = (2 * a / l) ** 2 * (Bi @ D2) # Linear operator: beta_t = L(beta)
dL = dt * L                      # Scaling for the finite difference loop

u_n = np.zeros((nx, nt))         # Numerical heat function
b_n = np.zeros(nx)               # Bernstein coefficients for the n-th solution

u_0 = A * np.sin((a ** 2 / l ** 2) * t)  # Boundary condition

# A note on enforcing boundary conditions in Bernstein (or any global modal spectral method):
# Say we have at i-th step an enforced Dirichlet BC on the left u_n[N, i] = u_0[i], then
# we have b_n[:] = solve(B,u_n[:,i])
t1=0
for i in range(nt - 1):

    # Enforce Dirichlet at left
    u_n[N, i] = u_0[i]

    # Enforce Neumann BC at right
    # u_n[0, i] = u_n[1, i]    
    # u_n[0, i]   = u_n[0, i]-np.dot(D[0,:],b_n)/DBi[0,0] 
    u_n[0, i]   = u_n[0, i]-np.dot(DBi[0,:],u_n[:, i])/DBi[0,0]           

    # Recalculate beta coefficients after changing the solution vector values
    # b_n = Bi @ u_n[:,i]
    b_n = solve(B,u_n[:,i])

 
    # Time stepping using the updated b_n
    b_n = b_n + np.dot(dL, b_n)
    # t1, b_n = rk4(t1, dt, L, b_n)
    # t1, b_n = ssprk54(t1, dt, L, b_n)

    
    # Recover the solution vector at i*1 timestep by a matvec
    u_n[:, i + 1] = B @ b_n 
    
    #Enforce Dirichlet at rigth
    # u_n[N, i+1] = u_0[i+1]
    # Enforce Neumann BC at left
    u_n[0, i+1]   = u_n[0, i+1]-np.dot(DBi[0,:],u_n[:, i+1])/DBi[0,0]
    # Recalculate beta coefficients after changing the solution vector values
    # b_n = solve(B,u_n[:,i+1]) 

# u_n[0, nt-1]   = u_n[0, nt-1]-np.dot(DBi[0,:],u_n[:, nt-1])/DBi[0,0]  

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
error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) # Infinity Norm of Abs. Error - Max. Abs. Err
print(f'Bernstein, dt= {dt}, N= {N}, time= {time:.2f}, maxerr= {error_norm:g}')
time = int((nt - 1) / 2 ) # 0.5
error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) # Infinity Norm of Abs. Error - Max. Abs. Err
print(f'Bernstein, dt= {dt}, N= {N}, time= {time:.2f}, maxerr= {error_norm:g}')
time = int((nt - 1) / 2 + (nt - 1) / 4) # 0.75
error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) # Infinity Norm of Abs. Error - Max. Abs. Err
print(f'Bernstein, dt= {dt}, N= {N}, time= {time:.2f}, maxerr= {error_norm:g}')
time = int(nt - 1) # 1
#error = np.abs(u_a[:, time] - u_n[:, time]) # Absolute Error
error_norm = np.linalg.norm((u_a[:, time] - u_n[:, time]), np.inf) # Infinity Norm of Abs. Error - Max. Abs. Err
print(f'Bernstein, dt= {dt}, N= {N}, time= {time:.2f}, maxerr= {error_norm:g}')


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
    plt.plot(x, u_a[:, k[i]], 'o', mfc='none', color='mediumvioletred', label='u_a', markeredgewidth=1.0)
    plt.plot(x, u_n[:, k[i]], '-',color='darkslateblue', label='u_n', linewidth=2)
    # plt.gca().invert_xaxis()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.ylim([u_min-3, u_max+3])
    # plt.axis('tight')
    plt.grid(True)
    plt.title(f'Time t = {t[k[i]]:.2f}')
    plt.legend()

plt.tight_layout()
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

# # Error plot
# plt.plot(x, error, 'o-', color='darkslateblue', label='maxerr', linewidth=1)
# plt.title('Bernstein, N = %d, dt=%g, $||err||_\infty$ = %e' % (N, dt, error_norm))
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('Absolute Error')
# plt.yscale('log')
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

# ax.plot_surface(X, T, U_n, cmap='twilight_shifted', alpha=0.7)
# ax.set_xlabel('Spatial coordinate x')
# ax.set_ylabel('Time t')
# ax.set_zlabel('Temperature u_n')
# ax.set_title('3D Visualization of Numerical Solution u_n (Bernstein Polynomials)')
# ax.view_init(elev=30, azim=240)

# plt.show()

