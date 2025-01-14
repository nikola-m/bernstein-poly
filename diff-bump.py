import numpy as np
from scipy.linalg import solve, inv, eig
from bernstein import bpoly_cgl_nst, interpolant, bpoly_cgl
from rk import ssprk54, rk4
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif'})
rc('font',**{'family':'serif','serif':['Asana Math'], 'size':'14'})



# def ssprk54(t,dt,A,u):
#     """SSPRK(5,4) The five-stage fourth-order SSPRK.
#        SSPRK - Strong Stability Preserving Runge-Kutta
#     """
#     u1 = u + 0.391752226571890*dt*(A @ u)
#     u2 = 0.444370493651235*u + 0.555629506348765*u1+0.368410593050371*dt*(A @ u1)
#     u3 = 0.620101851488403*u + 0.379898148511597*u2+0.251891774271694*dt*(A @ u2)
#     u4 = 0.178079954393132*u + 0.821920045606868*u3+0.544974750228521*dt*(A @ u3)
#     u  = 0.517231671970585*u2 +0.096059710526147*u3 + 0.063692468666290*dt*(A @ u3) \
#                               +0.386708617503269*u4 + 0.226007483236906*dt*(A @ u4)
#     return t+dt,u

# Solution domain:
a = -5.
b = 5.

# Parameters for bernstein-poly solution:
n = 31 # order of B-polynomial

# Diff mat
D2,D,B,x=bpoly_cgl_nst(n)

D2=D2[1:n,1:n]
D2=D2*1./(b**2)  # Ovo je zbog transformacije domena [-1,1] => [a,b] 
x=x*b # stretch x    

t = 0.1 # start time
dt = 1e-3 # timestep-size
tfinal = 1.0 #t + nsteps*dt # final time

# Parameters for the problem:
Dpar=1.0; t0=0.1
# Set initial value of function
u = np.exp(-x**2/(4*Dpar*t0))  

beta=solve(B,u)

Bi=inv(B[1:n,1:n])
A=np.dot(Bi,D2)

xx=np.linspace(a,b,100)
uin = np.exp(-xx**2/(4*Dpar*t0)) 

# Plots eigenvals of A
lam,V=eig(A)
plt.title('n = %d, max($\lambda_{real}$) = %e' % (n,np.max(lam.real)))
plt.scatter(lam.real, lam.imag, s=80, facecolors='none', edgecolors='m')
# plt.xlim((-32, 32))
# plt.ylim((-32, 32))
plt.grid('True')
plt.show()


# Do Runge-Kutta integration till final time
while t < tfinal:
#   Forward Euler 1st order:
#    beta += dt*np.dot(A,beta); t += dt
#   Second order Runge-Kutta:
#        t, beta[1:n] = rk2(t, dt, A, beta[1:n])
#   Forth order Runge-Kutta:
#    t, beta[1:n+1] = rk4(t, dt, A, beta[1:n+1]); #beta[0]=beta[n] #<-periodic BC
#   Five stage, forth order SSP Runge-Kutta
    t, beta[1:n] = ssprk54(t, dt, A, beta[1:n])

uex = np.zeros(n+1)
usol = np.zeros(n+1)

# Get the function values form Bernstein polynomials coefficients u = B.beta
usol = np.dot(B,beta)

# Analytical solution and error
uex = np.sqrt(t0/t)*np.exp(-x**2/(4*Dpar*t))
maxerr = abs(max(usol-uex))
print(f'%d maxerr = %e' % (n,maxerr))

#xx=np.linspace(a,b,100)
uu = interpolant(beta,n,a,b,xx) 



### Plot solutions #####################
plt.plot(xx, uin, '--',color='darkslateblue', label='initial', linewidth=1)
plt.plot(xx, uu, '-', color='darkslateblue', label='u_n', linewidth=2)
plt.plot(x, uex, 'o', mfc='none', color='mediumvioletred', label='u_a', markeredgewidth=1.5)
plt.title('ssprk(5,4), dt=%4.3f, n = %d, $||err||_\infty$ = %e' % (dt,n,maxerr))
plt.legend(('initial', 'final', 'analytical'))
plt.xlabel('x')
plt.ylabel('u')
plt.grid('True')
plt.axis('tight')
plt.show()
