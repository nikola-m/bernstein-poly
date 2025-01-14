import numpy as np
from scipy.linalg import solve, inv, eig
from bernstein import bpoly_cgl_nst, interpolant, bpoly_cgl
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif'})
rc('font',**{'family':'serif','serif':['Asana Math'], 'size':'14'})


#def rhs(A,u):
#    """  Linear operator acting on expansion coefficients 'beta'
#    """ 
#    rhs = np.dot(A,u)
#    #beta[0] = np.exp(bet*t); beta[n] = np.exp(alph+bet*t)
#    #bcvec = np.dot(Bi,dt*beta[0]*np.dot(gama*D-c*Et,d1) + dt*beta[n]*np.dot(gama*D-c*Et,d2))
#    return rhs

def ssprk54(t,dt,A,u):
    """SSPRK(5,4) The five-stage fourth-order SSPRK.
       SSPRK - Strong Stability Preserving Runge-Kutta
    """
    u1 = u + 0.391752226571890*dt*(A @ u)
    u2 = 0.444370493651235*u + 0.555629506348765*u1+0.368410593050371*dt*(A @ u1)
    u3 = 0.620101851488403*u + 0.379898148511597*u2+0.251891774271694*dt*(A @ u2)
    u4 = 0.178079954393132*u + 0.821920045606868*u3+0.544974750228521*dt*(A @ u3)
    u  = 0.517231671970585*u2 +0.096059710526147*u3 + 0.063692468666290*dt*(A @ u3) \
                              +0.386708617503269*u4 + 0.226007483236906*dt*(A @ u4)
    return t+dt,u

# Solution domain:
a = 0.
b = 1.

# Parameters for bernstein-poly solution:
n = 33 #33 # order of B-polynomial

# Diff mat
D2,D,B,x=bpoly_cgl_nst(n) 

#Transformacija doena i matrica
D=2*D; D2=4*D2 # [-1,1] => [0,1]
x=(x+1)/2.      # [-1,1] => [0,1]
#
#D=0.5*D; D2=0.25*D2 # [-1,1] => [-2,2]
#x=2*(x+1)      # [-1,1] => [-2,2]
#
#D=D*1./(b) #D2=D2*1./(b**2)  # Ovo je zbog trqansformacije domena [-1,1] => [a,b] 
#x=x*b # stretch x    

t = -1.0 # start time
dt = 1e-2 # timestep-size
tfinal = 0.2

# t = 0. # start time
# dt = 1e-3 # timestep-size
# tfinal = 1.0 #0.5

# Set initial value of function
#u = np.exp(-np.log(2)*(x+1)**2/sigma**2)
# u =np.exp(-50*x**2) # Gaussian pulse - smooth initial cond.
#u=np.zeros(n+1); u[n//3:2*n//3+1]=1. # square wave
#u=np.zeros(n+1); u[0:n/2+1]=1. # step wave
u = (np.cos(np.pi/2*x*np.exp(-t)))**2 #(np.cos(np.pi/2*x))**2

# Coeficients of Bernstein polynomials expansion - beta, of the initial state
beta=solve(B,u)

Bi=inv(B)
#BiD2=np.dot(BI,D2)  # Sada samo za advection radimo sa D

# Interpolate it to smoother grid using Bernstein intepolant
xx=np.linspace(a,b,100)
uin = interpolant(beta,n,a,b,xx)

# RHS Operator, in this case pure advection!
# A = -np.dot(Bi,D)

# Za problem u_t = -x*u_x
xm = np.tile(x,(n+1,1)).T
A = -np.dot(Bi,np.multiply(xm,D))

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
    t, beta = ssprk54(t, dt, A, beta)
#   Runge-Kutta-Fehlberg of order 4(5):
#        t, dt, u = rkf45(t, dt, u, nd, n, a, b, 1e-5, tfinal)
#   Dormand-Prince embedded Runge-Kutta (ode45 solver in MATLAB):
#        t, dt, u, err = dopri5(t, dt, u, nd, n, a, b, 1e-6, tfinal)
#        print 'New time step size: ', dt, ' ...current time: ', t, '...error: ',err
#    print 'Final time step size: ', dt, ' ...final time: ', t 

# Initialize vectors for the exact and numerical solution
#uex = np.zeros(n+1)
# usol = np.zeros(n+1)

# Get the function values form Bernstein polynomials coefficients u = B.beta
usol = np.dot(B,beta)

uex = (np.cos(np.pi/2*x*np.exp(-t)))**2
maxerr = max(abs(usol-uex))
print(f'%d maxerr = %e' % (n,maxerr))
#print("maxerr(n=33, without NST) = 1.517462e-08")


#xx=np.linspace(a,b,100)
#uu = interpolant(beta,n,a,b,xx)

### Plot solutions #####################
plt.plot(xx, uin, '--',color='darkslateblue', label='initial', linewidth=1)
plt.plot(x, usol, '-', color='darkslateblue', label='u_n', linewidth=2)
plt.plot(x, uex, 'o', mfc='none', color='mediumvioletred', label='u_a', markeredgewidth=1.5)
plt.title('ssprk(5,4), dt=%4.3f, n = %d, $||err||_\infty$ = %e' % (dt,n,maxerr))
plt.legend(('initial', 'final', 'analytical'))
plt.xlabel('x')
plt.ylabel('u')
plt.grid('True')
plt.axis('tight')
plt.show()


