import numpy as np
from scipy.linalg import solve, inv, eig
from bernstein import bpoly_cgl_nst, interpolant, bpoly_cgl
from cheb import cheb, cheb_trefethen
from rk import ssprk54
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif'})
rc('font',**{'family':'serif','serif':['Asana Math'], 'size':'14'})





def ssprk54_2(t,dt,A,u):
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


def chebyshev_solution(t, dt, tfinal, n):

    # Diff mat
    D,x = cheb(n)

    # Set initial value of function
    u = np.sin(2*x) # Eq. 3.7.13 p.168 in CHQZ1

    # RHS Operator, in this case pure advection!
    A=1.5*D

    # # Plots eigenvals of A
    # lam,V=eig(A)
    # plt.title('n = %d, max($\lambda_{real}$) = %e' % (n,np.max(lam.real)))
    # plt.scatter(lam.real, lam.imag, s=80, facecolors='none', edgecolors='m')
    # # plt.xlim((-32, 32))
    # # plt.ylim((-32, 32))
    # plt.grid('True')
    # plt.show()


    # Do Runge-Kutta integration till final time
    while t < tfinal:
        # Five stage, forth order SSP Runge-Kutta
        t, u = ssprk54_2(t, dt, A, u)
        # Enforce Dirichlet at left for new t
        u[n] = np.sin(-2-3*t)

    # Analytical solution and error
    uex = np.sin(2*x-3*t)
    maxerr = max(abs(u-uex))
    print(f'%d maxerr = %e' % (n,maxerr))

    # xx=np.linspace(-1.,1.,100)
    # # uu = interpolant(beta,n,-1.,1,xx) 
    # uin=np.sin(2*x)

    # ### Plot solutions #####################
    # plt.plot(x, uin, '--',color='darkslateblue', label='initial', linewidth=1)
    # plt.plot(x, u, '-', color='darkslateblue', label='u_n', linewidth=2)
    # plt.plot(x, uex, 'o', mfc='none', color='mediumvioletred', label='u_a', markeredgewidth=1.5)
    # plt.title('ssprk(5,4), dt=%4.3f, n = %d, $||err||_\infty$ = %e' % (dt,n,maxerr))
    # plt.legend(('initial', 'final', 'analytical'))
    # plt.xlabel('x')
    # plt.ylabel('u')
    # plt.grid('True')
    # plt.axis('tight')
    # plt.show()

    return maxerr     

def bernstein_solution(t, dt, tfinal, n, nst=False):

    # Diff mat
    if(nst):
        D2,D,B,x=bpoly_cgl_nst(n) 
    else:
        D2,D,B,x=bpoly_cgl(n)

    # Set initial value of function
    u = np.sin(2*x) # Eq. 3.7.13 p.168 in CHQZ1

    # Coeficients of Bernstein polynomials expansion - beta, of the initial state
    beta=solve(B,u)
    Bi = inv(B)
    # RHS Operator, in this case pure advection!
    A=-1.5*np.dot(Bi,D)

    # # Plots eigenvals of A
    # lam,V=eig(A)
    # plt.title('n = %d, max($\lambda_{real}$) = %e' % (n,np.max(lam.real)))
    # plt.scatter(lam.real, lam.imag, s=80, facecolors='none', edgecolors='m')
    # # plt.xlim((-32, 32))
    # # plt.ylim((-32, 32))
    # plt.grid('True')
    # plt.show()


    # Do Runge-Kutta integration till final time
    while t < tfinal:
        # Five stage, forth order SSP Runge-Kutta
        t, beta = ssprk54_2(t, dt, A, beta)
        # Enforce Dirichlet at left for new t
        u = B @ beta
        u[0] = np.sin(-2-3*t)
        beta = solve(B,u)

    # Get the function values form Bernstein polynomials coefficients u = B.beta
    u = B @ beta

    # Analytical solution and error
    uex = np.sin(2*x-3*t)
    maxerr = max(abs(u-uex))
    print(f'%d maxerr = %e' % (n,maxerr))

    # xx=np.linspace(-1.,1.,100)
    # # uu = interpolant(beta,n,-1.,1,xx) 
    # uin=np.sin(2*x)

    # ### Plot solutions #####################
    # plt.plot(x, uin, '--',color='darkslateblue', label='initial', linewidth=1)
    # plt.plot(x, u, '-', color='darkslateblue', label='u_n', linewidth=2)
    # plt.plot(x, uex, 'o', mfc='none', color='mediumvioletred', label='u_a', markeredgewidth=1.5)
    # plt.title('ssprk(5,4), dt=%4.3f, n = %d, $||err||_\infty$ = %e' % (dt,n,maxerr))
    # plt.legend(('initial', 'final', 'analytical'))
    # plt.xlabel('x')
    # plt.ylabel('u')
    # plt.grid('True')
    # plt.axis('tight')
    # plt.show()

    return maxerr 




# Solution domain:
#a = -1.
#b = 1.

#Transformacija domena i matrica
#D=2*D; D2=4*D2 # [-1,1] => [0,1]
#x=(x+1)/2.      # [-1,1] => [0,1]

#D=0.5*D; D2=0.25*D2 # [-1,1] => [-2,2]
#x=2*(x+1)      # [-1,1] => [-2,2]
#
#D=D*1./(a) #D2=D2*1./(a**2)  # Ovo je zbog transformacije domena [-1,1] => [-a,a] 
#x=x*a # stretch x    


t = 0. # start time
dt = 1e-5 # timestep-size
tfinal = 1.0


orders = [8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]
errors_chebyshev = []
errors_bernstein = []
errors_bernstein_nst = []

for n in orders:

    # Calculate the maximum absolute error Chebyshev
    error = chebyshev_solution(t, dt, tfinal, n)
    errors_chebyshev.append(error)

    # Calculate the maximum absolute error Bernstein
    error = bernstein_solution(t, dt, tfinal, n)
    errors_bernstein.append(error)

    # Calculate the maximum absolute error Bernstein
    error = bernstein_solution(t, dt, tfinal, n, nst=True)
    errors_bernstein_nst.append(error)

# Plot error vs polynomial order
plt.figure(figsize=(10, 6))
plt.plot(orders, errors_chebyshev, 'o', mfc='none', color='darkslateblue', label='Chebyshev', markersize=10, markeredgewidth=1.5)
plt.plot(orders, errors_bernstein, '+', mfc='none', color='mediumvioletred', label='Bernstein', markersize=15, markeredgewidth=1.5)
plt.plot(orders, errors_bernstein_nst, 'D', mfc='none', color='mediumvioletred', label='Bernstein+Symmetry+NST', markersize=10, markeredgewidth=1.0)
plt.title("Error for the numerical solution of advection equation")
plt.xlabel("Polynomial Order")
plt.ylabel("Maximum Absolute Error")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()




