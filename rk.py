import numpy as np

def rhs(A,u):
    """  The righ-hand side linear operator A acting on 
         expansion coefficients 'beta' (for Bernstein polynomials),
         or collocation values of function (for Chebyshev polynomials).
         Both are here represented by 'u'.
    """ 
    rhs = np.dot(A,u)

    return rhs

def ssprk54(t,dt,A,u):
    """SSPRK(5,4) The five-stage fourth-order SSPRK.
       SSPRK - Strong Stability Preserving Runge-Kutta
    """
    u1 = u + 0.391752226571890*dt*rhs(A,u)
    u2 = 0.444370493651235*u + 0.555629506348765*u1+0.368410593050371*dt*rhs(A,u1)
    u3 = 0.620101851488403*u + 0.379898148511597*u2+0.251891774271694*dt*rhs(A,u2)
    u4 = 0.178079954393132*u + 0.821920045606868*u3+0.544974750228521*dt*rhs(A,u3)
    u  = 0.517231671970585*u2 +0.096059710526147*u3 + 0.063692468666290*dt*rhs(A,u3) \
                              +0.386708617503269*u4 + 0.226007483236906*dt*rhs(A,u4)
    return t+dt,u


def rk2(t, dt, A, u):
    """ 2nd Order Runge-Kutta
        x - current time, like t
        h - like dt, a timestep
        u - initial value, here represented by beta coeff.
        f - derivative function u' = f(t, u)
    """
    up = rhs(A,u)

    u1 = u + 0.5*up         # odmah pripremam argument za sledeci korak, vektorka operacija

    up = rhs(A,u1)

    u = u + up*dt           # vektorska operacija 

    return t+dt, u

def rk4(t, dt, A, u):
    """ Forth order Runge-Kutta
        x - current time, like t
        h - like dt, a timestep
        u - initial value, here represented by beta coeff.
        f - derivative function u' = f(t, u)
    """

    k1 = dt * rhs(A,u)

    arg = u + 0.5*k1
    k2 = dt * rhs(A,arg)

    arg = u + 0.5*k2
    k3 = dt * rhs(A,arg)

    arg = u + 0.5*k3
    k4 = dt * rhs(A,arg)

    u = u + (k1 + 2*(k2 + k3) + k4)/6.0

    return t+dt, u

def rkf45(t, dt, y, nd, n, a, b, tolerance, dtmax):
    """Runge-Kutta-Fehlberg method of order 4(5).
    
    t = Current time.
    dt = Timestep.
    y = Initial value.
    f = Derivative function y' = f(t, y).
    tol = Error tolerance.
    
    Returns a (tf, dtf, yf) tuple after time dt has passed, where:
    tf = Final time.
    yf = Final value.
    dtf = Error-corrected timestep for next step.

    dtmax je ustvari tfinal jer integracije pocinje od t=0. 
    Ne zelimo da premasi finalni trenutak pa zato ogranicavamo dt.
    """
    
    min_scale_factor = 0.125
    max_scale_factor = 4.0
    arg = zeros(n+1)

    dt = min(dt, dtmax-t)
    
    # Calculate the slopes at various points.
    # Values taken from http://en.wikipedia.org/wiki/Runge-Kutta-Fehlberg_method
    k1 = dt * rhs(t,y,nd,n,a,b)

    arg =  y + k1*(1/4.0)*dt
    k2 = rhs(t+(1/4.0)*dt,arg,nd,n,a,b)

    arg = y + k1*(3/32.0)*dt + k2*(9/32.0)*dt
    k3 = rhs(t+(3/8.0)*dt,arg,nd,n,a,b)

    arg = y + k1*(1932/2197.0)*dt + k2*(-7200/2197.0)*dt + k3*(7296/2197.0)*dt  
    k4 = rhs(t+(12/13.0)*dt,arg,nd,n,a,b)

    arg = y + k1*(439/216.0)*dt + k2*(-8)*dt + k3*(3680/513.0)*dt + k4*(-845/4104.0)*dt
    k5 = rhs(t+dt,arg,nd,n,a,b)

    arg = y + k1*(-8/27.0)*dt + k2*(2)*dt + k3*(-3544/2565.0)*dt + k4*(1859/4104.0)*dt + k5*(-11/40.0)*dt
    k6 = rhs(t+(1/2.0)*dt,arg,nd,n,a,b)
    
    # 4th order approximation of the final value.
    yf4 = y + k1*(25/216.0)*dt + k3*(1408/2565.0)*dt + k4*(2197/4104.0)*dt + k5*(-1/5.0)*dt
    
    # 5th order approximation of the final value.
    yf5 = y + k1*(16/135.0)*dt + k3*(6656/12825.0)*dt + k4*(28561/56430.0)*dt + k5*(-9/50.0)*dt + k6*(2/55.0)*dt
    
    # Timestep scaling factor. From http://math.fullerton.edu/mathews/n2003/RungeKuttaFehlbergMod.html
    err = sqrt( dot(yf5-yf4,yf5-yf4) )
    scale = 1 if err==0 else (tolerance*dt/(2*err))**(1/4.0)

#   The scale factor is further constrained
    scale = min( max(scale,min_scale_factor), max_scale_factor)
    
    return t+dt, scale*dt, yf4

def dopri5(t, dt, y, nd, n, a, b, tolerance, dtmax):
    """Dormand-Prince adaptive Runge-Kutta method for    
       approximating the solution of the differential equation y'(x) = f(x,y) 
       Dormand-Prince is an embedded 5(4) order method
       with initial condition y(x0) = y.  This implementation evaluates       
       f(x,y) six times per step using embedded fourth order and fifth order  
       Runge-Kutta estimates to estimate not only the solution but also   
       the error.
       dtmax = tfinal-t0, fed in from the main program, if t0=0, dtmax = tfinal
    """
 
#   dtmax je ustvari tfinal jer integracije pocinje od t=0. Ne zelimo da premasi finalni trenutak pa zato ogranicavamo dt   
    dt = min(dt, dtmax - t)

    min_scale_factor = 0.125
    max_scale_factor = 4.0
    
    arg = zeros(n+1)

    k1 = rhs(t, y, nd, n, a, b) 
   
    arg = y + k1 * 0.2 * dt                                                
    k2 = rhs( t + 0.2*dt, arg, nd, n, a, b ) 

    arg =  y + dt * ( 0.075 * k1 + 0.225 * k2)                                   
    k3 = rhs( t + 0.3*dt, arg, nd, n, a, b )    

    arg =  y + dt * ( 0.3 * k1 - 0.9 * k2 + 1.2 * k3)                 
    k4 = rhs( t + 0.6*dt, arg, nd, n, a, b )  

    arg = y + 1.0 / 729.0 * dt * ( 226.0 * k1 - 675.0 * k2 + 880.0 * k3 + 55.0 * k4)           
    k5 = rhs( t+2/3.*dt, arg, nd, n, a, b ) 

    arg =  y + 1.0 / 2970.0 * dt * ( - 1991.0 * k1 + 7425.0 * k2 - 2660.0 * k3 - 10010.0 * k4 + 10206.0 * k5)  
    k6 = rhs( t+dt, arg, nd, n, a, b )  

    y = y +  dt * (31./540.*k1 + 190./297.*k3 - 145./108.*k4 + 351./220.*k5 + 1./20.*k6 )

#   The error is estimated to be:
    err = dt * ( 77.*k1 - 400.*k3 + 1925.*k4 - 1701.*k5 + 99.*k6 ) / 2520.
#    err = max(dt * ( 77.*k1 - 400.*k3 + 1925.*k4 - 1701.*k5 + 99.*k6 ) / 2520.)       

#   The step size h is then scaled by the scale factor    
#    errnrm = sqrt(dot(err,err))                 
#    scale =  sqrt( dot(  0.8*(tolerance/(errnrm * dtmax)*y)**0.25 ,  0.8*(tolerance/(errnrm * dtmax)*y)**0.25 )) 

#   Drugi nacin, neki rad i Wikipedia
    errnrm = sqrt( dot(err,err) ) 
    scale = (tolerance*dt/(2*errnrm))**0.2

#    errnrm = abs(err)  
#    scale = (tolerance*dt/(2*errnrm))**0.2
                     
#   The scale factor is further constrained
    scale = min( max(scale,min_scale_factor), max_scale_factor)

    return t+dt, scale*dt, y, errnrm
