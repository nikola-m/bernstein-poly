from scipy.special import comb, factorial
import numpy as np

def binomialCoefficient(n, k):
    if k < 0 or k > n:
        return 0
    if k > n - k: # take advantage of symmetry
        k = n - k
    c = 1
    for i in range(k):
        c = c * (n - (k - (i+1)))
        c = c // (i+1)
    return c

def basis_fun_eval(j,n,a,b,x):
    if(j > n or j < 0):
        return 0. # important for derivative evaluation
    else:
#        return binomialCoefficient(n,j)*(x-a)**j*(b-x)**(n-j)/(b-a)**n
        return comb(n,j,exact=1)*(x-a)**j*(b-x)**(n-j)/(b-a)**n



def basis_fun_derivative(k,j,n,a,b,x):
    if (k > n): 
        print('bernstein_basis_fun_derivative::Error: k > n!')
    elif (k == 1):
        return n/(b-a)*( basis_fun_eval(j-1,n-1,a,b,x) - basis_fun_eval(j,n-1,a,b,x) ) 
    else:
        return n/(b-a)*(basis_fun_derivative(k-1,j-1,n-1,a,b,x)-basis_fun_derivative(k-1,j,n-1,a,b,x))

def basis_fun_der(p,i,n,a,b,x):
    ks = max(0,i+p-n)
    ke = min(i,p)
    bmar = 1/(b-a)**p
    suma = 0.
    for k in range(ks,ke+1):
        suma += (-1)**(k+p)*comb(p,k,exact=1)*basis_fun_eval(i-k,n-p,a,b,x)
        #suma += (-1)**(k+p)*binomialCoefficient(p,k)*basis_fun_eval(i-k,n-p,a,b,x)
    return float(factorial(n,exact=1))/float(factorial(n-p,exact=1))*bmar*suma

def interpolant(beta,n,a,b,x):  
    f = 0.
    for j in range(n+1):
        f += beta[j]*basis_fun_eval(j,n,a,b,x)
    return f

### Routines for two-dimensional problems

def interpolant2D(n,m,x1,x2,y1,y2,beta,x,y):
    f = 0.
    for i in range(n+1):
        for j in range(m+1):
            f += beta[i][j]*basis_fun_eval(j,m,y1,y2,y)*basis_fun_eval(i,n,x1,x2,x)
    return f

def bpoly(a,b,n):
    '''Differentiation matrix and (equidistant) nodes
       for Bernstein polynomial interpolation.
       It is made to be reminescent of 
       N.Trefethen's cheb(n) MATLAB function.
    '''
    nd = np.linspace(a,b,n+1)  # (equidistant) nodes
    B = np.zeros((n+1,n+1))    # Basis function matrix
    D = np.zeros((n+1,n+1))     # Differentiation matrix
    D2 = np.zeros((n+1,n+1))    # 2nd order differentiation matrix
    for i in range(0,n+1):
        x = nd[i] 
        for j in range(0,n+1):  
            B[i,j] = basis_fun_eval(j,n,a,b,x)  
            D[i,j] = basis_fun_der(1,j,n,a,b,x)
            D2[i,j] = basis_fun_der(2,j,n,a,b,x)
    return D2,D,B,nd

def bpoly_cgl(n):
    '''Differentiation matrix and (CGL) nodes
       for Bernstein polynomial interpolation.
    '''
    #nd = np.cos(np.pi*np.linspace(n,0,n+1)/n) # Chebyshev Gauss Lobatto nodes
    nd = np.sin(np.pi*(n-2*np.linspace(n,0,n+1))/(2*n))
    B = np.zeros((n+1,n+1))    # Basis function matrix
    D = np.zeros((n+1,n+1))     # Differentiation matrix
    D2 = np.zeros((n+1,n+1))    # 2nd order differentiation matrix
    for i in range(0,n+1):
        x = nd[i] 
        for j in range(0,n+1):   
            B[i,j] = basis_fun_eval(j,n,-1.,1.,x) 
            D[i,j] = basis_fun_der(1,j,n,-1.,1.,x)
            D2[i,j] = basis_fun_der(2,j,n,-1.,1.,x)
    return D2,D,B,nd

def bpoly_cgl_nst(n):
    '''Differentiation matrix and (CGL) nodes
       for Bernstein polynomial interpolation.
       Negative Sum and Flip&Reflect tricks used
       to improve accuracy.
    '''
    #nd = np.cos(np.pi*np.linspace(n,0,n+1)/n) # Chebyshev Gauss Lobatto nodes
    nd = np.sin(np.pi*(n-2*np.linspace(n,0,n+1))/(2*n))
    B = np.zeros((n+1,n+1))     # Basis function matrix
    D = np.zeros((n+1,n+1))     # Differentiation matrix
    D2 = np.zeros((n+1,n+1))    # 2nd order differentiation matrix
    n2 = int(np.ceil((n+1)/2))
    for i in range(0,n2):
        x = nd[i] 
        for j in range(0,n+1):   
            B[i,j] = basis_fun_eval(j,n,-1.,1.,x) 
            D[i,j] = basis_fun_der(1,j,n,-1.,1.,x)
            D2[i,j] = basis_fun_der(2,j,n,-1.,1.,x)
    # Flip & Reflect Trick:
    n1 = (n+1)//2  # Indices used for flipping trick
    B[n1:,:] = np.flipud(np.fliplr(B[0:n2,:]))
    D[n1:,:] = -np.flipud(np.fliplr(D[0:n2,:]))
    D2[n1:,:] = np.flipud(np.fliplr(D2[0:n2,:]))
    # Negative Sum Trick
    #B -= np.eye(n+1) - np.diag(np.sum(B,axis=1)) # Results not better...
    #D[range(n+1),range(n+1)]  -= np.sum(D,axis=1)   #... 
    D -= np.diag(np.sum(D,axis=1))   # Correct main diagonal of D
    D2 -= np.diag(np.sum(D2,axis=1))   
    return D2,D,B,nd

def bpoly_cgl4(n):
    '''Differentiation matrix up to order 4 and CGL nodes
       for Bernstein polynomial interpolation.
    '''
    #nd = np.cos(np.pi*np.linspace(n,0,n+1)/n) # Chebyshev Gauss Lobatto nodes
    nd = np.sin(np.pi*(n-2*np.linspace(n,0,n+1))/(2*n))
    B = np.zeros((n+1,n+1))    # Basis function matrix
    D = np.zeros((n+1,n+1))     # Differentiation matrix
    D2 = np.zeros((n+1,n+1))    # 2nd order differentiation matrix
    D3 = np.zeros((n+1,n+1))
    D4 = np.zeros((n+1,n+1))
    for i in range(0,n+1):
        x = nd[i] 
        for j in range(0,n+1):   
            B[i,j] = basis_fun_eval(j,n,-1.,1.,x) 
            D[i,j] = basis_fun_der(1,j,n,-1.,1.,x)
            D2[i,j] = basis_fun_der(2,j,n,-1.,1.,x)
            D3[i,j] = basis_fun_der(3,j,n,-1.,1.,x)
            D4[i,j] = basis_fun_der(4,j,n,-1.,1.,x)
    return D4,D3,D2,D,B,nd
    
def bpoly_cgl4_sym(n):
    '''Differentiation matrix up to order 4 and CGL nodes
       for Bernstein polynomial interpolation.
    '''
    #nd = np.cos(np.pi*np.linspace(n,0,n+1)/n) # Chebyshev Gauss Lobatto nodes
    nd = np.sin(np.pi*(n-2*np.linspace(n,0,n+1))/(2*n))
    B = np.zeros((n+1,n+1))    # Basis function matrix
    D = np.zeros((n+1,n+1))     # Differentiation matrix
    D2 = np.zeros((n+1,n+1))    # 2nd order differentiation matrix
    D3 = np.zeros((n+1,n+1))
    D4 = np.zeros((n+1,n+1))
    n2 = int(np.ceil((n+1)/2))
    for i in range(0,n2):
        x = nd[i] 
        for j in range(0,n+1):   
            B[i,j] = basis_fun_eval(j,n,-1.,1.,x) 
            D[i,j] = basis_fun_der(1,j,n,-1.,1.,x)
            D2[i,j] = basis_fun_der(2,j,n,-1.,1.,x)
            D3[i,j] = basis_fun_der(3,j,n,-1.,1.,x)
            D4[i,j] = basis_fun_der(4,j,n,-1.,1.,x)
    # Flip & Reflect Trick:
    n1 = (n+1)//2  # Indices used for flipping trick
    B[n1:,:] = np.flipud(np.fliplr(B[0:n2,:]))
    D[n1:,:] = -np.flipud(np.fliplr(D[0:n2,:]))
    D2[n1:,:] = np.flipud(np.fliplr(D2[0:n2,:]))
    D3[n1:,:] = -np.flipud(np.fliplr(D3[0:n2,:]))
    D4[n1:,:] = np.flipud(np.fliplr(D4[0:n2,:]))
    return D4,D3,D2,D,B,nd

def laplace(D2,B):
    '''Laplace differential operator.
       Input:
       D2 - 2nd order differentiation matrix
       B - Bernstein basis function matrix
    '''
    return np.kron(B,D2)+np.kron(D2,B)

def helmholtz(D2,B,lam):
    '''Helmholtz differential operator.
       Input:
       D2 - 2nd order differentiation matrix
       B - Bernstein basis function matrix
       lam - a coefficient
    '''
    n=np.shape(D2)[0]-1
    return  np.kron(B,D2)+np.kron(D2,B)+lam**2*np.eye((n+1)**2) 

def basis_fun_eval_env(n,a,b,x):
    '''Envelope of Bernstein polynomials.
       Reference:
       Mabry. R. -"Problem 10990." Amer. Math. Monthly 110, 59, 2003.
    '''
    xx=(x-a)/(b-a)
    return 1/np.sqrt(2.*np.pi*n*xx*np.sqrt(1.-xx))

