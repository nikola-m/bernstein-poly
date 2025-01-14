import numpy as np

def cheb_trefethen(N):
    '''Chebushev polynomial differentiation matrix.
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
