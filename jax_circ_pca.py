from __future__ import annotations
from typing import Union
from dataclasses import dataclass
import textwrap
import jax
import jax.scipy.optimize
from jax import numpy as jnp
import scipy.optimize
import scipy.linalg
import numpy

def compute_circ_pca(data, time, r, R = None, verbose=False):
    '''
    Compute the optimal r-dimensional subspace that captures the maximum
    variation in X, which is allowed to vary according to `times` by

    W(t) = exp(A cos(t) + B sin(t) + C) W0

    where W0 is the rank r PCA estimate of weights,
    W(t) is the weightings for the r-dimensional subspace at time t,
    A, B, C are skew-symmetric matrices of rank r,
    (specifically they are zero outside of the first r rows and columns,
    and so are parametrized as just n x r matrices)
    and exp() is the matrix exponential function.

    verbose: if True, print out during iterations
    R: number of dimensions to reduce to (via PCA) prior to performing circ PCA analysis.
        Set this to lower numbers (<100) to speed up large datasets, if they are well-captured
        by a PCA of this dimension.
        If None, then no reduction is performed

    returns: a CircPCA instance
    '''

    data = jnp.array(data)
    time = jnp.array(time)

    assert data.shape[0] == len(time), f"Expected data and time to have the same number of rows, instead had {data.shape[0]} and {len(time)}"
    assert r > 0
    N,k = data.shape

    if k > 100 and R is None:
        print(f"Warning: large number of columns may make this procedure slow. Recommended to use parameter R < 100 to reduce first")

    if 3 * r >= N:
        raise ValueError(f"Requested dimension r={r} is too high for the provided number of observations N={N}. Must have 3r < N")
    if 3 * r >= k:
        raise ValueError(f"Requested dimension r={r} is too high for the provided number of variables k={k}. Must have 3r < k")

    
    if R is not None:
        if 3 * r >= R:
            raise ValueError(f"Requested dimension r={r} is too high for the provided number of reduced variables R={R}. Must have 3r < R")
        if R > k:
            raise ValueError(f"Requested reduction to R={R} dimensions canot be higher than the provided number of variables k={k}.")

        # Reduce to the specific number of variables first before
        # performing Circ PCA
        reduction_weights = low_rank_weights(data, R)
        reduced_data = data @ reduction_weights
    else:
        reduced_data = data

    # Perform the actual circ PCA
    result = jax_circ_pca(reduced_data, time, r, verbose=verbose)

    if R is not None:
        # Correct the result CircPCA object to describe the dimension reduction performed
        result.nvars = k
        result.reduction_weights = reduction_weights

    return result

def low_rank_weights(X, r):
    ''' give best rank r weights to approximate X '''
    u,d,vt = jnp.linalg.svd(X, full_matrices=False)
    return vt[:r,:].T

def expm_AATv(A, v, nterms=15):
    ''' Given A, a lower triangular (rectangular) matrix and a tall matrix v
    compute exp(A - A.T) v
    where B = A - A.T is a large (sparse) square matrix with A on the left
    and A.T on the top, zeros in the bottom left block
    Assumes A and v have the same shape
    Much faster than doing jnp.scipy.linalg.exp(A - A.T) @ V for tall matrices
    due to not having to compute the nxn matrix exponential
    '''
    r = v.shape[1]
    res = v # first term of Taylor expansion
    Bkv = v # B^k v / k!  - initializes at v for k=0
    for k in range(1,nterms):
        top = -A.T @ Bkv + A[:r] @ Bkv[:r]
        bottom = A[r:] @ Bkv[:r]
        Bkv = Bkv.at[:r,:].set(top/k)
        Bkv = Bkv.at[r:, :].set(bottom/k)
        res += Bkv
    return res
expm_AATv = jax.jit(expm_AATv, static_argnums=2)

@jax.jit
def eval(A, B, C, W0, X, times):
    def func(i, ssr):
        x = X[[i],:]
        t = times[i]
        L = expm_AATv(jnp.cos(t) * A + jnp.sin(t) * B + C, W0)
        return ssr + jnp.linalg.norm(x - x @ L @ L.T)**2
    ssr = jax.lax.fori_loop(
        0,
        len(X),
        func,
        init_val = jnp.asarray([0.]),
    )
    return ssr / len(X)

@dataclass
class CircPCA:
    '''
    Represents the results of a "Circadian PCA".
    '''

    # Values used to fit
    data: jnp.ndarray
    time: jnp.ndarray

    r: int # Rank of reduction
    nobs: int # number of observations
    nvars: int # number of variables measured per observation

    # Results
    # PCA of the data
    W0: jnp.ndarray
    # The circadian terms:
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray

    # sum-of-squares of residuals of the
    # projection to the given subspace
    resid_sum_squares: float
    # Initial RSS from project to the fixed
    # PCA subpsace, invariant in time
    PCA_resid_sum_squares: float

    # Convergence info
    niter: int # number of iterations
    RSS_history: jnp.ndarray # list of RSS during the fit, to check for convergence

    # Reduction to lower dimension prior to circ PCA info:
    nvars_reduced: int # number of variables reduced to by PCA before performing circ PCA
    reduction_weights: Union[jnp.ndarry, None] #Weights of the original variables used to perform the reduction

    def weights(self, t):
        ''' return n x r matrix of weights at time t
        
        t in radians'''
        w = expm_AATv(self.A * jnp.cos(t) + self.B * jnp.sin(t) + self.C, self.W0)
        if self.reduction_weights is not None:
            return self.reduction_weights @ w
        return w

    def angles(self, ts = jnp.linspace(0,2*jnp.pi, 31), from_mat=None):
        ''' Compute the angles between the r-dimensional subspace at time t
        compared to that at time 0.

        from_mat: matrix to obtain angles with respect to. If None (default),
                    then this will be the t=0 matrix
        '''
        weights = [self.weights(t) for t in ts]
        if from_mat is None:
            from_mat = weights[0]
        angles = jnp.array([scipy.linalg.subspace_angles(Wt, from_mat)
                                for Wt in weights])
        return angles, ts

    def summary(self):
        angles, ts = self.angles()
        angles_from_W0, ts = self.angles(self.W0)
        ''' Summarize the results of the structure '''
        return textwrap.dedent(f'''
        Circadian PCA results:
        W(t) = exp(A cos(t) + B sin(t) + C) W0

        nobs: {self.data.shape[0]}  nvars: {self.nvars} reduced to nvars: {self.nvars_reduced}
        rank r: {self.r}

        niter: {self.niter}
             PCA RSS: {self.PCA_resid_sum_squares:0.3f}
        CIRC PCA RSS: {self.resid_sum_squares:0.3f}

        L-infinity NORMS:
        |A| = {jnp.linalg.norm(self.A, ord=float('inf'))**2:0.3f}\t|B| = {jnp.linalg.norm(self.B, ord=float('inf'))**2:0.3f}\t|C| = {jnp.linalg.norm(self.C, ord=float('inf'))**2:0.3f}
        LARGEST ANGLE FROM t=0: {jnp.max(angles):0.3f}
        LARGEST ANGLE FROM W0: {jnp.max(angles_from_W0):0.3f}
        ''').strip()

def jax_circ_pca(X, times, r, verbose=False):
    """
    Compute the optimal r-dimensional subspace that captures the maximum
    variation in X, which is allowed to vary according to `times` by

    W(t) = exp(A cos(t) + B sin(t) + C) W0

    where W0 is the rank r PCA estimate of weights,
    W(t) is the weightings for the r-dimensional subspace at time t,
    A, B, C are skew-symmetric matrices of rank r,
    (specifically they are zero outside of the first r rows and columns,
    and so are parametrized as just n x r matrices)
    and exp() is the matrix exponential function.

    Returns W0, A, B, C.
    To determine W(t), use expm_AATv(A*cos(t) + B*sin(t) + C, W0).
    To determine the projection of data X onto the dimension r subpsace,
    perform X @ W(t) @ W(t).T
    """

    # By-hand optimizer for the parametrized PCA
    def extract(mat, k, r):
        # Return the lower triangular version
        #idxs = jnp.tril_indices(n=k, k=-1, m=r)
        #return jnp.zeros((k,r)).at[idxs].set(mat)
        return jnp.concatenate([jnp.zeros((r,r)), mat.reshape((k-r,r))])
    n,k = X.shape
    times = jnp.asarray(times)
    W0 = jnp.asarray(low_rank_weights(X, r))
    X = jnp.asarray(X)
    #N = k*r - (r*(r+1)//2) # Num free vars per matrix
    N = k*r - r*r
    def f(A,B,C):
        Atri = extract(A, k, r)
        Btri = extract(B, k, r)
        Ctri = extract(C, k, r)
        return eval(Atri,Btri,Ctri, W0, X, times)[0]
    val_and_grad = jax.value_and_grad(f, argnums=[0,1,2])
    beta1 = 0.9
    beta2 = 0.99
    alpha = 0.001 #Learning rate
    epsilon = 1e-8
    @jax.jit
    def update(A,B,C, m, v, i):
        # Adam optimizer
        residual, (gradA, gradB, gradC) = val_and_grad(A,B,C)
        mA, mB, mC = m
        m = (
            beta1 * mA + (1 - beta1) * gradA,
            beta1 * mB + (1 - beta1) * gradB,
            beta1 * mC + (1 - beta1) * gradC,
        )
        vA, vB, vC = v
        v = (
            beta2 * vA + (1 - beta2) * gradA**2,
            beta2 * vB + (1 - beta2) * gradB**2,
            beta2 * vC + (1 - beta2) * gradC**2,
        )
        mHat = [mX / (1 - beta1**i) for mX in m]
        vHat = [vX / (1 - beta2**i) for vX in v]
        A = A - alpha * mHat[0] / jnp.sqrt(vHat[0] + epsilon)
        B = B - alpha * mHat[1] / jnp.sqrt(vHat[1] + epsilon)
        C = C - alpha * mHat[2] / jnp.sqrt(vHat[2] + epsilon)
        return A, B, C, m, v, residual
        #return A - alpha * gradA, B - alpha * gradB, C - alpha * gradC, v
    A,B,C = jnp.zeros(N), jnp.zeros(N), jnp.zeros(N)
    m = [jnp.zeros(N) for i in range(3)]
    v = [jnp.zeros(N) for i in range(3)]
    PCA_resid_sum_squares = val_and_grad(A, B, C)[0]
    resids = []
    for i in range(1500):
        A,B,C,m, v, res = update(A,B,C, m ,v, i+1)
        resids.append(res)
        if (i % 20) == 0 and verbose:
            print(f"{i}, RSS = {float(res):0.4f}")
            print(f"\t|A| = {jnp.linalg.norm(A, float('inf'))**2:0.3f}\t|B| = {jnp.linalg.norm(B, float('inf'))**2:0.3f}\t|C| = {jnp.linalg.norm(C, float('inf'))**2:0.3f}")
    Atri = extract(A, k, r)
    Btri = extract(B, k, r)
    Ctri = extract(C, k, r)
    result = CircPCA(
        X, times,
        r, n,
        k,
        W0, Atri, Btri, Ctri,
        resid_sum_squares = res,
        PCA_resid_sum_squares = PCA_resid_sum_squares,
        niter = i+1,
        RSS_history = jnp.array(resids),
        nvars_reduced = k,
        reduction_weights = None,
    )
    return result

if __name__ == '__main__':
    ## EXAMPLE DATA
    numpy.random.seed(0)
    N = 500
    t = numpy.linspace(0,2 * numpy.pi, N)
    scores = numpy.random.normal(size=N)
    simd = numpy.concatenate([
        [numpy.cos(t/2) * scores*5 + scores * 10],
        [numpy.sin(t/2) * scores*5 + scores * 10],
        numpy.random.normal(size=(1,N))*5,
        numpy.random.normal(size=(100,N)),
    ], axis=0).T

    result = compute_circ_pca(simd, t, 2, R = 10, verbose=True)
    print(result.summary())