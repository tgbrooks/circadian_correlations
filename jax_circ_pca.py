from __future__ import annotations
import jax
import jax.scipy.optimize
from jax import numpy as jnp

from circ_pca import expm_AATv, low_rank_weights, CircPCA

def expm_AATv(A, v, nterms=15):
    ''' Given A, a lower triangular (rectangular) matrix and a tall matrix v
    compute exp(A - A.T) v
    where B = A - A.T is a large (sparse) square matrix with A on the left
    and A.T on the top, zeros in the bottom left block
    Assumes A and v have the same shape
    Much faster than doing np.scipy.linalg.exp(A - A.T) @ V for tall matrices
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