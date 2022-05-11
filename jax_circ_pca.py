import jax
import jax.scipy.optimize
from jax import numpy as jnp
import scipy.optimize
import numpy

numpy.random.seed(0)
N = 500
t = numpy.linspace(0,2 * numpy.pi, N)
scores = numpy.random.normal(size=N)
simd = numpy.concatenate([
    [numpy.cos(t/2) * scores*15],
    [numpy.sin(t/2) * scores*15],
    numpy.random.normal(size=(1,N))*5,
    numpy.random.normal(size=(20,N)),
], axis=0).T

def low_rank_loadings(X, r):
    ''' give best rank r loadings to approximation X '''
    u,d,vt = jnp.linalg.svd(X, full_matrices=False)
    #return u[:,:r] @ jnp.diag(d[:r]) @ vt[:r,:]
    return vt[:r,:].T

def expm_AATv(A, v, nterms=15):
    ''' Given A, a lower triangular (rectangular) matrix and a tall matrix v
    compute exp(A - A.T) v
    where B = A - A.T is a large (sparse) square matrix with A on the left
    and A.T on the top, zeros in the bottom left block
    Assumes A and v have the same shape
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

def expm_v(A, v, nterms=15):
    ''' Compute exp(A) v by Taylor expansion 
    
    Much faster for large A than jax.scipy.linalg.expm
    Also works for matrices v, best if tall not wide
    '''
    expAv = v # exp(A) v - starts with I v = v
    Akv = v # A^k v/k! - starts at v
    for k in range(1,nterms):
        Akv = A @ Akv / k
        expAv += Akv
    return expAv
expm_v = jax.jit(expm_v, static_argnums=2)

@jax.jit
def eval(A, B, C, L0, X, times):
    def func(i, ssr):
        x = X[[i],:]
        t = times[i]
        #L = jax.scipy.linalg.expm(jnp.cos(t) * A + jnp.sin(t) * B + C) @ L0
        #L = expm_v(jnp.cos(t) * A + jnp.sin(t) * B + C, L0)
        L = expm_AATv(jnp.cos(t) * A + jnp.sin(t) * B + C, L0)
        return ssr + jnp.linalg.norm(x - x @ L @ L.T)**2
    ssr = jax.lax.fori_loop(
        0,
        len(X),
        func,
        init_val = jnp.asarray([0.]),
    )
    return ssr / len(X)

@jax.jit
def resid(A, B, C, L0, X, times):
    def func(ssr, xt):
        x = xt[:-1]
        t = xt[-1]
        L = jax.scipy.linalg.expm(jnp.cos(t) * A + jnp.sin(t) * B + C) @ L0
        resid = x - x @ L @ L.T
        return ssr + jnp.linalg.norm(resid)**2, resid
    xt = jnp.concatenate([X, times.reshape((-1,1))], axis=1)
    ssr, resid = jax.lax.scan(
        func,
        init = jnp.asarray([0.]),
        xs = xt,
    )
    return resid


def jax_circ_pca_scipy(X, times, r):
    n,k = X.shape
    N = k*r - (r*(r+1)//2) # Num free vars per matrix
    #N = (k-r)*r
    times = jnp.asarray(times)
    L0 = jnp.asarray(low_rank_loadings(X, r))
    X = jnp.asarray(X)
    def extract(vars):
        # Pull out our A,B,C matrices from a flattened vector
        A_ = vars[  0:   N]
        B_ = vars[  N: 2*N]
        C_ = vars[2*N: 3*N]
        # Convert to rectangular lower triangular matrices
        idxs = jnp.tril_indices(n=k, k=-1, m=r)
        Atri = jnp.zeros((k,r)).at[idxs].set(A_)
        Btri = jnp.zeros((k,r)).at[idxs].set(B_)
        Ctri = jnp.zeros((k,r)).at[idxs].set(C_)
        #Atri = jnp.concatenate([jnp.zeros((r,r)), A_.reshape((-1,r))], axis=0)
        #Btri = jnp.concatenate([jnp.zeros((r,r)), B_.reshape((-1,r))], axis=0)
        #Ctri = jnp.concatenate([jnp.zeros((r,r)), C_.reshape((-1,r))], axis=0)
        return Atri, Btri, Ctri
    @jax.jit
    def f(vars):
        A, B, C = extract(vars)
        return eval(A, B, C, L0, X, times)[0]
    def pr(vars):
        A,B,C = extract(vars)
        print(eval(A,B,C, L0, X, times)[0])
    x0 = jnp.concatenate([
        jnp.zeros(N),
        jnp.zeros(N),
        jnp.zeros(N),
    ])
    #res = jax.scipy.optimize.minimize(
    #    f,
    #    x0 = x0,
    #    method = "BFGS",
    #    tol = 1e-2,
    #    options = {"gtol": 1e-2},
    #)
    res = scipy.optimize.minimize(
        f,
        x0,
        jac = jax.jacrev(f),
        callback = pr,
        method = "BFGS",
        tol = 1e-2,
        options = {"gtol": 1e-2, "maxiter": 100},
    )
    return (L0, *extract(res.x),res)
#L0, A, B, C, res = jax_circ_pca_scipy(simd, t, 2)

def jax_circ_pca(X, times, r):
    # By-hand optimizer for the parametrized PCA
    def extract(mat, k, r):
        # Return the lower triangular version
        idxs = jnp.tril_indices(n=k, k=-1, m=r)
        return jnp.zeros((k,r)).at[idxs].set(mat)
        #return jnp.concatenate([jnp.zeros((r,r)), mat.reshape((k-r,r))])
    n,k = X.shape
    times = jnp.asarray(times)
    L0 = jnp.asarray(low_rank_loadings(X, r))
    X = jnp.asarray(X)
    N = k*r - (r*(r+1)//2) # Num free vars per matrix
    #N = k*r - r*r
    def f(A,B,C):
        Atri = extract(A, k, r)
        Btri = extract(B, k, r)
        Ctri = extract(C, k, r)
        return eval(Atri,Btri,Ctri, L0, X, times)[0]
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
    for i in range(1500):
        A,B,C,m, v, res = update(A,B,C, m ,v, i+1)
        print(i, float(res))
    return (L0, extract(A, k, r), extract(B, k, r), extract(C, k, r), res)
L0, A, B, C, res = jax_circ_pca(simd, t, 2)