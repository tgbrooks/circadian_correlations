from typing import Union
from dataclasses import dataclass
import textwrap

import numpy as np
import scipy.linalg
import scipy.cluster.hierarchy
import statsmodels.api as sm

@dataclass
class CircPCA:
    '''
    Represents the results of a "Circadian PCA".
    '''

    # Values used to fit
    data: np.ndarray
    time: np.ndarray

    r: int # Rank of reduction
    nobs: int # number of observations
    nvars: int # number of variables measured per observation

    # Results
    # PCA of the data
    W0: np.ndarray
    # The circadian terms:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray

    # sum-of-squares of residuals of the
    # projection to the given subspace
    resid_sum_squares: float
    # Initial RSS from project to the fixed
    # PCA subpsace, invariant in time
    PCA_resid_sum_squares: float

    # Convergence info
    niter: int # number of iterations
    RSS_history: np.ndarray # list of RSS during the fit, to check for convergence

    # Reduction to lower dimension prior to circ PCA info:
    nvars_reduced: int # number of variables reduced to by PCA before performing circ PCA
    reduction_weights: Union[np.ndarray, None] #Weights of the original variables used to perform the reduction

    def weights(self, t):
        ''' return n x r matrix of weights at time t
        
        t in radians'''
        w = expm_AATv(self.A * np.cos(t) + self.B * np.sin(t) + self.C, self.W0)
        if self.reduction_weights is not None:
            return self.reduction_weights @ w
        return w

    def angles(self, ts = np.linspace(0,2*np.pi, 31), from_mat=None):
        ''' Compute the angles between the r-dimensional subspace at time t
        compared to that at time 0.

        from_mat: matrix to obtain angles with respect to. If None (default),
                    then this will be the t=0 matrix
        '''
        weights = [self.weights(t) for t in ts]
        if from_mat is None:
            from_mat = weights[0]
        angles = np.array([scipy.linalg.subspace_angles(Wt, from_mat)
                                for Wt in weights])
        return angles, ts

    def summary(self):
        angles, ts = self.angles()
        angles_from_W0, ts = self.angles(expm_AATv(self.C, self.W0))
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
        |A| = {np.linalg.norm(self.A, ord=float('inf'))**2:0.3f}\t|B| = {np.linalg.norm(self.B, ord=float('inf'))**2:0.3f}\t|C| = {np.linalg.norm(self.C, ord=float('inf'))**2:0.3f}
        LARGEST ANGLE FROM t=0: {np.max(angles):0.3f}
        LARGEST ANGLE FROM exp(C)W0: {np.max(angles_from_W0):0.3f}
        AVERAGE ANGLE FROM exp(C)W0: {np.mean(angles_from_W0[:,0]):0.3f}
        ''').strip()

    def plot(self):
        ts = np.linspace(0, 2*np.pi, 31)
        weights = np.array([self.weights(t) for t in ts])
        total_weight_by_time = (weights**2).sum(axis=2)

        import pylab
        fig, ax = pylab.subplots()
        ax.imshow(total_weight_by_time.T)
        ax.set_xlabel('time')
        ax.set_ylabel('variable')
        return fig

    def plot_residuals(self):
        ''' Plot versus time the residuals of the projection '''
        import pylab
        PCA_weights = self.reduction_weights @ self.W0
        residuals = []
        pca_residuals = []
        for t, row in zip(self.time, self.data):
            W = self.weights(t)
            # Residuals from Circ PCA
            projection = row @ W @ W.T
            residual = np.linalg.norm(row - projection)**2
            residuals.append(residual)
            # Standard PCA residuals
            pca_projection = row @ PCA_weights @ PCA_weights.T
            pca_residual = np.linalg.norm(row - pca_projection)**2
            pca_residuals.append(pca_residual)
        fig, ax = pylab.subplots(figsize=(6,6))
        ax.scatter(
            self.time % 24,
            residuals,
        )
        fit_t =  np.linspace(0,24,25)
        loess_fit = sm.nonparametric.lowess(
            residuals,
            self.time,
            xvals = fit_t,
        )
        ax.plot(fit_t, loess_fit, label="CircPCA")

        ax.scatter(
            self.time % 24,
            pca_residuals,
        )
        fit_t =  np.linspace(0,24,25)
        pca_loess_fit = sm.nonparametric.lowess(
            pca_residuals,
            self.time,
            xvals = fit_t,
        )
        ax.plot(fit_t, pca_loess_fit, label="PCA")
        ax.set_ylim(0)
        ax.set_xticks(np.arange(0,25, 3))
        return fig

    def plot_RSS_history(self):
        import pylab
        fig, ax = pylab.subplots()
        ax.plot(
            self.RSS_history
        )
        ax.set_xlabel("Iteration count")
        ax.set_ylabel("Residual Sum of Squares")
        return fig

    def plot_corr(self):
        ts = np.linspace(0,24,25)
        weights = [self.weights(t) for t in ts]
        linkage = scipy.cluster.hierarchy.linkage(self.data.T, metric="correlation")
        leaves_list = scipy.cluster.hierarchy.leaves_list(linkage)

        import pylab
        fig, axes = pylab.subplots(figsize=(10,6), ncols=2, sharex=True, sharey=True)
        corr = weights[0][:,[0]] @ weights[0][:,[0]].T
        axes[0].imshow(
            corr[leaves_list][:,leaves_list]
        )
        corr = weights[12][:,[0]] @ weights[0][:,[0]].T
        axes[1].imshow(
            corr[leaves_list][:,leaves_list]
        )
        return fig

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
    res = v.copy() # first term of Taylor expansion
    Bkv = v.copy() # B^k v / k!  - initializes at v for k=0
    for k in range(1,nterms):
        top = -A.T @ Bkv + A[:r] @ Bkv[:r]
        bottom = A[r:] @ Bkv[:r]
        Bkv[:r,:] = (top/k)
        Bkv[r:, :] = (bottom/k)
        res += Bkv
    return res

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

    data = np.array(data)
    time = np.array(time)

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
    from jax_circ_pca import jax_circ_pca
    result = jax_circ_pca(reduced_data, time, r, verbose=verbose)

    if R is not None:
        # Correct the result CircPCA object to describe the dimension reduction performed
        result.nvars = k
        result.data = data
        result.reduction_weights = reduction_weights

    return result

def low_rank_weights(X, r):
    ''' give best rank r weights to approximate X '''
    u,d,vt = np.linalg.svd(X, full_matrices=False)
    return vt[:r,:].T

if __name__ == '__main__':
    ## EXAMPLE DATA
    np.random.seed(0)
    N = 500
    t = np.linspace(0,2 * np.pi, N)
    scores = np.random.normal(size=N)
    simd = np.concatenate([
        [np.cos(t/2) * scores*5 + scores * 10],
        [np.sin(t/2) * scores*5 + scores * 10],
        np.random.normal(size=(1,N))*5,
        np.random.normal(size=(100,N)),
    ], axis=0).T

    result = compute_circ_pca(simd, t, 2, R = 10, verbose=True)
    print(result.summary())