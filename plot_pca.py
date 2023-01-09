import jax.numpy as jnp
import pylab

from jax_circ_pca import expm_AATv

def plot_parameterized_pca(circ_pca_results):
    weights, ts = circ_pca_results.weights()

    # We don't parameterize the components individually
    # so rotation between the r components doesn't matter
    # therefore we take the sum-of-squares of the weights
    # in all components to summarize the contribution
    # of a variable across time
    total_weight_by_time = (Os**2).sum(axis=2)

    fig, ax = pylab.subplots(figsize=(6,6))
    h = ax.imshow(total_weight_by_time, vmin=0, vmax=1)
    fig.legend(h)