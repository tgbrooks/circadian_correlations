import numpy
import scipy.stats
import pandas
import pylab

def scan_for_diff_corr(Xin, times, studies):
    '''
    Scans all pairs of variables in X for differential correlation through the day by checking
    y ~ 1 + (cos(t) + sin(t))*x
    for interaction terms (cos(t):x or sin(t):x) for all variables x,y in X

    data = n x r dataframe, n is the number of observations and k is the number of features
    times = n list of timepoints
    studies = n list of studies the samples came from
    '''
    Xin = numpy.log10(Xin + 0.01)
    means = Xin.groupby(studies).mean()
    X = Xin - means.loc[studies].values
    n,k = X.shape
    assert len(times) == n

    COS = numpy.cos(times * 2 * numpy.pi / 24)
    SIN = numpy.sin(times * 2 * numpy.pi / 24)
    results_list = []
    for col in X.columns:
        # Regress all `y`s versus one 'x'
        x = X[col]
        # Construct the covariate matrix
        covariates = numpy.array([
            numpy.ones(n),
            COS,
            SIN,
            x,
            COS * x,
            SIN * x,
        ]).T 

        # Linear regression across all the samples
        params, ssr, rank, singvals = numpy.linalg.lstsq(
            covariates,
            X, # all the possible Y's at once
            rcond = None
        )

        _, restricted_ssr, _, _= numpy.linalg.lstsq(
            covariates[:,:-2],
            X, # all the possible Y's at once
            rcond = None
        )
        F = (restricted_ssr - ssr) / (2) / (ssr / (n - covariates.shape[1]))
        dof = (2, n - covariates.shape[1])
        p_value = scipy.stats.f(*dof).sf(F)

        # Extract results
        results_list.append(pandas.DataFrame({
            "y_var": X.columns,
            "x_var": [col for _ in range(k)],
            "x": params[3],
            "xcos": params[4],
            "xsin": params[5],
            "ssr": ssr,
            "restricted_ssr": restricted_ssr,
            "F": F,
            "p": p_value,
        }))
    results =  pandas.concat(results_list)
    results = results[results.y_var != results.x_var] # Drop the trivial comparisons where x = y
    return results

def plot_scan(X, geneX, geneY, studies, times):
    Xmod = numpy.log10(X + 0.01)
    means = Xmod.groupby(studies).mean()
    Xmod = Xmod - means.loc[studies].values

    COS = numpy.cos(times * 2 * numpy.pi / 24)
    SIN = numpy.sin(times * 2 * numpy.pi / 24)
    results_list = []
    x = Xmod[geneX]
    y = Xmod[geneY]

    fig, axes = pylab.subplots(nrows=2)
    axes[0].scatter(
        COS * x,
        y,
    )
    axes[1].scatter(
        SIN * x,
        y
    )