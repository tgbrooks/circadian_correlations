'''
Simulate one single study of same study design as Morton20
Then attempt to detect correlation rhythms in this study.
'''
import numpy
import pylab
import pandas
import statsmodels.formula.api as smf
import statsmodels.api as sm

TIMEPOINTS = 13
TIMESTEP = 3
TO_RADIANS = 2 * numpy.pi / 24
N_SAMPLES = 6
N_ITER = 1000

numpy.random.seed(0)
t_by_timepoint = numpy.arange(0,13)* TIMESTEP # one per timepoint
t = numpy.repeat(t_by_timepoint, N_SAMPLES)# one per sample

true_corr = 0.25 - numpy.sin(TO_RADIANS * t)/4 # ranges from 0 to 0.5 throughout the day
#true_corr =  numpy.sin(TO_RADIANS * t)/2 # ranges from -0.5 to 0.5 throughout the day
#true_corr = 0.5 * numpy.ones(len(t))

results_list = []
computed_corrs = []
for iter in range(N_ITER):
    data = numpy.concatenate([
        numpy.random.multivariate_normal(
            [0,0],
            cov = [
                [1, true_corr[i*N_SAMPLES]],
                [true_corr[i*N_SAMPLES], 1],
            ],
            size = 6
        )
        for i in range(TIMEPOINTS)
    ])

    computed_corr = numpy.array([
        numpy.corrcoef(
            data[i*N_SAMPLES:(i+1)*N_SAMPLES, :].T,
        )[0,1]
        for i in range(TIMEPOINTS)
    ])
    computed_corrs.append(computed_corr)

    fit = smf.ols(
        "corr ~ cos + sin",
        data = {
            "corr": computed_corr,
            "cos": numpy.cos(t_by_timepoint * TO_RADIANS),
            "sin": numpy.sin(t_by_timepoint * TO_RADIANS),
        }
    ).fit()
    p = fit.f_test("cos = 0, sin = 0").pvalue

    fit2 = smf.ols(
        "y ~ (cos + sin) * x",
        data = {
            "x": data[:, 0],
            "y": data[:, 1],
            "cos": numpy.cos(t * TO_RADIANS),
            "sin": numpy.sin(t * TO_RADIANS),
        }
    ).fit()
    p_ols = fit2.f_test("cos:x = 0, sin:x = 0").pvalue
    results_list.append({
        "iter": iter,
        "p": p,
        "p_ols": p_ols,
    })
results = pandas.DataFrame(results_list)

# Plot some of the data
fig, ax = pylab.subplots()
for i in range(10):
    ax.plot(computed_corrs[i])
ax.plot(true_corr[::N_SAMPLES], color="k", linewidth=3)

# Estimate with a background of non-differentially correlated genes
FRAC_DIFF_CORR = 0.05
N_COMPARISONS = 5000**2
N_NON_DIFF_CORR = int(N_COMPARISONS * (1 - FRAC_DIFF_CORR))
N_DIFF_CORR = int(N_COMPARISONS * FRAC_DIFF_CORR)

non_diff_corr_ps = numpy.random.uniform(size=N_NON_DIFF_CORR)
diff_corr_ps = results.p_ols.sample(n=N_DIFF_CORR, replace=True)
_, qs, _, _ = sm.stats.multipletests(numpy.concatenate([non_diff_corr_ps, diff_corr_ps]), method="fdr_bh")