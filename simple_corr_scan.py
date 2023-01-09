'''
Process just one single study - Morton20
To check for differential correlation with a very simple model
'''
import pathlib
import numpy
import pylab
import pandas
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import tqdm

outdir = pathlib.Path("results/")
outdir.mkdir(exist_ok=True)

# Data
data = pandas.read_csv("~/data/circadian_controls/results/Liver/tpm_all_samples.txt", sep="\t", index_col=0)
sample_info = pandas.read_csv("~/data/circadian_controls/results/Liver/all_samples_info.txt", sep="\t", index_col=0)
outlier_samples = [l.strip() for l in open(pathlib.Path("~/data/circadian_controls/results/Liver/outlier_samples.txt").expanduser()).readlines()]
excluded_studies = ['Greenwell19_AdLib', 'Manella21_Liver', 'Greenwell19_NightFeed']
data_full = data.copy()
data = data_full.loc[:, (~data_full.columns.isin(outlier_samples)) & (~data_full.columns.map(sample_info.study).isin(excluded_studies))]

# Metadata / covariates
studies = pandas.Series(data.columns[1:].map(sample_info.study), index=data.columns[1:])
times = pandas.Series(data.columns[1:].map(sample_info.time), index=data.columns[1:])
times_24 = times % 24
cos = numpy.cos(times * 2 * numpy.pi / 24)
sin = numpy.sin(times * 2 * numpy.pi / 24)
cos2 = numpy.cos(2 * times * 2 * numpy.pi / 24)
sin2 = numpy.sin(2 * times * 2 * numpy.pi / 24)

# Select just the Morton20 data and just some high expressed genes
d = data.iloc[:,1:]
d = d.loc[~d.index.isin(outlier_samples), studies == "Morton20_Liver"]
high_expressed = (d.median(axis=1) > 1)
#X = d.loc[high_expressed].iloc[:5000, 1:].T
# Run using ALL GENES of high expression
X = d.loc[high_expressed].T

# Select the data from the next largest study to compare
d2 = data.iloc[:,1:]
d2 = d2.loc[~d2.index.isin(outlier_samples), studies == "Atger15_AdLib"]
X2 = d2.loc[X.columns].T

# Compute the correlations and format as a long vector
corr_by_time = X.groupby(times).corr()
corr_by_time.index.names = ["time", "A"]
corr_by_time.columns.name = "B"
corr_by_time = corr_by_time.reset_index()
corr_by_time['cos'] = numpy.cos( 2 * numpy.pi * corr_by_time.time / 24)
corr_by_time['sin'] = numpy.sin( 2 * numpy.pi * corr_by_time.time / 24)
#corr_by_time = corr_by_time.melt(
#    id_vars = ["time", "cos", "sin", "A"],
#    var_name = "B",
#    value_name = "corr",
#)
## Drop duplicates arising from symmetry
#corr_by_time = corr_by_time[corr_by_time.A > corr_by_time.B]

# And grab all but the selected study for comparison
d_all = data.iloc[:,1:]
d_all = d_all.loc[~d_all.index.isin(outlier_samples), studies != "Morton20_Liver"]
X_all = d_all.loc[X.columns].T

def regress_out_cosinor(data):
    # For each gene in data,
    # generate it's residual from a cosinor linear regression fit
    residuals = pandas.DataFrame(0, index = data.index, columns = data.columns)
    for study, study_data in data.groupby(studies):
        exog = pandas.DataFrame({
            "cos": cos[study_data.index],
            "sin": sin[study_data.index],
            "const": 1,
        })
        fit = sm.OLS(
            study_data,
            exog,
        ).fit()
        residuals.loc[fit.resid.index, fit.resid.columns] = fit.resid
    return residuals

X_all_regressed = regress_out_cosinor(X_all)


def compare_f_test(fit, restricted):
    # Work-around for statsmodels issue when the endog matrix is 2d
    def ssr(fit):
        return numpy.sum(fit.wresid * fit.wresid, axis=0)
    ssr_full = ssr(fit)
    ssr_restr = ssr(restricted)
    df_full = fit.df_resid
    df_restr = restricted.df_resid

    df_diff = (df_restr - df_full)
    f_value = (ssr_restr - ssr_full) / df_diff / ssr_full * df_full
    p_value = scipy.stats.f.sf(f_value, df_diff, df_full)
    return f_value, p_value, df_diff


results_list = []
iters = 0
for A, gene_data in tqdm.tqdm(corr_by_time.groupby("A")):
    # Fit the correlation values from Morton20 to a cosinor
    endog = gene_data.drop(columns=["time", "cos", "sin", "A"])
    exog = sm.tools.add_constant(gene_data[["time", "cos", "sin"]])
    fit = sm.OLS(
        endog = endog,
        exog = exog,
    ).fit()

    restricted = sm.OLS(
        endog = endog,
        exog = exog[['const']],
    ).fit()

    f, p, df_diff = compare_f_test(fit, restricted)

    # Fit x*y values to a cosinor in all the other studies
    # after removing the expression-level cosinors
    xy_endog = (X_all_regressed[A].T * X_all_regressed.T).T
    # And scale as sqrt(xy), preserving the sign
    xy_endog = numpy.sqrt(numpy.abs(xy_endog)) * numpy.sign(xy_endog)
    xy_exog = pandas.DataFrame({
        "cos": cos[X_all_regressed.index],
        "sin": sin[X_all_regressed.index],
        "const": 1,
    })
    xy_fit = sm.OLS(
        endog = xy_endog,
        exog = xy_exog,
    ).fit()
    xy_fit_reduced = sm.OLS(
        endog = xy_endog,
        exog = xy_exog[['const']],
    ).fit()
    _, xy_p, _ = compare_f_test(xy_fit, xy_fit_reduced)

    hits = pandas.DataFrame({
        "A": A,
        "p": pandas.Series(p, index=endog.columns),
        "xy_p": pandas.Series(xy_p, index=xy_endog.columns),
    }).reset_index()
    # Store only the potential hits
    results_list.append( hits[(hits.p < 0.01) & (hits.xy_p < 0.01)] )
    iters += 1
results = pandas.concat(results_list)
results = results[results['index'] != results['A']] # Don't need A <-> A tests

# Plot the best one
top_results = results[(results.p < 1e-3) & (results.xy_p < 1e-5)]
A,B,p,xy_p = top_results.sort_values(by="p").iloc[0]
def plot_genes(A,B):
    gene_data = corr_by_time[(corr_by_time.A == A)]

    fig, ax = pylab.subplots()
    ax.scatter(
        gene_data.time % 24,
        gene_data[B],
    )

    expr_data = {
        "A": X[A],
        "B": X[B],
        "time": X.index.map(times),
    }
    fig = sns.relplot(
        x = "A",
        y = "B",
        col = "time",
        col_wrap = 4,
        data = expr_data,
    )
    fig.fig.suptitle("Morton20")

    # In another study
    #expr_data2 = {
    #    "A": X2[A],
    #    "B": X2[B],
    #    "time": X2.index.map(times),
    #}
    #fig = sns.relplot(
    #    x = "A",
    #    y = "B",
    #    col = "time",
    #    col_wrap = 4,
    #    data = expr_data2,
    #)
    #fig.fig.suptitle("Atger15")

    # In all the other studies:
    xy =pandas.DataFrame({
         "xy": X_all_regressed[A] * X_all_regressed[B],
        "time": times[X_all_regressed.index],
        "study": studies[X_all_regressed.index],
    })
    sns.relplot(
        x = "time",
        y = "xy",
        hue = "study",
        data = xy
    )

plot_genes(A,B)