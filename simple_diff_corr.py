import pathlib
import pandas
import random
import numpy
import pylab
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy

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
studies = data.columns[1:].map(sample_info.study)
times = data.columns[1:].map(sample_info.time)
cos = numpy.cos(times * 2 * numpy.pi / 24)
sin = numpy.sin(times * 2 * numpy.pi / 24)
cos2 = numpy.cos(2 * times * 2 * numpy.pi / 24)
sin2 = numpy.sin(2 * times * 2 * numpy.pi / 24)

high_expressed = ((data.iloc[:,1:].groupby(studies, axis=1).apply(lambda x: x.mean(axis=1)).median(axis=1) > 5)
    & (data.loc[:,~data.columns.isin(outlier_samples)].iloc[:,1:].min(axis=1) > 0))
X = data.loc[high_expressed].iloc[:500, 1:].T


geneA = 'ENSMUSG00000055116'
geneA = 'ENSMUSG00000020889'
geneB = 'ENSMUSG00000059824'
geneB = 'ENSMUSG00000059824'
gene_list = [
    'ENSMUSG00000055116',
    'ENSMUSG00000059824',
    'ENSMUSG00000020038',
    'ENSMUSG00000068742',
    'ENSMUSG00000020893',
    'ENSMUSG00000020889',
    'ENSMUSG00000021775'
]
gene_list = X.columns #WARNING: far too lage, will never complete


def regress_out_cosinor(data):
    # For each gene in data,
    # generate it's residual from a cosinor linear regressino fit
    results = []
    residuals = pandas.DataFrame(0, index = data.index, columns = data.columns)
    for gene, gene_data in data.iterrows():
        d = pandas.DataFrame({
            "X": numpy.log10(gene_data + 0.01),
            "study": studies,
            "time": times,
            "cos": cos,
            "sin": sin,
            "sample": gene_data.index,
        })
        d = d[~d['sample'].isin(outlier_samples)]
        for study, group in d.groupby('study'):
            fit = smf.ols(
                "X ~ 1 + sin + cos",
                data = group,
            ).fit()
            residuals.loc[gene, fit.resid.index] = fit.resid
    return residuals

X_regressed = regress_out_cosinor(X.T)


def model_XY(geneA, geneB, data):
    geneA_data = data.loc[geneA]
    geneB_data = data.loc[geneB]
    d = pandas.DataFrame({
        "X": geneA_data,
        "Y": geneB_data,
        "study": studies,
        "time": times,
        "cos": cos,
        "sin": sin,
        "sample": data.columns,
    })
    d = d[~d['sample'].isin(outlier_samples)]
    d['XY'] = d['X'] * d['Y']

    res = smf.ols(
        "XY ~ 1 + (sin + cos)",
        data = d,
    ).fit()
    #print(res.summary())

    return res, d


# Compute the fits for the selected genes
gene_pairs = [(a,b) for a in gene_list for b in gene_list]
random.seed(0)
random.shuffle(gene_pairs)
results_list = []
num_significant = 0
num_run = 0
for geneA, geneB in gene_pairs:
    if geneA == geneB:
        continue

    res, d = model_XY(geneA,geneB, X_regressed)

    res_summary = {
        "geneA": geneA,
        "geneB": geneB,
        "1": res.params["Intercept"],
        "sin": res.params["sin"],
        "cos": res.params["cos"],
        "corr_diff_p": res.f_test("sin = 0, cos = 0").pvalue
    }
    results_list.append(res_summary)
    if res_summary['corr_diff_p'] < 1e-3:
        num_significant += 1
    num_run += 1
    if num_run % 1000 == 0:
        print(num_run, num_significant)
results = pandas.DataFrame(results_list)

results.to_csv(outdir / "simple_diff_corr.results.txt", sep="\t", index=False)
