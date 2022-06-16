import pathlib
import pandas
import random
import numpy
import pylab
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy

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


fam = sm.families.Gaussian()
ind = sm.cov_struct.Exchangeable()
def main_model(geneA, geneB, data):
    geneA_data = data.iloc[:,1:].loc[geneA]
    geneB_data = data.iloc[:,1:].loc[geneB]
    d = pandas.DataFrame({
        "X": numpy.log10(geneA_data + 0.01),
        "Y": numpy.log10(geneB_data + 0.01),
        "study": studies,
        "time": times,
        "cos": cos,
        "sin": sin,
        #"cos2": cos2,
        #"sin2": sin2,
        "sample": data.iloc[:,1:].columns,
    })
    d = d[~d['sample'].isin(outlier_samples)]
    d['X'] -= d.X.mean()
    d['Y'] -= d.Y.mean()

    res = smf.gee(
        "Y ~ 1 + X + (sin + cos)*X",
        groups = "study",
        data = d,
        family = fam,
        cov_struct = ind,
    ).fit()
    #print(res.summary())

    return res, d


# Compute the fits for the selected genes
data_selected = data.loc[gene_list]
gene_pairs = [(a,b) for a in gene_list for b in gene_list]
random.seed(0)
random.shuffle(gene_pairs)
results_list = []
num_significant = 0
num_run = 0
for geneA, geneB in gene_pairs:
    if geneA == geneB:
        continue

    res, d = main_model(geneA,geneB, data_selected)

    res_summary = {
        "geneA": geneA,
        "geneB": geneB,
        "symbolA": data_selected.loc[geneA].Symbol,
        "symbolB": data_selected.loc[geneB].Symbol,
        "X": res.params["X"],
        "sin:X": res.params["sin:X"],
        "cos:X": res.params["cos:X"],
        "corr_diff_p": res.f_test("sin:X = 0, cos:X = 0").pvalue
    }
    results_list.append(res_summary)
    if res_summary['corr_diff_p'] < 1e-2:
        num_significant += 1
    num_run += 1
    if num_run % 1000 == 0:
        print(num_run, num_significant)
results = pandas.DataFrame(results_list)

def plot_genes(A, B):
    ''' Visualize the time-dependent correlation between the genes A and B '''
    res, d = main_model(A, B, data)
    fit = res.params

    symbolA = data.Symbol.loc[A]
    symbolB = data.Symbol.loc[B]

    plot_d = d.copy()
    plot_d['time_bucket'] = pandas.cut((d.time % 24), numpy.linspace(0,24,7), right=False)
    plot_d['time24'] = plot_d.time %24
    fig = sns.lmplot(
        x = "X",
        y = "Y",
        hue = "study",
        ci = None,
        col = "time_bucket",
        col_wrap = 3,
        data = plot_d,
    )
    fig.set(xlabel=symbolA, ylabel=symbolB)

    #res_reduced_Y = smf.gee(
    #    "Y ~ 1 + sin + cos + sin2 + cos2",
    #    groups = "study",
    #    data = d,
    #    family = fam,
    #    cov_struct = ind
    #).fit()
    #res_reduced_X = smf.gee(
    #    "X ~ 1 + sin + cos + sin2 + cos2",
    #    groups = "study",
    #    data = d,
    #    family = fam,
    #    cov_struct = ind
    #).fit()


    ##estimated interaction plot
    #fig, (ax) = pylab.subplots()
    #t = numpy.linspace(0,24,25)
    #cost = numpy.cos(t * 2 * numpy.pi / 24) 
    #sint = numpy.sin(t * 2 * numpy.pi / 24) 
    #cos2t = numpy.cos(2 * t * 2 * numpy.pi / 24) 
    #sin2t = numpy.sin(2 * t * 2 * numpy.pi / 24) 
    #LARGE = 1e6
    #ax.plot(
    #    t,
    #    #fit['X'] + fit['cos:X'] * cost + fit['sin:X'] * sint,
    #    #fit['X'] + fit['cos:X'] * cost + fit['sin:X'] * sint + fit['cos2:X'] * cos2t + fit['sin2:X'] * sin2t,
    #    # NOTE: we multiple and divide by LARGE so that non-interaction effects (i.e., without X) are made negligible
    #    res.predict(
    #        {"X": LARGE * numpy.ones(len(t)), "cos": cost, "sin": sint, "cos2": cos2t, "sin2": sin2t}
    #    ) / LARGE,
    #    label="estimated corr"
    #)
    #ax.axhline(0, color='k')
    #ax.axhline(1, color='k')
    ## 'spurious' correlation from mean-value changes
    ## take derivatives of the two fit curves and if they move together, then we have 'spurious correlation'
    #derivA = -res_reduced_X.params['cos'] * sint + res_reduced_X.params['sin'] * cost - 2 * res_reduced_X.params['cos2'] *sin2t + 2* res_reduced_X.params['sin2'] * cos2t
    #derivB = -res_reduced_Y.params['cos'] * sint + res_reduced_Y.params['sin'] * cost - 2 * res_reduced_Y.params['cos2'] *sin2t + 2* res_reduced_Y.params['sin2'] * cos2t
    #fake_corr = derivA * derivB
    #ax.plot(
    #    t,
    #    fake_corr,
    #    label="spurious correlation"
    #)
    #ax.set_xticks([0,4,8,12,16,20,24])
    #fig.legend()

    fig = sns.lmplot(
        x = "X",
        y = "Y",
        hue = "time24",
        palette = "twilight",
        ci = None,
        data = plot_d,
        col = "study",
        col_wrap=8,
        facet_kws = {"sharex":False, "sharey": False},
    )