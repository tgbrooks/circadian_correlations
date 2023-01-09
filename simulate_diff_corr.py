import pandas
import numpy
import pylab
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

numpy.random.seed(0)
NUM_STUDIES = 20
STUDY_TIMEPOINTS = numpy.repeat(numpy.arange(0,24, 4), 4)
NOISE_SIZE = 0.3

data = pandas.DataFrame({
    "time": numpy.repeat(STUDY_TIMEPOINTS, NUM_STUDIES),
    "study": [i for j in range(len(STUDY_TIMEPOINTS))
                for i in range(NUM_STUDIES)],
})

by_study_bias = numpy.random.normal(size=NUM_STUDIES)
data['by_study_bias1'] = data.study.map(lambda x: by_study_bias[x])
by_study_bias = numpy.random.normal(size=NUM_STUDIES)
data['by_study_bias2'] = data.study.map(lambda x: by_study_bias[x])

data['cos'] = numpy.cos(data.time * 2 * numpy.pi / 24)
data['sin'] = numpy.sin(data.time * 2 * numpy.pi / 24)
data['cos2'] = numpy.cos(2 * data.time * 2 * numpy.pi / 24) # Twice the frequency/half the period
data['sin2'] = numpy.sin(2 * data.time * 2 * numpy.pi / 24)
common_component = numpy.random.normal(size=len(data))
data['X'] = common_component * (1       ) + numpy.random.normal(size=len(data)) * NOISE_SIZE + data.sin * data.by_study_bias2
data['Y'] = common_component * (data.sin) + numpy.random.normal(size=len(data)) * NOISE_SIZE + data.cos * data.by_study_bias1

data['XY'] = data.X * data.Y

data['group'] = 0

fam = sm.families.Gaussian()
ind = sm.cov_struct.Exchangeable()
res = smf.gee(
    "Y ~ 1 + X + sin* X + cos* X",
    groups = "study",
    data = data,
    family = fam,
    cov_struct = ind,
).fit()
print(res.summary())

res_reduced_Y = smf.gee(
    "Y ~ 1 + sin + cos",
    groups = "study",
    data = data,
    family = fam,
    cov_struct = ind
).fit()
res_reduced_X = smf.gee(
    "X ~ 1 + sin + cos",
    groups = "study",
    data = data,
    family = fam,
    cov_struct = ind
).fit()


#res = smf.mixedlm(
#    "Y ~ 1 + X + sin : X + cos : X",
#    groups = "study",
#    re_formula = "~  1 + X",#  + cos* X",
#    data = data,
#).fit()
#print(res.summary())
#

data['Y_only_fit'] = res_reduced_Y.fittedvalues
data['X_only_fit'] = res_reduced_X.fittedvalues
data['fit'] = res.fittedvalues

plot_d = data.copy()
plot_d['x'] = data.X - data.X_only_fit
plot_d['y'] = data.Y - data.Y_only_fit
fig = sns.lmplot(
    x = "x",
    y = "y",
    col = "time",
    col_wrap = 3,
    data = plot_d,
)

fig = sns.lmplot(
    x = "X",
    y = "Y",
    hue = "time",
    palette = "twilight",
    ci = None,
    data = data,
    col = "study",
    col_wrap=5,
)

#fig = sns.relplot(
#    x = "time",
#    y = "Y",
#    data = data,
#    col = "study",
#    col_wrap=5,
#)

#fig = sns.relplot(
#    x = "time",
#    y = "fit",
#    data = data,
#    col = "study",
#    col_wrap=5,
#)

pylab.show()