import pathlib
import pandas
import numpy
import pickle

import circ_pca

## Load the data
HOME = pathlib.Path("~/tgb/").expanduser()

data = pandas.read_csv(HOME / "data/circadian_controls/results/Liver/tpm_all_samples.txt", sep="\t", index_col=0)
sample_info = pandas.read_csv(HOME / "data/circadian_controls/results/Liver/all_samples_info.txt", sep="\t", index_col=0)
outlier_samples = [l.strip() for l in open(HOME / "data/circadian_controls/results/Liver/outlier_samples.txt").readlines()]
excluded_studies = ['Greenwell19_AdLib', 'Manella21_Liver', 'Greenwell19_NightFeed']
data_full = data.copy()
data = data_full.loc[:, (~data_full.columns.isin(outlier_samples)) & (~data_full.columns.map(sample_info.study).isin(excluded_studies))]

# Metadata / covariates
studies = data.columns[1:].map(sample_info.study)
times = data.columns[1:].map(sample_info.time)
cos = numpy.cos(times * 2 * numpy.pi / 24)
sin = numpy.sin(times * 2 * numpy.pi / 24)

high_expressed = ((data.iloc[:,1:].groupby(studies, axis=1).apply(lambda x: x.mean(axis=1)).median(axis=1) > 5)
    & (data.loc[:,~data.columns.isin(outlier_samples)].iloc[:,1:].min(axis=1) > 0))

# Aggregate the data
X = data.loc[high_expressed].iloc[:, 1:].T
means = X.groupby(sample_info.study).mean().loc[X.index.map(sample_info.study)].values
stds = X.groupby(sample_info.study).std().loc[X.index.map(sample_info.study)].values
X = (X - means) / stds


## Run circ PCA on it
result = circ_pca.compute_circ_pca(X, times, 3, 50)
with open(pathlib.Path("~/tgb/data/circadian_correlation/results/circ_pca.pickle").expanduser(), "wb") as f:
    pickle.dump(result, f)