import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

import dit
import admUI
import math

pp = PdfPages('multipage2.pdf')
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "martital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

df = pd.read_csv(data_url, names=column_names)
for k, v in zip(column_names, df.dtypes):
    if v == "object":
        df[k] = df[k].apply(lambda x: x.strip())
age_groups = ['<24', '24-35', '36-50', '>50']
df['age-group'] = df.age.apply(lambda x: age_groups[np.digitize(x, [24, 36, 51])])

hours_per_week_group = ['<=40', '>40']
df['hours-per-week-group'] = df['hours-per-week'].apply(lambda x: hours_per_week_group[0 if x <= 40 else 1])
selected_columns = [
    'income',
    'education',
    'sex',
    'race',
    'occupation',
    'age-group',
    'hours-per-week-group'
]

rvs_names = [
    'S',  # income
    'E',  # education
    'G',  # sex
    'R',  # race
    'O',  # occupation
    'A',  # age
    'H',  # hours-per-week
]

rvs_to_name = dict(zip(rvs_names, selected_columns))
# take all samples with attributes that we're interested in
data_array = list(map(lambda r: tuple(r[k] for k in selected_columns), df.to_dict("record")))

# create distribution from the samples with uniform distribution
dist_census = dit.Distribution(data_array, [1. / df.shape[0] ] * df.shape[0])

# set variable aliases to the discribution
dist_census.set_rv_names("".join(rvs_names))

# define measurement metrics
metric_keys = ['si', 'ci', 'ui_0', 'ui_1']

# define legend labels
latex_labels = [
    '$SI$',
    '$CI$',
    '$UI(S; Y \\backslash Z)$',
    '$UI(S; Z \\backslash Y)$'
]

# define colour for ploting
colors = [
    '#27AB93',
    '#FF5716',
    '#D33139',
    '#522F60',
]
def plot(data):

    # definte the size of figure
    plt.figure(figsize=(6, 0.5*len(data)))

    labels = []
    metrics = []

    # suffix each variable name with an alias {Y, Z}
    for v in data:
        labels.append(", ".join(map(lambda p: "%s (%s)" % p, zip(v['variables'], ['Y', 'Z']))))

    for m in metric_keys:
        mm = []
        # extract corresponding metric from elements in array
        for v in data:
            mm.append(v['metrics'][m]/v['metrics']['mi'])
            plt.title(label='Mutual information: '+str(v['metrics']['mi']),loc='right')
        metrics.append(mm)


    left = np.array([0]*len(labels))

    # plotting
    for i, kk in enumerate(metric_keys):
        plt.barh(
            labels, metrics[i], align='center', height=.5, left=left, label=latex_labels[i],color=colors[i],
        )
        left = left + metrics[i]
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(pp, format='pdf')
    plt.show()


def information_decomposition(dist, src, to=""):
    rvs = src+to

    P = dist.marginal(rvs)

    variables = P._rvs

    q_SXY = admUI.computeQUI(distSXY = P)

    h_SgXY =  dit.shannon.conditional_entropy(q_SXY, 'S', 'XY')

    ui_SX_Y = dit.shannon.conditional_entropy(q_SXY, 'S', 'Y') - h_SgXY
    ui_SY_X = dit.shannon.conditional_entropy(q_SXY, 'S', 'X') - h_SgXY

    si_SXY_1 = dit.shannon.mutual_information(q_SXY, 'S', 'X') - ui_SX_Y
    si_SXY_2 = dit.shannon.mutual_information(q_SXY, 'S', 'Y') - ui_SY_X

    # sanity check
    assert math.isclose(si_SXY_1, si_SXY_2, abs_tol=1e-6), "SI_S_XY: %f | %f" % (si_SXY_1, si_SXY_2)

    si_SXY = si_SXY_1

    ci_SXY = si_SXY - dit.multivariate.coinformation(P, rvs)
    i_S_XY = dit.shannon.mutual_information(P, 'S', to)

    # sanity check
    assert math.isclose(i_S_XY, si_SXY + ci_SXY + ui_SX_Y + ui_SY_X, abs_tol=1e-6), \
        "MI = decompose : %f | %f" % (i_S_XY, si_SXY + ci_SXY + ui_SX_Y + ui_SY_X)

    uis = [ui_SX_Y, ui_SY_X]
    return {
        "variables": tuple(map(lambda x: rvs_to_name[x], to)),
        "metrics": {
            "mi": i_S_XY,
            "si": si_SXY,
            "ci": ci_SXY,
            "ui_0": uis[variables[to[0]]-1] ,
            "ui_1": uis[variables[to[1]]-1]
        }
    }
decomp_S_HO = information_decomposition(dist_census, 'S', 'HO')
decomp_S_EG = information_decomposition(dist_census, 'S', 'EG')
decomp_S_ER = information_decomposition(dist_census, 'S', 'ER')
decomp_S_RO = information_decomposition(dist_census, 'S', 'RO')
decomp_S_AG = information_decomposition(dist_census, 'S', 'AG')

# didn't converge, should we remove it?
# decomp_S_EO = compute_decomposition_from(dist_census, 'S', ['E', 'O'])
plot([
    decomp_S_EG,
    decomp_S_ER,
    decomp_S_RO,
    decomp_S_AG,
    decomp_S_HO,
][::-1])
pp.close()
