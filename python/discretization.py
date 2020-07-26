import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

import dit
import admUI
import math

pp = PdfPages('disc05.pdf')
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
rvs_names = [
    'S',  # income
    'E',  # education
    'G',  # sex
    'R',  # race
    'O',  # occupation
    'A',  # age
    'H',  # hours-per-week
]


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
df = pd.read_csv(data_url, names=column_names)
for k, v in zip(column_names, df.dtypes):
    if v == "object":
        df[k] = df[k].apply(lambda x: x.strip())
age_groups = ['<24', '24-35', '36-50', '>50']
df['age-group'] = df.age.apply(lambda x: age_groups[np.digitize(x, [24, 36, 51])])
num_bins = 5
intervals = np.linspace(0,40,num_bins)
df['hours-per-week-bins'] = pd.cut(df['hours-per-week'],bins = intervals)
df['hours-per-week-bins'] = df['hours-per-week-bins'].cat.add_categories('>40').fillna('>40').astype(str)
selected_columns = [
    'income',
    'education',
    'sex',
    'race',
    'occupation',
    'age-group',
    'hours-per-week-bins'
]
# define measurement metrics
metric_keys = ['si', 'ci', 'ui_0', 'ui_1']
rvs_to_name = dict(zip(rvs_names, selected_columns))


# take all samples with attributes that we're interested in
data_array = list(map(lambda r: tuple(r[k] for k in selected_columns), df.to_dict("record")))

# create distribution from the samples with uniform distribution
dist_census = dit.Distribution(data_array, [1. / df.shape[0] ] * df.shape[0])
# set variable aliases to the discribution
dist_census.set_rv_names("".join(rvs_names))

def plot(data):

    # definte the size of figure
    plt.figure(figsize=(6, 0.5*len(data)))

    labels = (", ".join(map(lambda p: "%s (%s)" % p, zip(data['variables'], ['Y', 'Z']))))

    # suffix each variable name with an alias {Y, Z}
    mm = []
    for m in metric_keys:
        mm.append((data['metrics'][m]/data['metrics']['mi']))
        # extract corresponding metric from elements in array



    left = np.array([0]*len(metric_keys))
    # plotting
    for i, kk in enumerate(metric_keys):
        plt.barh(
            labels, mm[i], align='center', height=.5,left=left, label=latex_labels[i], color=colors[i],
        )
        left = left + mm[i]
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.xlabel('Mutual information: '+str(data['metrics']['mi']))
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
decomp_S_EG = information_decomposition(dist_census, 'S', 'HE')
decomp_S_ER = information_decomposition(dist_census, 'S', 'HA')
decomp_S_RO = information_decomposition(dist_census, 'S', 'HR')
decomp_S_AG = information_decomposition(dist_census, 'S', 'HG')

# didn't converge, should we remove it?
# decomp_S_EO = compute_decomposition_from(dist_census, 'S', ['E', 'O'])
plot(decomp_S_EG)
plot(decomp_S_ER)
plot(decomp_S_RO)
plot(decomp_S_AG)
plot(decomp_S_HO)
pp.close()

def bar_graph(data,pp3):
    x_labels = ['I(S;YZ)', 'I(S;Y)', 'I(S;Z)', 'UI(S;Z\Y)', 'UI(S;Y\Z)']
    i_yz = data['metrics']['mi']
    i_y = data['metrics']['si']+data['metrics']['ui_0']
    i_z = data['metrics']['si']+data['metrics']['ui_1']
    print(data['metrics']['ci'])
    values = (round(i_yz,5),round(i_y,5),round(i_z,5),round(data['metrics']['ui_1'],5),round(data['metrics']['ui_0'],5))
    ind = np.arange(len(x_labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.set_title((", ".join(map(lambda p: "%s (%s)" % p, zip(data['variables'], ['Y', 'Z'])))))
    rects = ax.bar(ind - width/2, values, width,color='SkyBlue')
    ax.set_xticks(ind)
    ax.set_xticklabels(x_labels)

    #Auto labelling values above each bar
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    fig.tight_layout()
    plt.savefig(pp3, format='pdf')
def plot_values():

    interval_lengths = [5,10,20,40]
    for i in interval_lengths:
        temp = 'bins'+str(i)+'.pdf'
        print(temp)
        pp2 = PdfPages(temp)
        intervals = np.linspace(0,40,(40/i +1))
        df['hours-per-week-bins'] = pd.cut(df['hours-per-week'],bins = intervals)
        df['hours-per-week-bins'] = df['hours-per-week-bins'].cat.add_categories('>40').fillna('>40').astype(str)
        selected_columns = [
            'income',
            'education',
            'sex',
            'race',
            'occupation',
            'age-group',
            'hours-per-week-bins'
        ]


        # take all samples with attributes that we're interested in
        data_array = list(map(lambda r: tuple(r[k] for k in selected_columns), df.to_dict("record")))

        # create distribution from the samples with uniform distribution
        dist_census = dit.Distribution(data_array, [1. / df.shape[0] ] * df.shape[0])
        # set variable aliases to the discribution
        dist_census.set_rv_names("".join(rvs_names))
        decomp_S_HA = information_decomposition(dist_census, 'S', 'HA')
        decomp_S_HE = information_decomposition(dist_census, 'S', 'HE')
        decomp_S_HR = information_decomposition(dist_census, 'S', 'HR')
        decomp_S_HG = information_decomposition(dist_census, 'S', 'HG')
        decomp_S_HO = information_decomposition(dist_census, 'S', 'HO')
        bar_graph(decomp_S_HA,pp2)
        bar_graph(decomp_S_HE,pp2)
        bar_graph(decomp_S_HR,pp2)
        bar_graph(decomp_S_HG,pp2)
        bar_graph(decomp_S_HO,pp2)
        pp2.close()
plot_values()
