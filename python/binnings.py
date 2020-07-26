import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

import dit
import admUI
import math

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
# define measurement metrics
metric_keys = ['si', 'ci', 'ui_0', 'ui_1']
data_vals = {}
df = pd.read_csv(data_url, names=column_names)
for k, v in zip(column_names, df.dtypes):
    if v == "object":
        df[k] = df[k].apply(lambda x: x.strip())
age_groups = ['<24', '24-35', '36-50', '>50']
df['age-group'] = df.age.apply(lambda x: age_groups[np.digitize(x, [24, 36, 51])])
#rvs_to_name = dict(zip(rvs_names, selected_columns))
interval_lengths = [5,10,20,40]

def information_decomposition(names,dist, src, to=""):
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
    i_y = si_SXY + uis[variables[to[0]]-1]
    i_z = si_SXY + uis[variables[to[1]]-1]
    return {
        "variables": tuple(map(lambda x: names[x], to)),
        "metrics": {
            "I(S;YZ)": i_S_XY,
            "SI(S;YZ)": si_SXY,
            "CI(S;YZ)": ci_SXY,
            "UI(S;Y\Z)": uis[variables[to[0]]-1] ,
            "UI(S;Z\Y)": uis[variables[to[1]]-1],
            "I(S;Y)": i_y,
            "I(S;Z)": i_z
        }
    }


def bar_graph(data):
    labels = ['I(S;YZ)', 'I(S;Y)', 'I(S;Z)', 'UI(S;Z\Y)', 'UI(S;Y\Z)']
    for j in labels:
        pdfname = j + '.pdf'
        pp = PdfPages(pdfname)
        counter = 0
        while counter<len(data[5]):
            values = []
            for l in interval_lengths:
                values.append(round(data[l][counter]['metrics'][j],5))
            ind = np.arange(len(interval_lengths))
            width = 0.35
            fig, ax = plt.subplots()
            ax.set_title((", ".join(map(lambda p: "%s (%s)" % p, zip(data[l][counter]['variables'], ['Y', 'Z'])))))
            rects = ax.bar(ind - width/2, values, width,color='SkyBlue')
            ax.set_xticks(ind)
            ax.set_xticklabels(interval_lengths)
            ax.set_xlabel('Bin Width')
            ax.set_ylabel(j)
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
            plt.savefig(pp, format='pdf')
            counter += 1
        pp.close()
def save_values():
        for j in interval_lengths:
            loc = 'hours-perk-week'+str(j)
            temp = np.linspace(0,40,(40//j +1))
            temp = temp.astype(int)
            print(loc)
            df[loc] = pd.cut(df['hours-per-week'],bins = temp)
            df[loc] = df[loc].cat.add_categories('>40').fillna('>40').astype(str)
            selected_columns = [
                'income',
                'education',
                'sex',
                'race',
                'occupation',
                'age-group',
                loc
            ]
            # take all samples with attributes that we're interested in
            data_array = list(map(lambda r: tuple(r[k] for k in selected_columns), df.to_dict("record")))

            # create distribution from the samples with uniform distribution
            dist_census = dit.Distribution(data_array, [1. / df.shape[0] ] * df.shape[0])
            # set variable aliases to the discribution
            dist_census.set_rv_names("".join(rvs_names))
            rvs_to_name = dict(zip(rvs_names, selected_columns))
            decomp_S_HA = information_decomposition(rvs_to_name,dist_census, 'S', 'HA',)
            decomp_S_HE = information_decomposition(rvs_to_name,dist_census, 'S', 'HE')
            decomp_S_HR = information_decomposition(rvs_to_name,dist_census, 'S', 'HR')
            decomp_S_HG = information_decomposition(rvs_to_name,dist_census, 'S', 'HG')
            decomp_S_HO = information_decomposition(rvs_to_name,dist_census, 'S', 'HO')
            data_vals[j] = [decomp_S_HA,decomp_S_HE,decomp_S_HR,decomp_S_HG,decomp_S_HO,]
        bar_graph(data_vals)

save_values()
