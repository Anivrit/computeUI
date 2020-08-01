#first no discretization, then discretizations, then play around with num samples
import numpy as np
import dit
import admUI
import random
import math
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
    i_y = si_SXY + uis[variables[to[0]]-1]
    i_z = si_SXY + uis[variables[to[1]]-1]
    return {
        "metrics": {
            "I(S;YZ)": i_S_XY,
            "SI(S;YZ)": si_SXY,
            "CI(S;YZ)": ci_SXY,
            "UI(S;Y\Z)": uis[variables[to[0]]-1],
            "UI(S;Z\Y)": uis[variables[to[1]]-1],
            "I(S;Y)": i_y,
            "I(S;Z)": i_z
        }
    }

#Commented section generates the covariance matrix. Currently have the covariance matrix fixed to find the error.
'''
def get_value():
    i = random.random()
    j = random.randint(0,1)
    if j == 1:
        return -i
    else:
        return i
finaldata = {}
finaldata['vals'] = [0,0,0]
def gen_matrix():
    a = get_value()
    b = get_value()
    c = get_value()
    finaldata['vals'][0] = a
    finaldata['vals'][1] = b
    finaldata['vals'][2] = c
    check = (2*a*b*c - a*a - b*b - c*c +1)
    if check >0 :
        covariance = [[1.0,a,c],[a,1.0,b],[c,b,1.0]]
        finaldata['cov'] = covariance
    else:
        gen_matrix()

gen_matrix()
s = finaldata['vals'][0]
x = finaldata['vals'][1]
y = finaldata['vals'][2]
matrix = finaldata['cov']
'''
matrix = [[1.0,0.5,0.5],[0.5,1.0,0.5],[0.5,0.5,1.0]]
mean = [0,0,0]
samples = np.random.multivariate_normal(mean, matrix, 100)
print(samples)
labels = ['S','X','Y']
df = pd.DataFrame(samples, columns = labels,dtype = float)
df['S'] = pd.qcut(df['S'],10)
df['X'] = pd.qcut(df['X'],10)
df['Y'] = pd.qcut(df['Y'],10)
data_array = list(map(lambda r: tuple(r[k] for k in labels), df.to_dict("record")))
distribution = dit.Distribution(data_array, [1. / df.shape[0] ] * df.shape[0])
rvsnames = ['S','X','Y']
distribution.set_rv_names("".join(rvsnames))
decomp = information_decomposition(distribution,'S','XY')

    def test_decomp():
    s = 0.5
    x = 0.5
    y = 0.5
    print(s)
    print(x)
    print(y)
    form_ixy = (math.log(1/(1-s*s),2))/2
    form_ixz = (math.log(1/(1-y*y),2))/2
    num = (1-x*x)
    denom = (2*s*x*y + 1 - (s*s)-(x*x)-(y*y))
    form_ixyz = (math.log(num/denom,2))/2
    ixy = decomp['metrics']['I(S;Y)']
    ixz = decomp['metrics']['I(S;Z)']
    ixyz = decomp['metrics']['I(S;YZ)']
    list = [form_ixy,form_ixz,form_ixyz,ixy,ixz,ixyz]
    return list

comparisons = test_decomp()
print(comparisons)
