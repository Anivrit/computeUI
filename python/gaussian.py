#first no discretization, then discretizations, then play around with num samples
import numpy as np
import dit
import admUI
import random
import math
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.backends.backend_pdf import PdfPages


#Commented section generates the covariance matrix. Currently have the covariance matrix fixed to find the error.

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



#matrix = [[1.0,0.5,0.5],[0.5,1.0,0.5],[0.5,0.5,1.0]]
mean = [0,0,0]
samples = []
results = {}
bins = [10,20,30,40]
for b in bins:
    results[str(b)] = []
for i in range(5):
    gen_matrix()
    matrix = finaldata['cov']
    samples.append(np.random.multivariate_normal(mean, matrix, 10000))

values = ['IS:X','IS:Y','IS:XY']
labels = ['S','X','Y']
def information_decomposition(dist, numbins):
    temp = {}
    #s = 0.5
    #x = 0.5
    #y = 0.5
    s = finaldata['vals'][0]
    x = finaldata['vals'][1]
    y = finaldata['vals'][2]
    temp['barr_IS:X'] = round(((math.log(1/(1-s*s),2))/2),5)
    temp['barr_IS:Y'] = round(((math.log(1/(1-y*y),2))/2),5)
    num = (1-x*x)
    denom = (2*s*x*y + 1 - (s*s)-(x*x)-(y*y))
    temp['barr_IS:XY'] = round(((math.log(num/denom,2))/2),5)
    isx =  (dit.shannon.entropy(dist, 'S') - dit.shannon.conditional_entropy(dist, 'S', 'X'))
    temp['IS:X'] = round(isx,5)
    isy = (dit.shannon.entropy(dist, 'S') - dit.shannon.conditional_entropy(dist, 'S', 'Y'))
    temp['IS:Y'] =  round(isy,5)
    isxy = (dit.shannon.mutual_information(dist, 'S', 'XY'))
    temp['IS:XY'] = round(isxy,5)
    results[str(numbins)].append(temp)
def gen_data(bins,gen_samples):
    for sample in gen_samples:
        df = pd.DataFrame(sample, columns = labels,dtype = float)
        df['S'] = pd.qcut(df['S'],bins)
        df['X'] = pd.qcut(df['X'],bins)
        df['Y'] = pd.qcut(df['Y'],bins)
        data_array = list(map(lambda r: tuple(r[k] for k in labels), df.to_dict("record")))
        distribution = dit.Distribution(data_array, [1. / df.shape[0] ] * df.shape[0])
        distribution.set_rv_names('SXY')
        information_decomposition(distribution,bins)

def plot(data):
    for j in values:
        barrstring = 'barr_'+j
        pdfname = j + 'rngcov'+'.pdf'
        pp = PdfPages(pdfname)
        for k in range(5):
            y1 = []
            y2 = []
            for l in bins:
                y1.append(results[str(l)][k][j])
                y2.append(results[str(l)][k][barrstring])
            textstr = '\n'.join(map(str,matrix))
            fig, ax = plt.subplots()
            ax.set_title(j+" Trial:"+str(k+1))
            ax.plot(bins, y1, marker = '*', label = 'dit')
            ax.plot(bins, y2, marker = '*', label = 'barrett')
            for wx in zip(bins, y1):
                ax.annotate('(%s, %s)' % wx, xy=wx, textcoords='data')
            for yz in zip(bins, y2):
                ax.annotate('(%s, %s)' % yz, xy=yz, textcoords='data')
            ax.set_xlabel('Number of Bins')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            ax.text(0.95, 0.40, textstr, transform=ax.transAxes, fontsize=6,verticalalignment='top', bbox=props)
            plt.legend()
            fig.tight_layout()
            plt.savefig(pp, format='pdf')
        pp.close()
for i in bins:
    gen_data(i,samples)
plot(results)
