# The following two lines are only needed because of this repository organization
import os
import sys
import copy
sys.path.insert(1, os.path.realpath(os.path.pardir))
import torch
from scipy.stats import binom
# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import numpy as np
from torchhd import functional
import numpy as np
import warnings
warnings.filterwarnings("ignore")

D = 10000
num_hash_fun = 100
dens = num_hash_fun/D
N = 10000
simul = 100
BEST_ABF = []
CURRENT_BF = []
BEST_BF = []
hd = torch.full((D,N), 0, dtype=torch.long)


for i in hd:
    perm = torch.randperm(D)[:num_hash_fun]
    i[perm] = 1

thr_limit = torch.tensor([20]*30 + [30]*20 + [40]*15 + [50]*15 + [60]*20)

f_range = torch.range(50,5000,50)
current_rbf = torch.zeros(num_hash_fun, 3)

for i in range(len(f_range)):
    F = f_range[i]
    rbf_tpr = torch.zeros(simul)
    rbf_fpr = torch.zeros(simul)
    for j in range(simul):
        rand_perm = torch.randperm(N)
        ind = rand_perm[:int(F)]
        ll_ind = rand_perm[int(F):]
        # l_ind = torch.full((N,1), 1)
        rbf = torch.sum(hd[ind,:], dim=0)
        positive = torch.tensor(1.0)
        negative = torch.tensor(0.0)
        rbf = torch.where(rbf > 0, positive, negative)
        rbf_ind = ((rbf != 0).nonzero())

        puncture = int(D*(0.001*(len(rbf_ind)/D)))+1
        p_ind = torch.randperm(N)[:puncture]

        rbf[p_ind] = -1

        dp = torch.matmul(hd, rbf.long())

        rbf_tpr[j] = (sum(dp[ind] == num_hash_fun)/F.item()).item()
        rbf_fpr[j] = ((sum(dp[ll_ind] == num_hash_fun)/(N-F.item())).item())

    current_rbf[i,0] = torch.mean(rbf_tpr)
    current_rbf[i,1] = torch.mean(rbf_fpr)
    current_rbf[i,2] = (current_rbf[int(i),0]+(1-current_rbf[i,1]))/2

    p1 = dens
    pdf_bins = binom.pmf(range(0,int(F)+1), int(F), p1)
    thr = list(range(0,thr_limit[int(i)]))
    p0_bf = torch.zeros(len(thr))

    for k in thr:
        p0_bf[k] = sum(pdf_bins[0:thr[k]+1])

    bins2 = range(num_hash_fun+1)
    pdf_bins2 = torch.zeros(len(thr), len(bins2))
    pdf_bins3 = torch.zeros(len(thr), len(bins2))
    for k in range(len(thr)):
        l1 = [D*(x) for x in list(range(0,k+1))]
        exp = (F*num_hash_fun - sum([l1[i]*pdf_bins[:k+1][i] for i in range(len(l1))])) / (F*num_hash_fun)
        pdf_bins2[k] = torch.tensor(binom.pmf(bins2, num_hash_fun, exp))
        pdf_bins3[k] = torch.tensor(binom.pmf(bins2, num_hash_fun, (1-p0_bf[k])))

    tnfp = [ [] for _ in range(len(thr)) ]
    for j in range(len(thr)):
        for k in range(num_hash_fun):
            b1 = sum(pdf_bins2[j][k:])
            b2 = sum(pdf_bins3[j][k:])
            tnfp[j].append([b1, b2, (b1+(1-b2))/2])

    THR = torch.zeros(len(thr), 5)
    for k in range(len(thr)):
        col0 = [row[0] for row in tnfp[k]]
        index = [i for i, v in enumerate(col0) if v >= 0.9]
        val = [row[2] for row in tnfp[k]]
        ind = len([val[i] for i in index])
        THR[k][0] = ind-1
        THR[k][1:4] = torch.tensor(tnfp[k][ind-1])

    THR[:,4] = torch.tensor(thr)
    ind = torch.max(THR[:,3])
    BEST_ABF.append(THR[int(ind),:])
    CURRENT_BF.append(tnfp[0][-1])
    one_n_opt = round((D/int(F))*np.log(2))
    if one_n_opt == 0:
        one_n_opt = 1
    fpr = pow(1-np.exp(-(one_n_opt*int(F)/D)), one_n_opt)
    BEST_BF.append([one_n_opt, 1, fpr, (1+(1-fpr))/2])
print(BEST_BF)
print(BEST_ABF)
print(CURRENT_BF)
print(current_rbf)
