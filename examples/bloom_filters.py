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

D = 10000
num_hash_fun = 100
dens = num_hash_fun/D
N = 10000
simul = 10

hd = torch.full((D,N), -1)

for i in hd:
    perm = torch.randperm(D)[:num_hash_fun]
    i[perm] = 1

thr_limit = torch.tensor([20]*30 + [30]*20 + [40]*15 + [50]*15 + [60]*20)

f_range = torch.range(50,5050,50)
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
        rbf = functional.multiset(hd[ind,:])

        positive = torch.tensor(1.0)
        negative = torch.tensor(-1.0)
        rbf = torch.where(rbf > -F, positive, negative)
        rbf_ind = ((rbf != -1).nonzero())

        puncture = int(D*(0.001*(len(rbf_ind)/D)))
        p_ind = torch.randperm(N)[:puncture]

        rbf[p_ind] = -1
        dp = functional.bind(hd, rbf)
        rbf_tpr[j] = (sum(dp[ind,1] != 1)/F.item()).item()
        rbf_fpr[j] = ((sum(dp[ind,1] != 1)/(N-F.item())).item())

    print(rbf_tpr)
    print(rbf_fpr)

    current_rbf[int(F),0] = torch.mean(rbf_tpr)
    current_rbf[int(F),1] = torch.mean(rbf_fpr)
    current_rbf[int(F),2] = (current_rbf[int(i),0]+(1-current_rbf[int(F),1]))/2

    p1 = dens
    pdf_bins = binom.pmf(range(0,int(F)+1), int(F), p1)
    thr = list(range(0,thr_limit[int(i)]))
    p0_bf = torch.zeros(len(thr))

    for k in thr:
        p0_bf[k] = sum(pdf_bins[0:thr[k]+1])

    print(pdf_bins)
    bins2 = list(range(num_hash_fun+1))
    pdf_bins2 = torch.zeros(len(thr), len(bins2))
    for k in range(len(thr)):
        print(k)
        print(F*num_hash_fun)
        print(F*num_hash_fun)
        print((F*num_hash_fun - sum((D*list(range(k)))*pdf_bins[:k])))
        print(sum((D*list(range(k)))*pdf_bins[:k]))
        print(F*num_hash_fun)

    break
