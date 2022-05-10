# The following two lines are only needed because of this repository organization
import sys, os

sys.path.insert(1, os.path.realpath(os.path.pardir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics

from torchhd import functional
from torchhd import embeddings
from torchhd import structures

D = 10000
num_hash_fun = 100
dens = num_hash_fun/D
N = 10000
simul = 10

selection = torch.full((D,N), -1)

for i in selection:
    perm = torch.randperm(D)[:num_hash_fun]
    i[perm] = 1

thr_limit = torch.tensor([20]*30 + [30]*20 + [40]*15 + [50]*15 + [60]*20)

f_range = torch.range(0,5000,50)
current_rbf = torch.zeros(num_hash_fun, 3)

for i in f_range:
    rbf_tpr = torch.zeros(simul)
    rbf_fpr = torch.zeros(simul)
    for j in range(simul):
        rand_perm = torch.randperm(N)
        ind = rand_perm[:int(i)]
        ll_ind = rand_perm[int(i):]
        l_ind = torch.full(N, -1)
        l_ind[ind] = 1

