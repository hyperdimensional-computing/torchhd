import time
start_time = time.time()

# The following two lines are only needed because of this repository organization
import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from hdc import functional
from hdc import embeddings
import torchmetrics
from torch.utils.data import RandomSampler, Subset
from hdc.datasets.EMG_based_hand_gesture import EMG_based_hand_gesture

device = torch.device("cuda:2")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_LEVELS = 21
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
WINDOW = 256
N_GRAM_SIZE = 4
DOWNSAMPLE = 5

class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.iMch1 = embeddings.Random(WINDOW, DIMENSIONS)
        self.iMch2 = embeddings.Random(WINDOW, DIMENSIONS)
        self.iMch3 = embeddings.Random(WINDOW, DIMENSIONS)
        self.iMch4 = embeddings.Random(WINDOW, DIMENSIONS)

        self.CiMch1 = embeddings.Level(NUM_LEVELS, DIMENSIONS, high=20)
        self.CiMch2 = embeddings.Level(NUM_LEVELS, DIMENSIONS, high=20)
        self.CiMch3 = embeddings.Level(NUM_LEVELS, DIMENSIONS, high=20)
        self.CiMch4 = embeddings.Level(NUM_LEVELS, DIMENSIONS, high=20)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):

        samples = np.arange(0, WINDOW, int(WINDOW/DOWNSAMPLE))
        x = x[:,samples,:]

        sample_hv_ch1 = functional.bind(self.iMch1(x[:,:,0]), self.CiMch1(x[:,:,0]))
        sample_hv_ch2 = functional.bind(self.iMch2(x[:,:,1]), self.CiMch2(x[:,:,1]))
        sample_hv_ch3 = functional.bind(self.iMch3(x[:,:,2]), self.CiMch3(x[:,:,2]))
        sample_hv_ch4 = functional.bind(self.iMch4(x[:,:,3]), self.CiMch4(x[:,:,3]))
        sample_hv = torch.stack([sample_hv_ch1,sample_hv_ch2,sample_hv_ch3,sample_hv_ch4])
        sample_hv = functional.batch_bundle(sample_hv, dim=(0, 1))


        for i in range(len(samples) - N_GRAM_SIZE + 1):
            n_gram_hv = functional.identity_hv(1, DIMENSIONS, dtype=torch.float, device=device)

            for n in range(N_GRAM_SIZE):
                n_gram_hv = functional.permute(n_gram_hv)
                n_gram_hv = functional.bind(n_gram_hv, sample_hv[i+n])

            sample_hv = functional.bundle(sample_hv, n_gram_hv)
        sample_hv = functional.batch_bundle(sample_hv)
        return functional.hard_quantize(sample_hv)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


def experiment(subjects=[0]):
    print("List of subjects " + str(subjects))
    train_ds = EMG_based_hand_gesture("../data", download=True, subjects=subjects)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_size = int(len(train_ld)*0.3)

    sample_ds = Subset(train_ds, random.sample(range(len(train_ld)), test_size))
    sample_sampler = RandomSampler(sample_ds)
    test_ld = torch.utils.data.DataLoader(train_ds, sampler=sample_sampler, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(len(train_ds.classes), train_ds[0][0].size(-1))
    model = model.to(device)

    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = model.encode(samples)
            model.classify.weight[labels] += samples_hv

        model.classify.weight[:] = F.normalize(model.classify.weight)

    accuracy = torchmetrics.Accuracy()

    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)

            outputs = model(samples)
            predictions = torch.argmax(outputs, dim=0, keepdim=True)
            accuracy.update(labels, predictions.cpu())

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")

for i in range(5):
    experiment([i])

print("Duration", time.time() - start_time)
