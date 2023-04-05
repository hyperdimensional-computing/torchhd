import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
import torch.utils.data as data
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
import copy
import warnings
import torchhd.datasets as ds
warnings.filterwarnings("ignore")
torch.manual_seed(19)
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIMENSIONS = 10000
DATASETS = [ds.PittsburgBridgesType, ds.AcuteInflammation, ds.AcuteNephritis, ds.Balloons,
            ds.BreastCancer, ds.BreastCancerWiscProg, ds.ConnBenchSonarMinesRocks,
            ds.Echocardiogram, ds.Fertility, ds.Hepatitis, ds.IlpdIndianLiver,
            ds.Ionosphere, ds.MolecBiolPromoter, ds.Monks1, ds.Monks2,
            ]

DATASETS = [ds.StatlogHeart]
#, ds.Spectf, ds.StatlogHeart, ds.Echocardiogram, ds.HorseColic, ds.IlpdIndianLiver, ds.CreditApproval]
#DATASETS = [ds.Spambase, ds.Titanic, ds.ConnBenchSonarMinesRocks, ds.CylinderBands, ds.HabermanSurvival]
ENCODINGS = []
HD_LEARN = ['add', 'add_online', 'add_online2', 'add_online_noise']
HD_LEARN = ['add','add_online', 'add_online2']
HD_LEARN = ['add_combined']
#HD_LEARN = ['add_combined']
METRICS = ['accuracy', 'average_similarity']

def create_min_max_normalize(min, max):
    def normalize(input):
        return torch.nan_to_num((input - min) / (max - min))
    return normalize


def normalize(w, eps=1e-12) -> None:
    """Transforms all the class prototype vectors into unit vectors.

    After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
    Training further after calling this method is not advised.
    """
    norms = w.norm(dim=1, keepdim=True)
    norms.clamp_(min=eps)
    w.div_(norms)


def load_train_test(dataset):
    d = dataset("../../data", download=False, train=True)
    num_classes = len(d.classes)

    train_size = int(0.7 * len(d))
    test_size = len(d) - train_size
    train, test = torch.utils.data.random_split(d, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    min_val = torch.min(train.dataset.data, 0).values.to(device)
    max_val = torch.max(train.dataset.data, 0).values.to(device)
    transform = create_min_max_normalize(min_val, max_val)
    train.transform = transform
    test.transform = transform

    train_loader = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    size = train[0][0].size(-1)
    return train_loader, test_loader, num_classes, size


class Encoder(nn.Module):
    def __init__(self, size):
        super(Encoder, self).__init__()
        self.embed = embeddings.Density(size, DIMENSIONS)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = self.embed(x).sign()
        return torchhd.hard_quantize(sample_hv)


def load_encoder(size):
    encode = Encoder(size)
    encode = encode.to(device)
    return encode


def load_model(num_classes):
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)
    return model


def hd_learn(model, encode, train_loader, test_loader, num_classes, learn):

    if 'accuracy' in METRICS:
        accuracy_test = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        accuracy_test2 = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    if 'recall' in METRICS:
        recall_test = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes)
    if 'precision' in METRICS:
        precision_test = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes)
    if 'f1' in METRICS:
        f1_test = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes)
    if 'loss' in METRICS:
        criterion = nn.CrossEntropyLoss()
        loss_test = []
    if 'average_similarity' in METRICS:
        sum_sim = 0
        sum_sim_diff = 0

    with torch.no_grad():
        for i in range(1):
            if 'accuracy' in METRICS:
                accuracy_train = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
                accuracy_train2 = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
            if 'recall' in METRICS:
                recall_train = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes)
            if 'precision' in METRICS:
                precision_train = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes)
            if 'f1' in METRICS:
                f1_train = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes)
            if 'loss' in METRICS:
                loss_train = []

            model.miss_predict = {}

            for samples, labels in tqdm(train_loader, desc="Training", disable=True):
                samples = samples.to(device)
                labels = labels.to(device)

                samples_hv = encode(samples)
                if learn == 'add':
                    logit = model.add(samples_hv, labels)
                if learn == 'add_online':
                    logit = model.add_online(samples_hv, labels)
                if learn == 'add_online2':
                    logit = model.add_online2(samples_hv, labels)
                if learn == 'add_online_noise':
                    logit = model.add_online_noise(samples_hv, labels)
                if learn == 'add_combined':
                    logit, logit_base = model.add_online_combined(samples_hv, labels)
                    pred_base = logit_base.argmax(1)

                pred = logit.argmax(1)

                if 'accuracy' in METRICS:
                    accuracy_train.update(pred.cpu(), labels)
                    if learn == 'add_combined':
                        accuracy_train2.update(pred_base.cpu(), labels)
                if 'recall' in METRICS:
                    recall_train.update(pred.cpu(), labels)
                if 'precision' in METRICS:
                    precision_train.update(pred.cpu(), labels)
                if 'f1' in METRICS:
                    f1_train.update(pred.cpu(), labels)
                if 'loss' in METRICS:
                    loss_train.append(criterion(logit, labels))
                if 'average_similarity' in METRICS:
                    sum_sim += torch.sum(logit)/2
                    sum_sim_diff += abs(logit[0][0] - logit[0][1])

            #model_test = copy.deepcopy(model)
            model_test = model
            model_test.normalize()
            print("Train", (model.train_accuracy.compute().item() * 100), (model.train_accuracy_base.compute().item() * 100))

            for samples, labels in tqdm(test_loader, desc="Testing", disable=True):
                samples = samples.to(device)

                samples_hv = encode(samples)
                combined = False
                if learn == 'add_combined':
                    combined = True
                    logit, logit_base = model_test.forward2(samples_hv, dot=True, combined=combined, test=True)
                    pred_base = logit_base.argmax(1)
                else:
                    logit = model_test(samples_hv, dot=True, combined=combined, test=True)
                pred = logit.argmax(1)


                if 'accuracy' in METRICS:
                    accuracy_test.update(pred.cpu(), labels)
                    if learn == 'add_combined':
                        accuracy_test2.update(pred_base.cpu(), labels)
                if 'recall' in METRICS:
                    recall_test.update(pred.cpu(), labels)
                if 'precision' in METRICS:
                    precision_test.update(pred.cpu(), labels)
                if 'f1' in METRICS:
                    f1_test.update(pred.cpu(), labels)
                if 'loss' in METRICS:
                    loss_test.append(criterion(logit, labels))
    if learn == 'add_combined':
        sim1 = abs(torchhd.cos_similarity(model.weight[0], model.weight[1])).item()
        sim2 = abs(torchhd.cos_similarity(model.weight_base[0], model.weight_base[1])).item()
        print(learn, (accuracy_test.compute().item() * 100), (accuracy_test2.compute().item() * 100),
              sim1, sim2, sim1 > sim2)
    else:
        print(learn, (accuracy_test.compute().item() * 100), sum_sim.item()/len(train_loader), sum_sim_diff/len(train_loader))


def evaluate():
    for dataset in DATASETS:
        print("DATASET", dataset)
        train_loader, test_loader, num_classes, size = load_train_test(dataset)
        encode = load_encoder(size)
        for learn in HD_LEARN:
            model = load_model(num_classes)
            hd_learn(model, encode, train_loader, test_loader, num_classes, learn)
        print()




evaluate()



