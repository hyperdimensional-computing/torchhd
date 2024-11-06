# A partial implementation of https://arxiv.org/abs/2109.02157 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Note: this example requires the napkinXC library: https://napkinxc.readthedocs.io/
from napkinxc.datasets import load_dataset
from napkinxc.measures import precision_at_k

from tqdm import tqdm
import torchhd
from torchhd import embeddings, HRRTensor
import torchhd.tensors
from scipy.sparse import  vstack, lil_matrix
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


DIMENSIONS = 400 
NUMBER_OF_EPOCHS = 1
BATCH_SIZE = 1  
DATASET_NAME =  "eurlex-4k" # tested on "eurlex-4k", and "Wiki10-31K"
FC_LAYER_SIZE = 512

    
def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy csr matrix to pytorch sparse tensor
    """
    data_batch, targets_batch = zip(*batch)

    data_batch = vstack(data_batch).tocoo()
    data_batch = torch.sparse_coo_tensor(np.array(data_batch.nonzero()), data_batch.data, data_batch.shape)

    targets_batch = torch.stack(targets_batch)

    return data_batch, targets_batch

class multilabel_dataset(Dataset):
    def __init__(self,x,y,n_classes) -> None:
        self.x = x
        self.y = y
        self.n_classes = n_classes
        
    
    # Define the length of the dataset.
    def __len__(self):
        return self.x.shape[0]
    
    # Return a single sample from the dataset.
    def __getitem__(self, idx):
        labels = torch.zeros(self.n_classes, dtype=torch.int64)
        labels[self.y[idx]] = 1.0
        return self.x[idx], labels


X_train, Y_train = load_dataset(DATASET_NAME, "train", verbose=True)
X_test, Y_test = load_dataset(DATASET_NAME, "test", verbose=True)


if DATASET_NAME == "Wiki10-31K": # Because of this issue https://github.com/mwydmuch/napkinXC/issues/18
    X_train = lil_matrix(X_train[:,:-1])
    
N_features = X_train.shape[1]
N_classes   = max(max(classes) for classes in Y_train if classes != []) + 1

train_dataset = multilabel_dataset(X_train,Y_train,N_classes)
train_dataloader = DataLoader(train_dataset,BATCH_SIZE, collate_fn=sparse_batch_collate) 
test_dataset  = multilabel_dataset(X_test,Y_test,N_classes)
test_dataloader = DataLoader(test_dataset,collate_fn=sparse_batch_collate) 


print("Traning on \033[1m {} \033[0m. It has {} features, and {} classes."
      .format(DATASET_NAME,N_features,N_classes))


# Fully Connected model for the baseline comparision 
class FC(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FC, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.fc_layer_size = FC_LAYER_SIZE
        
        # Network Layers
        self.fc1 = nn.Linear(self.num_features, self.fc_layer_size)
        self.fc2 = nn.Linear(self.fc_layer_size, self.fc_layer_size )
        self.olayer = nn.Linear(self.fc_layer_size, self.num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.olayer(x)
        return x

    def pred(self, out,threshold=0.5):
        y = F.sigmoid(out)
        v,i = y.sort(descending=True)
        ids = i[v>=threshold]
        ids = ids.tolist()     
        return ids
    
    def loss(self,out,target):
        loss = nn.BCEWithLogitsLoss()(out, target.type(torch.float64))
        return loss
    
# Modified version of FC model that returns an HRRTensor with dim << output of the FC model. 
# It makes the model to have fewer parameters  
class FCHRR(nn.Module):
    def __init__(self, num_features, num_classes,dim):
        super(FCHRR, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.fc_layer_size = FC_LAYER_SIZE
        self.dim = dim
        
        self.classes_vec = embeddings.Random(N_classes, dim,vsa="HRR")
        n_vec, p_vec = torchhd.HRRTensor.random(2,dim)
        self.register_buffer("n_vec", n_vec)
        self.register_buffer("p_vec", p_vec)
        
        # Network Layers
        self.fc1 = nn.Linear(self.num_features, self.fc_layer_size)
        self.fc2 = nn.Linear(self.fc_layer_size, self.fc_layer_size )
        self.olayer = nn.Linear(self.fc_layer_size, dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.olayer(x)
        return x.as_subclass(HRRTensor)

    def pred(self, out,threshold=0.1):
        
        tmp_positive = self.p_vec.exact_inverse().bind(out)
        sims = tmp_positive.cosine_similarity(self.classes_vec.weight)
        
        v,i = sims.sort(descending=True)
        ids = i[v>=threshold]
        ids = ids.tolist()     
        
        return ids
    
    def loss(self,out,target):
        
        loss = torch.tensor(0, dtype=torch.float32,device=device)
        
        tmp_positives  = self.p_vec.exact_inverse().bind(out)
        tmp_negatives = self.n_vec.exact_inverse().bind(out)
        for i in range(target.shape[0]):
            
            cp = self.classes_vec.weight[target[i]==1,:]
            
            j_p = (1 - tmp_positives[i].cosine_similarity(cp)).sum()
            j_n = tmp_negatives[i].cosine_similarity(cp.multibundle())
            
            loss += j_p + j_n
        
        loss /= target.shape[0]
                
        return loss
    


hrr_model = FCHRR(N_features,N_classes,DIMENSIONS)
hrr_model = hrr_model.to(device)

baseline_model = FC(N_features,N_classes)
baseline_model = baseline_model.to(device)


for model_name, model in {"HRR-FC":hrr_model,"FC":baseline_model}.items():
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    model.train()
    for epoch in tqdm(range(1,NUMBER_OF_EPOCHS + 1), desc=f"{model_name} epochs",leave=False):
        
        for samples, labels in tqdm(train_dataloader, desc="Training",leave=False):
            samples = samples.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(samples)
            loss = model.loss(out, labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step()

    Y_pred = []
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_dataloader,desc="Validating",leave=False):
            data, target = data.to(device).float(), target.to(device)
            out = model(data) 
            ids = model.pred(out)
            Y_pred.append(ids)

    # Calculating the P@1 metric 
    p_at_1 = precision_at_k(Y_test, Y_pred, k=1)[0]
    print("Result of {} model ---->  P@1 = {}".format(model_name, p_at_1))