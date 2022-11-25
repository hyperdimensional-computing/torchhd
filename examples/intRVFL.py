import torch
import torch.nn as nn
from tqdm import tqdm



# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics

import torchhd
from torchhd import embeddings
from torchhd.map import MAP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


# DK: The current code will not work for other values. Do we want to make it more generic?
BATCH_SIZE = 1  
NUM_FOLDS = 4

# DK: these parameters are found via grid search for intRVFL approach
RVFL_HYPERPARAMETERS = [
                ('Abalone',1450,32,15),
                ('Acute-Inflammation',50,0.000976562500000000,1),
                ('Acute-Nephritis',50,0.000976562500000000,1),
                ('Adult',1150,0.0625000000000000,3),
                ('Annealing',1150,0.0156250000000000,7),
                ('Arrhythmia',1400,0.000976562500000000,7),
                ('Audiology-Std',950,16,3),
                ('Balance-Scale',50,32,7),
                ('Balloons',50,0.000976562500000000,1),
                ('Bank',200,0.00195312500000000,7),
                ('Blood',50,16,7),
                ('Breast-Cancer',50,32,1),
                ('Breast-Cancer-Wisc',650,16,3),
                ('Breast-Cancer-Wisc-Diag',1500,2,3),
                ('Breast-Cancer-Wisc-Prog',1450,0.0156250000000000,3),
                ('Breast-Tissue',1300,0.125000000000000,1),
                ('Car',250,32,3),
                ('Cardiotocography-10clases',1350,0.000976562500000000,3),
                ('Cardiotocography-3clases',900,0.00781250000000000,15),
                ('Chess-Krvk',800,4,1),
                ('Chess-Krvkp',1350,0.0156250000000000,3),
                ('Congressional-Voting',100,32,15),
                ('Conn-Bench-Sonar-Mines-Rocks',1100,0.0156250000000000,3),
                ('Conn-Bench-Vowel-Deterding',1350,8,3),
                ('Connect-4',1100,0.500000000000000,3),
                ('Contrac',50,8,7),
                ('Credit-Approval',200,32,7),
                ('Cylinder-Bands',1100,0.000976562500000000,7),
                ('Dermatology',900,8,3),
                ('Echocardiogram',250,32,15),
                ('Ecoli',350,32,3),
                ('Energy-Y1',650,0.125000000000000,3),
                ('Energy-Y2',1000,0.0625000000000000,7),
                ('Fertility',150,32,7),
                ('Flags',900,32,15),
                ('Glass',1400,0.0312500000000000,3),
                ('Haberman-Survival',100,32,3),
                ('Hayes-Roth',50,16,1),
                ('Heart-Cleveland',50,32,15),
                ('Heart-Hungarian',50,16,15),
                ('Heart-Switzerland',50,8,15),
                ('Heart-Va',1350,0.125000000000000,15),
                ('Hepatitis',1300,0.0312500000000000,1),
                ('Hill-Valley',150,0.0156250000000000,1),    
                ('Horse-Colic',850,32,1),
                ('Ilpd-Indian-Liver',1200,0.250000000000000,7),
                ('Image-Segmentation',650,8,1),
                ('Ionosphere',1150,0.00195312500000000,1),
                ('Iris',50,4,3),
                ('Led-Display',50,0.000976562500000000,7),
                ('Lenses',50,0.0312500000000000,1),
                ('Letter',1500,32,1),
                ('Libras',1250,0.125000000000000,3),
                ('Low-Res-Spect',1400,8,7),
                ('Lung-Cancer',450,0.000976562500000000,1),
                ('Lymphography',1150,32,1),
                ('Magic',800,16,3),
                ('Mammographic',150,16,7),
                ('Miniboone',650,0.0625000000000000,15),
                ('Molec-Biol-Promoter',1250,32,1),
                ('Molec-Biol-Splice',1000,8,15),
                ('Monks-1',50,4,3),
                ('Monks-2',400,32,1),
                ('Monks-3',50,4,15),
                ('Mushroom',150,0.250000000000000,3),
                ('Musk-1',1300,0.00195312500000000,7),
                ('Musk-2',1150,0.00781250000000000,7),
                ('Nursery',1000,32,3),
                ('Oocytes_Merluccius_Nucleus_4d',1500,1,7),
                ('Oocytes_Merluccius_States_2f',1500,0.0625000000000000,7),
                ('Oocytes_Trisopterus_Nucleus_2f',1450,0.00390625000000000,3),
                ('Oocytes_Trisopterus_States_5b',1450,2,7),
                ('Optical',1100,32,7),
                ('Ozone',50,0.00390625000000000,1),
                ('Page-Blocks',800,0.00195312500000000,1),
                ('Parkinsons',1200,0.500000000000000,1),
                ('Pendigits',1500,0.125000000000000,1),
                ('Pima',50,32,1),
                ('Pittsburg-Bridges-Material',100,8,1),
                ('Pittsburg-Bridges-Rel-L',1200,0.500000000000000,1),
                ('Pittsburg-Bridges-Span',450,4,7),
                ('Pittsburg-Bridges-T-Or-D',1000,16,1),
                ('Pittsburg-Bridges-Type',50,32,7),
                ('Planning',50,32,1),
                ('Plant-Margin',1350,2,7),
                ('Plant-Shape',1450,0.250000000000000,3),
                ('Plant-Texture',1500,4,7),
                ('Post-Operative',50,32,15),
                ('Primary-Tumor',950,32,3),
                ('Ringnorm',1500,0.125000000000000,3),
                ('Seeds',550,32,1),
                ('Semeion',1400,32,15),
                ('Soybean',850,1,3),
                ('Spambase',1350,0.00781250000000000,15),
                ('Spect',50,32,1),
                ('Spectf',1100,0.250000000000000,15),
                ('Statlog-Australian-Credit',200,32,15),
                ('Statlog-German-Credit',500,32,15),
                ('Statlog-Heart',50,32,7),
                ('Statlog-Image',950,0.125000000000000,1),
                ('Statlog-Landsat',1500,16,3),
                ('Statlog-Shuttle',100,0.125000000000000,7),
                ('Statlog-Vehicle',1450,0.125000000000000,7),
                ('Steel-Plates',1500,0.00781250000000000,3),
                ('Synthetic-Control',1350,16,3),
                ('Teaching',400,32,3),
                ('Thyroid',300,0.00195312500000000,7),
                ('Tic-Tac-Toe',750,8,1),
                ('Titanic',50,0.000976562500000000,1),
                ('Trains',100,16,1),
                ('Twonorm',1100,0.00781250000000000,15),
                ('Vertebral-Column-2clases',250,32,3),
                ('Vertebral-Column-3clases',200,32,15),
                ('Wall-Following',1200,0.00390625000000000,3),
                ('Waveform',1400,8,7),
                ('Waveform-Noise',1300,0.000976562500000000,15),
                ('Wine',850,32,1),
                ('Wine-Quality-Red',1100,32,1),
                ('Wine-Quality-White',950,8,3),
                ('Yeast',1350,4,1),
                ('Zoo',400,8,7),
                ]


class intRVFL(nn.Module):
    #DK: Class description is missing
    def __init__(self, dimensions, lamb, kappa, num_classes, num_feat, ):
        super(intRVFL, self).__init__()
        
        self.dimensions = dimensions
        self.lamb = lamb
        self.kappa = kappa
        self.num_classes = num_classes
        self.num_feat = num_feat
        self.key = torchhd.random_hv(self.num_feat, self.dimensions, model=MAP)
        # DK: this will likely have to be double-checked once API for embeddigns is revised
        self.density_encoding = embeddings.Thermometer(self.dimensions+1,self.dimensions, low=0, high=1)
        self.classify = nn.Linear(self.dimensions, self.num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):
        sample_hv = MAP.bind(self.key, self.density_encoding(x))
        sample_hv = MAP.multibundle(sample_hv)
        # DK: consider making this function into the functinal.py - could be useful in other contexts 
        #Clipping function
        sample_hv[sample_hv>self.kappa]=self.kappa
        sample_hv[sample_hv<-self.kappa]=-self.kappa

        
        return sample_hv

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit
    
    #DK: not sure if it done nicely here since device is not inside the class
    #Gets readout matrix via ridge regression
    def fit(self, train_ld, min_val, max_val):
        # Get number of training samples 
        num_train = train_ld.dataset.targets.size(0)
        total_samples_hv = torch.zeros(num_train, self.dimensions, dtype=self.key.dtype, device=device,)
        labels_one_hot = torch.zeros(num_train, self.num_classes, dtype=self.key.dtype, device=device,)
        
        with torch.no_grad():
            count = 0
            for samples, labels in tqdm(train_ld, desc="Training"):
                
                samples = samples.to(device)
                labels = labels.to(device)
                
                labels_one_hot[count,labels] = 1 
                
                #DK: I think this needs to be somehow connected to transform (callable, optional) that we have for datasets
                #Normalize 
                samples = (samples-min_val)/(max_val-min_val)
                
                samples_hv = self.encode(samples)
                total_samples_hv[count,:] = samples_hv
                
                count += 1
                #model.classify.weight[labels] += samples_hv
                
            
            #Compute the readout matrix
            Wout = torch.t(labels_one_hot) @ total_samples_hv @ torch.linalg.pinv(torch.t(total_samples_hv) @ total_samples_hv + self.lamb*torch.diag(torch.var(total_samples_hv, 0)) )  
            self.classify.weight[:] = Wout

#ID of the dataset to fetch from RVFL_HYPERPARAMETERS.
#DK: Eventually, this should be for loop to cover all the datasets
dataset_id = 0

#Gets the current dataset
dataset = getattr(torchhd.datasets, RVFL_HYPERPARAMETERS[dataset_id][0])

# Get number of classes       
num_classes = len(dataset.classes)

#Fetch the hyperparameters for the corresponding dataset
dimensions = RVFL_HYPERPARAMETERS[dataset_id][1]
kappa = RVFL_HYPERPARAMETERS[dataset_id][2]
lamb = RVFL_HYPERPARAMETERS[dataset_id][3]

    
accuracy_dataset = 0.
# If no separate test dataset available - do 4-fold cross-validation
if not dataset.has_test:    
    for fold_id in range(NUM_FOLDS):
        #Set datasets for the current fold
        train_ds = dataset("../data", train=True, download=True, fold = fold_id)
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
           
        test_ds = dataset("../data", train=False, download=False, fold = fold_id)
        test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    
        # Get number of features
        num_feat = train_ds[0][0].size(-1)

        #Initialize model
        model = intRVFL(dimensions, lamb, kappa, num_classes, num_feat,)
        model = model.to(device)

        # Used to make min-max normalization of the data
        min_val = torch.min(train_ds.data, 0).values.to(device)
        max_val = torch.max(train_ds.data, 0).values.to(device)        
        
        #Obtain the readout matrix
        model.fit(train_ld, min_val, max_val)
    
    
        accuracy = torchmetrics.Accuracy()
        
        with torch.no_grad():
            for samples, labels in tqdm(test_ld, desc="Testing"):
                samples = samples.to(device)
        
                #Normalize 
                samples = (samples-min_val)/(max_val-min_val)
        
                outputs = model(samples)
                predictions = torch.argmax(outputs, dim=-1)
                accuracy.update(predictions.cpu(), labels)
    
    
        #print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
        accuracy_dataset +=accuracy.compute().item()        
    accuracy_dataset = accuracy_dataset/NUM_FOLDS
    
# Case of avaiable test set
# DK: this needs to be tested once such a dataset is included
else:     
    #Set datasets
    train_ds = dataset("../data", train=True, download=True)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
       
    test_ds = dataset("../data", train=False, download=False)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Get number of features
    num_feat = train_ds[0][0].size(-1)

    #Initialize model
    model = intRVFL(dimensions, lamb, kappa, num_classes, num_feat,)
    model = model.to(device)

    # Used to make min-max normalization of the data
    min_val = torch.min(train_ds.data, 0).values.to(device)
    max_val = torch.max(train_ds.data, 0).values.to(device)        
    
    #Obtain the readout matrix
    model.fit(train_ld, min_val, max_val)


    accuracy = torchmetrics.Accuracy()
    
    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)
    
            #Normalize 
            samples = (samples-min_val)/(max_val-min_val)
    
            outputs = model(samples)
            predictions = torch.argmax(outputs, dim=-1)
            accuracy.update(predictions.cpu(), labels)


    #print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
    accuracy_dataset =accuracy.compute().item()        

print()
print(f"Dataset {dataset_id} - {RVFL_HYPERPARAMETERS[dataset_id][0]}: average testing accuracy of {(accuracy_dataset* 100):.3f}%")
