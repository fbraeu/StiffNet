#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
import torchvision.transforms as tt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import import_ipynb
from utils import *

from tqdm.notebook import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

best_val_loss = -1
n_iter = 0


# In[2]:


home = os.path.expanduser("~")

test = False
pretrained = False
weight_decay = 1.0e-5
print_freq = 1

output_dir  = home+"/workspace/StiffnessClassification/Classification/Output/"
arch        = "StiffClassification"
solver      = "adam"
start_epoch = 0
epochs      = 100
batch_size  = 14
lr          = 0.0005
save_path = '{}/{}_{}_{}epochs_b{}_lr{}'.format(output_dir,
                                                arch,
                                                solver,
                                                epochs,
                                                batch_size,
                                                lr)

print('=> will save everything to {}'.format(save_path))
if not os.path.exists(save_path):
    os.makedirs(save_path)
        
train_writer = SummaryWriter(os.path.join(save_path,'train'))
val_writer = SummaryWriter(os.path.join(save_path,'val'))


# In[3]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    #print(f"Moved to {device}")
    return data.to(device, non_blocking=True)

#Pick GPU if available, else CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


# In[4]:


def get_stats_channels(path="./", batch_size=12):
    """
    Create two tuples with mean and std for each RGB channel of the dataset
    """
    data = datasets.ImageFolder(path, tt.Compose([tt.ToTensor()]))
    loader = DataLoader(data, batch_size, num_workers=8, pin_memory=True)

    nimages = 0
    mean = 0.
    std = 0.
    for batch, _ in tqdm(loader):
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        std += batch.std(2).sum(0)


    mean /= nimages
    std /= nimages

    return mean, std


# normalization_stats = get_stats_channels("./Preprocessing/data_processed/Train")
# normalization_stats = (0.1587, 0.1587, 0.1587), (0.1353, 0.1353, 0.1353)
# normalization_stats = (0.1587, 0.1587, 0.1587), (0.1353, 0.1353, 0.1353)


# In[5]:


# random nonlinear intensity shift: img = -a + (1+a+b)*img^p ((min p, max p), (min a, max a), (min b, max b))
class AddNonlinIntensShift():
    def __init__(self, p=(0.6, 1.4), a=(0.0, 0.1), b=(0.0, 0.1)):
        self.p_min = p[0]
        self.p_max = p[1]
        self.a_min = a[0]
        self.a_max = a[1]
        self.b_min = b[0]
        self.b_max = b[1]
        
    def __call__(self, tensor):
        p = np.random.uniform(self.p_min, self.p_max)
        a = np.random.uniform(self.a_min, self.a_max)
        b = np.random.uniform(self.b_min, self.b_max)
        tensor = -a + (1+a+b)*torch.pow(tensor, p)
        tensor = torch.clip(tensor, 0, 1)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(p=({},{}), a=({},{}), b=({},{}))'.format(self.p_min, self.p_max, self.a_min, self.a_max, self.b_min,self.b_max)

# multiplicative gaussian speckle noise (out = I * n*I with n Gaussian distribution with expectation mean and standard deviation std)
class AddGaussSpeckleNoise():
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())*self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)
    
    
data_transforms = {
    'Train': tt.Compose([tt.ToTensor(), 
                         tt.RandomHorizontalFlip(p=0.5),
                         tt.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1)),
                         AddNonlinIntensShift(p=(0.6, 1.4), a=(0.0, 0.1), b=(0.0, 0.1)),
                         AddGaussSpeckleNoise(mean=0.0, std=0.1)]),
    'Valid': tt.Compose([tt.ToTensor()]),
    'Test': tt.Compose([tt.ToTensor()])
}

train_data = datasets.ImageFolder("./Preprocessing/data_processed/Train/", 
                                  transform=data_transforms["Train"])
valid_data = datasets.ImageFolder("./Preprocessing/data_processed/Val/", 
                                  transform=data_transforms["Valid"])
if test:
    test_data = datasets.ImageFolder("./Preprocessing/data_processed/Test/",
                                     transform=data_transforms["Test"])


# In[6]:


def get_label_data():
    batch_size = 10

    # Train dataset
    train_loader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True)
    train_labels = []
    for _, tr_lbls in tqdm(train_loader):
        train_labels.extend(tr_lbls.numpy().tolist())

    out = np.array(train_labels)
    unique, counts = np.unique(out,return_counts=True)

    data_train = {"Class": ["HighDef", "LowDef"],
                  "Count": [counts[0], counts[1]]
                  }

    df_train = pd.DataFrame(data_train, columns=["Class", "Count"])


    # Validation dataset
    valid_loader = DataLoader(valid_data, batch_size, num_workers=8, pin_memory=True)
    valid_labels = []
    for _, val_lbls in tqdm(valid_loader):
        valid_labels.extend(val_lbls.numpy().tolist())

    out = np.array(valid_labels)
    unique, counts = np.unique(out,return_counts=True)

    # Creating our own dataframe
    data_val = {"Class": ["HighDef", "LowDef"],
                "Count": [counts[0], counts[1]]
                }

    # Now convert this dicnary type data into a pandas dataframe
    # specifying what are the column names
    df_val = pd.DataFrame(data_val, columns=["Class", "Count"])
    
    if test:
        # Test dataset
        test_loader = DataLoader(test_data, batch_size, num_workers=8, pin_memory=True)
        test_labels = []
        for _, test_lbls in tqdm(test_loader):
            test_labels.extend(test_lbls.numpy().tolist())

        out = np.array(test_labels)
        unique, counts = np.unique(out,return_counts=True)

        # Creating our own dataframe
        data_test = {"Class": ["HighDef", "LowDef"],
                      "Count": [counts[0], counts[1]]
                     }

        # Now convert this dicnary type data into a pandas dataframe
        # specifying what are the column names
        df_test = pd.DataFrame(data_test, columns=["Class", "Count"])
    else:
        df_test = df_val # dummy test dataframe
    
    return df_train, df_val, df_test


df_train, df_val, df_test = get_label_data()

if test:
    # Bar plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(5,8), sharex=True)
    df_train.plot(x="Class", y="Count", ax=ax1, kind="bar")
    ax1.set(title="Train label")

    df_val.plot(x="Class", y="Count", ax=ax2, kind="bar")
    ax2.set(title="Validation labels")

    df_test.plot(x="Class", y="Count", ax=ax3, kind="bar")
    ax3.set(title="Test labels")

    plt.show()
else:
    # Bar plots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5,8), sharex=True)
    df_train.plot(x="Class", y="Count", ax=ax1, kind="bar")
    ax1.set(title="Train label")

    df_val.plot(x="Class", y="Count", ax=ax2, kind="bar")
    ax2.set(title="Validation labels")

    plt.show()



# Calculate cross entropy weights for an imbalanced training set
nr_highdef = df_train.loc[df_train.index[df_train["Class"]=="HighDef"][0],"Count"]
nr_lowdef = df_train.loc[df_train.index[df_train["Class"]=="LowDef"][0],"Count"]

if nr_highdef > nr_lowdef:
    cross_entropy_weights = [1.0, nr_highdef/nr_lowdef]
else:
    cross_entropy_weights = [nr_lowdef/nr_highdef, 1.0]
    
print("CrossEntropy Weights:[HighDef, LowDef] {}".format(cross_entropy_weights))


# In[7]:


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[8]:


train_dl = DataLoader(train_data, batch_size=batch_size, 
                      shuffle=True, pin_memory=True, num_workers=4)
valid_dl = DataLoader(valid_data, batch_size=batch_size, 
                      shuffle=True, pin_memory=True, num_workers=4)
if test:
    test_dl = DataLoader(test_data, batch_size=batch_size, pin_memory=True, num_workers=4)


# In[9]:


train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
if test:
    test_dl = DeviceDataLoader(test_dl, device)


# In[10]:


#================================================================================
class TransferResnet(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, classes=2, cross_entropy_weights=[1.0, 1.0]):
        
        super().__init__()
        
        # Initialize weights for cross entropy loss
        if torch.is_tensor(cross_entropy_weights):
            self.weights = cross_entropy_weights
        else:
            self.weights = to_device(torch.as_tensor(cross_entropy_weights).float(), device)
        
        # Use a pretrained model
        self.network = models.resnet34(pretrained=False)
        
        #self.network.avgpool = AdaptiveConcatPool2d()
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                        nn.ReLU(),  
                                        nn.Dropout(0.50), 
                                        nn.Linear(128,classes))
    
    def forward(self, xb):
        out = self.network(xb)
        return out

    def feed_to_network(self, batch):
        images, labels = batch 
        out = self(images)
        # Balanced validation set but unbalanced training set
        if self.training:
            loss = F.cross_entropy(out, labels, self.weights)
        else:
            loss = F.cross_entropy(out, labels, to_device(torch.as_tensor([1.0, 1.0]).float(), device))
        #Don't pass the softmax to the cross entropy
        out = F.softmax(out, dim=1)

        return loss, out


# In[11]:


def StiffNet(data=None):

    model = TransferResnet(classes=2, cross_entropy_weights=cross_entropy_weights)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model



# create model
if pretrained:
    network_data = torch.load(pretrained)
    print('=> using pre-trained model')
else:
    network_data = None
    print('creating model')

model = StiffNet(network_data)
model = to_device(model, device)


# In[12]:


assert(solver in ['adam', 'sgd'])
print('=> setting {} solver'.format(solver))

param_groups = [{"params": model.network.fc.parameters(), "lr": lr},
                {"params": model.network.layer4.parameters(), "lr": lr/2.5},
                {"params": model.network.layer3.parameters(), "lr": lr/5},
                {"params": model.network.layer2.parameters(), "lr": lr/10},
                {"params": model.network.layer1.parameters(), "lr": lr/100}]

if solver == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif solver == 'sgd':
    optimizer = optim.SGD(param_groups, lr, weight_decay=weight_decay)


# In[13]:


def get_scores(labels, prediction, loss=None):
    "Return classification scores"
    accuracy = accuracy_score(labels, prediction) 
    f1 = f1_score(labels, prediction, 
                  average='weighted', zero_division=0)
    precision = precision_score(labels, prediction, 
                                average='weighted', zero_division=0)
    recall = recall_score(labels, prediction, 
                          average='weighted', zero_division=0)
    if loss:
        return [accuracy, f1, precision, recall, loss]
    else: 
        return [accuracy, f1, precision, recall]


# In[14]:


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()

    train_labels = []
    train_predictions = []
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # compute output and loss
        loss, out = model.feed_to_network(batch)
        train_predictions += torch.argmax(out, dim=1).tolist()
        train_labels += batch[1].tolist()
        
        # record cross entropy loss
        losses.update(loss.item(), out.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t CrossEntropy Loss {5}'
                  .format(epoch, i, len(train_loader), batch_time,
                          data_time, losses))
            
        n_iter += 1
    
    return train_labels, train_predictions, losses.avg


# In[15]:


def validate(val_loader, model, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    val_labels = []
    val_predictions = []
    for i, batch in enumerate(val_loader):
        
        # compute output and loss
        loss, out = model.feed_to_network(batch)
        val_predictions += torch.argmax(out, dim=1).tolist()
        val_labels += batch[1].tolist()
                
        # record CrossEntropy Loss
        losses.update(loss.item(), out.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t Time {2}\t CrossEntropy Loss {3}'
                  .format(i, len(val_loader), batch_time, losses))       

    print(' * CrossEntropy Loss {:.3f}'.format(losses.avg))

    return val_labels, val_predictions, losses.avg


# In[ ]:


for epoch in range(start_epoch, epochs):

    # train for one epoch
    train_labels, train_predictions, epoch_train_loss = train(train_dl, model, optimizer, epoch, train_writer)
    
#     print(train_labels)
#     print(train_predictions)
    
    train_metrics = get_scores(train_labels, train_predictions)
    train_writer.add_scalar('Epoch Loss', epoch_train_loss, epoch)
    train_writer.add_scalar('Accuracy', train_metrics[0], epoch)
    train_writer.add_scalar('F1', train_metrics[1], epoch)
    train_writer.add_scalar('Precision', train_metrics[2], epoch)
    train_writer.add_scalar('Recall', train_metrics[3], epoch)

    # evaluate on validation dataset
    with torch.no_grad():
        val_labels, val_predictions, epoch_val_loss = validate(valid_dl, model, epoch)
    
#     print(val_labels)
#     print(val_predictions)
    
    val_metrics = get_scores(val_labels, val_predictions)
    val_writer.add_scalar('Epoch Loss', epoch_val_loss, epoch)
    val_writer.add_scalar('Accuracy', val_metrics[0], epoch)
    val_writer.add_scalar('F1', val_metrics[1], epoch)
    val_writer.add_scalar('Precision', val_metrics[2], epoch)
    val_writer.add_scalar('Recall', val_metrics[3], epoch)

    if best_val_loss < 0:
        best_val_loss = epoch_val_loss

    is_best = epoch_val_loss < best_val_loss
    best_val_loss = min(epoch_val_loss, best_val_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': arch,
        'state_dict': model.state_dict(),
        'best_EPE': best_val_loss,
    }, is_best, save_path)


# In[ ]:




