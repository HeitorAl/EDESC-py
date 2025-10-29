from __future__ import division, print_function

import numpy as np
from pickleshare import pickle
import torch
from torch.utils.data import Dataset, dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, models
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import h5py
import scipy.io
from sklearn import preprocessing
import numpy as np
import os
import pickle
import tensorflow as tf

def load_reuters_data(path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        # make_reuters_data(data_path)
        print(('reutersidf saved to ' + path))
    data = np.load(os.path.join(path, 'reutersidf10k.npy'),allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float32')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y

def load_mnist(dataset='mnist'):
    if dataset == 'mnist':
        train = datasets.MNIST('./data/mnist', train=True, download=True)
        test = datasets.MNIST('./data/mnist', train=False, download=True)
    elif dataset == 'fashion_mnist':
        train = datasets.FashionMNIST('./data/fmnist', train=True, download=True)
        test = datasets.FashionMNIST('./data/fmnist', train=False, download=True)

    x_train, y_train = train.data.numpy(), train.targets.numpy()
    x_test, y_test = test.data.numpy(), test.targets.numpy()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    x = x.reshape((x.shape[0], -1)).astype('float32') / 255.0
    y = y.astype('int64')

    x -= x.mean(axis=0)

    print((dataset.upper(), 'samples', x.shape))
    return x, y

def load_cifar10(path='./data/cifar10'):
    transform = transforms.Compose([
        transforms.Resize(224),  # redimensiona
        transforms.ToTensor(),   # converte para tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = models.resnet50(weights='IMAGENET1K_V1')
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove a última camada (classificação)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = []
    y = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # achata a saída
            x.append(outputs.cpu().numpy())
            y.append(labels.numpy())

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # achata a saída
            x.append(outputs.cpu().numpy())
            y.append(labels.numpy())

    x = np.concatenate(x)
    y = np.concatenate(y)

    return x, y

def load_cifar100(path='./data/cifar100'):
    transform = transforms.Compose([
        transforms.Resize(224),  # redimensiona
        transforms.ToTensor(),   # converte para tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root=path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = models.resnet50(weights='IMAGENET1K_V1')
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove a última camada (classificação)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = []
    y = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # achata a saída
            x.append(outputs.cpu().numpy())
            y.append(labels.numpy())

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # achata a saída
            x.append(outputs.cpu().numpy())
            y.append(labels.numpy())

    x = np.concatenate(x)
    y = np.concatenate(y)

    return x, y

def load_stl10(path='./data/stl10'):
    transform = transforms.Compose([
        transforms.Resize(224),  # redimensiona
        transforms.ToTensor(),   # converte para tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.STL10(root=path, train=True, download=True, transform=transform)
    test_set = datasets.STL10(root=path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = models.resnet50(weights='IMAGENET1K_V1')
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove a última camada (classificação)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = []
    y = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # achata a saída
            x.append(outputs.cpu().numpy())
            y.append(labels.numpy())

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # achata a saída
            x.append(outputs.cpu().numpy())
            y.append(labels.numpy())

    x = np.concatenate(x)
    y = np.concatenate(y)

    return x, y

def load_csv(datapath='./data/bases', filename='ssf_wide_indices_3000_4_Oct_2025.csv'):
    import pandas as pd
    import os

    file_path = os.path.join(datapath, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    data = pd.read_csv(file_path)

    feature_cols = [
        c for c in data.columns
        if any(prefix in c for prefix in ['B', 'NDVI', 'EVI'])
    ]

    label_col = 'label'

    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found in the dataset.")
    
    y = data[label_col].astype('category').cat.codes.values
    x = data[feature_cols].astype(np.float32)
    x = x.fillna(x.mean()).values
    x = np.clip(x, -1, 1)

    print(f"Loaded {x.shape[0]} samples with {x.shape[1]} features and {len(np.unique(y))} classes from {filename}")
    print('Classes:', dict(enumerate(data[label_col].astype('category').cat.categories)))

    return x.astype(np.float32), y.astype(np.int32)

class LoadDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()

        if self.dataset_name == 'reuters':
            self.x, self.y = load_reuters_data()
        elif self.dataset_name == 'mnist':
            self.x, self.y = load_mnist(dataset='mnist')
        elif self.dataset_name == 'fashion_mnist':
            self.x, self.y = load_mnist(dataset='fashion_mnist')
        elif self.dataset_name == 'cifar10':
            self.x, self.y = load_cifar10()
        elif self.dataset_name == 'cifar100':
            self.x, self.y = load_cifar100()
        elif self.dataset_name == 'stl10':
            self.x, self.y = load_stl10()
        elif self.dataset_name == 'ssfreg':
            self.x, self.y = load_csv(filename='ssf_wide_indices_3000_reg_4_Oct_2025.csv')
        elif self.dataset_name == 'ssf':
            self.x, self.y = load_csv(filename='ssf_wide_indices_3000_4_Oct_2025.csv')
        elif self.dataset_name == 'pampa':
            self.x, self.y = load_csv(filename='pampa_wide_indices_4_Oct_2025.csv')
        elif self.dataset_name == 'cerrados':
           self.x, self.y = load_csv(filename='cerrados_wide_indices_4_Oct_2025.csv')
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))

def cluster_acc(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
