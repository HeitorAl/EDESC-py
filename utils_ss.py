from __future__ import division, print_function

import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, models
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.preprocessing import RobustScaler
import h5py
import scipy.io
from sklearn import preprocessing
import numpy as np
import os
import pickle
import tensorflow as tf
import pandas as pd


def load_reuters_data(path='./data/reuters'):
    import os
    file_path = os.path.join(path, 'reutersidf10k.npy')

    if not os.path.exists(file_path):
        print('Arquivo reutersidf10k.npy não encontrado.')
        print('Por favor, gere o arquivo primeiro ou verifique o caminho.')
        raise FileNotFoundError(f"Arquivo de features não encontrado em {file_path}")

    data = np.load(file_path, allow_pickle=True).item()
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float32')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y


def load_mnist(dataset='mnist', path=None):
    if dataset == 'mnist':
        data_path = path if path is not None else './data/mnist'
        file_name = 'mnist_features.npy'
    elif dataset == 'fashion_mnist':
        data_path = path if path is not None else './data/fmnist'
        file_name = 'fashion_mnist_features.npy'
    else:
        raise ValueError("Dataset não suportado (apenas mnist ou fashion_mnist)")

    file_path = os.path.join(data_path, file_name)

    if not os.path.exists(file_path):
        print(f"Criando features {dataset}... Isso é executado apenas uma vez.")
        os.makedirs(data_path, exist_ok=True)

        if dataset == 'mnist':
            train = datasets.MNIST(data_path, train=True, download=True)
            test = datasets.MNIST(data_path, train=False, download=True)
        elif dataset == 'fashion_mnist':
            train = datasets.FashionMNIST(data_path, train=True, download=True)
            test = datasets.FashionMNIST(data_path, train=False, download=True)

        x_train, y_train = train.data.numpy(), train.targets.numpy()
        x_test, y_test = test.data.numpy(), test.targets.numpy()

        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))

        x = x.reshape((x.shape[0], -1)).astype('float32') / 255.0
        y = y.astype('int64')

        x -= x.mean(axis=0)

        data = {'data': x, 'label': y}
        np.save(file_path, data)
        print(f"Features {dataset} salvas em {file_path}")

    data = np.load(file_path, allow_pickle=True).item()
    x = data['data']
    y = data['label']

    print((dataset.upper(), 'samples', x.shape))
    return x, y


@torch.no_grad()
def _create_image_features(dataset_name, root_path, file_path):
    print(f"Criando features {dataset_name}... Isso pode demorar um pouco.")
    os.makedirs(root_path, exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name == 'CIFAR10':
        train_set = datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        train_set = datasets.CIFAR100(root=root_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root=root_path, train=False, download=True, transform=transform)
    elif dataset_name == 'STL10':
        train_set = datasets.STL10(root=root_path, split='train', download=True, transform=transform)
        test_set = datasets.STL10(root=root_path, split='test', download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} não suportado por esta função.")

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    model = models.resnet50(weights='DEFAULT')
    model.fc = torch.nn.Identity()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img_list = []
    label_list = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        img_list.append(outputs.cpu())
        label_list.append(labels.cpu())

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        img_list.append(outputs.cpu())
        label_list.append(labels.cpu())

    x = torch.cat(img_list, dim=0).numpy()
    y = torch.cat(label_list, dim=0).numpy()

    data = {'data': x, 'label': y}
    np.save(file_path, data)
    print(f"Features {dataset_name} salvas em {file_path}")

def load_cifar10(path='./data/cifar10'):
    file_path = os.path.join(path, 'cifar10_features.npy')
    if not os.path.exists(file_path):
        _create_image_features('CIFAR10', path, file_path)
    data = np.load(file_path, allow_pickle=True).item()
    x, y = data['data'], data['label']
    print(('CIFAR10 samples', x.shape))
    return x, y

def load_cifar100(path='./data/cifar100'):
    file_path = os.path.join(path, 'cifar100_features.npy')
    if not os.path.exists(file_path):
        _create_image_features('CIFAR100', path, file_path)
    data = np.load(file_path, allow_pickle=True).item()
    x, y = data['data'], data['label']
    print(('CIFAR100 samples', x.shape))
    return x, y

def load_stl10(path='./data/stl10'):
    file_path = os.path.join(path, 'stl10_features.npy')
    if not os.path.exists(file_path):
        _create_image_features('STL10', path, file_path)
    data = np.load(file_path, allow_pickle=True).item()
    x, y = data['data'], data['label']
    print(('STL10 samples', x.shape))
    return x, y


# --- CARREGAMENTO DOS DATASETS DE SENSORIAMENTO REMOTO ---
# Idêntico ao utils.py (ver comentários lá para o porquê da normalização).
MODIS_FEATURE_RE = re.compile(r'^(ndvi|evi|nir|mir)\d{2}$', re.IGNORECASE)
SENTINEL_FEATURE_RE = re.compile(r'^(B\d{2}|B8A|NDVI|EVI)\.\d+$')
EXTREME_THRESHOLD = 1e5
CLIP_LOW_Q, CLIP_HIGH_Q = 0.01, 0.99
POST_SCALE_CLIP = 10.0

REMOTE_SENSING_FILES = {
    'ssf':               ('ssf.sentinel.raw.xlsx', 'sentinel'),
    'ssf_filtered':      ('ssf.sentinel.filtered.xlsx', 'sentinel'),
    'pampa':             ('pampa.sentinel.raw.xlsx', 'sentinel'),
    'pampa_filtered':    ('pampa.sentinel.filtered.xlsx', 'sentinel'),
    'cerrado_modis':     ('cerrado.i.modis.raw.xlsx', 'modis'),
    'cerrado_filtered':  ('cerrado.i.modis.filtered.xlsx', 'modis'),
    'cerrado_sentinel':  ('cerrado.ii.sentinel.raw.xlsx', 'sentinel'),
}


def load_csv(dataset_name, datapath='./data/bases'):
    filename, family = REMOTE_SENSING_FILES[dataset_name]
    csv_file_path = os.path.join(datapath, filename)
    npy_file_path = os.path.join(datapath, os.path.splitext(filename)[0] + '_features.npy')

    if not os.path.exists(npy_file_path):
        print(f"Processando {filename} e salvando features... Isso é executado apenas uma vez.")
        os.makedirs(datapath, exist_ok=True)

        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"Arquivo {csv_file_path} não encontrado.")

        data = pd.read_excel(csv_file_path) if csv_file_path.lower().endswith(('.xlsx', '.xls')) \
            else pd.read_csv(csv_file_path)

        regex = MODIS_FEATURE_RE if family == 'modis' else SENTINEL_FEATURE_RE
        feature_cols = [c for c in data.columns if regex.match(str(c))]

        label_col = 'label'
        if label_col not in data.columns:
            raise ValueError(f"Coluna '{label_col}' não encontrada no dataset.")

        y_codes = data[label_col].astype('category').cat.codes.values
        y_cats = dict(enumerate(data[label_col].astype('category').cat.categories))

        x_raw = data[feature_cols].astype(np.float64)

        extreme_frac = float((x_raw.abs() > EXTREME_THRESHOLD).values.mean())
        print(f"[{dataset_name}] {len(feature_cols)} feature cols | "
              f"{extreme_frac * 100:.2f}% das células além de uma faixa física plausível "
              f"(|x|>{EXTREME_THRESHOLD:.0e}) antes da limpeza.")
        if extreme_frac > 0.5:
            print(f"  AVISO: mais da metade de '{dataset_name}' é dominada por esses valores "
                  f"implausíveis o sinal normalizado tende a ser majoritariamente ruído.")

        lower = x_raw.quantile(CLIP_LOW_Q)
        upper = x_raw.quantile(CLIP_HIGH_Q)
        x_clipped = x_raw.clip(lower=lower, upper=upper, axis=1)
        x_clipped = x_clipped.fillna(x_clipped.mean())

        x_scaled = RobustScaler().fit_transform(x_clipped)
        x_scaled = np.clip(x_scaled, -POST_SCALE_CLIP, POST_SCALE_CLIP)
        x_scaled = (x_scaled + POST_SCALE_CLIP) / (2 * POST_SCALE_CLIP)  # -> [0, 1]

        save_data = {'data': x_scaled.astype(np.float32), 'label': y_codes.astype(np.int32),
                     'classes': y_cats, 'extreme_fraction': extreme_frac}
        np.save(npy_file_path, save_data)
        print(f"Features salvas em {npy_file_path}")

    data = np.load(npy_file_path, allow_pickle=True).item()
    x, y = data['data'], data['label']
    print(f"Carregado {x.shape[0]} amostras com {x.shape[1]} features e {len(np.unique(y))} "
          f"classes (extreme_fraction={data.get('extreme_fraction', float('nan')):.2%})")
    return x, y


class LoadDataset(Dataset):
    def __init__(self, dataset_name, labeled_ratio=0.05, eval_ratio=0.02, seed=42):
        """
        labeled_ratio: fração das amostras que manterão rótulo
                       (ex: 0.1 = 10% rotuladas)
        """
        self.dataset_name = dataset_name.lower()

        if self.dataset_name == 'reuters':
            self.x, self.y = load_reuters_data()
        elif self.dataset_name == 'mnist':
            self.x, self.y = load_mnist(dataset='mnist', path='./data/mnist')
        elif self.dataset_name == 'fashion_mnist':
            self.x, self.y = load_mnist(dataset='fashion_mnist', path='./data/fmnist')
        elif self.dataset_name == 'cifar10':
            self.x, self.y = load_cifar10()
        elif self.dataset_name == 'cifar100':
            self.x, self.y = load_cifar100()
        elif self.dataset_name == 'stl10':
            self.x, self.y = load_stl10()
        elif self.dataset_name in REMOTE_SENSING_FILES:
            self.x, self.y = load_csv(self.dataset_name)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        np.random.seed()
        n = len(self.y)

        y_train = self.y.copy()
        y_eval = self.y.copy()

        train_mask = np.random.rand(n) < labeled_ratio
        eval_mask = (~train_mask) & (np.random.rand(n) < eval_ratio)

        y_train[~train_mask] = -1
        y_eval[~eval_mask] = -1

        self.y_train = y_train
        self.y_eval = y_eval

        print(
            f"[Semi-supervised] {train_mask.sum()} / {n} "
            f"amostras rotuladas ({labeled_ratio*100:.1f}%)"
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx])).float(),
            torch.tensor(self.y[idx]).long(),
            torch.tensor(idx)
        )


# --- FUNÇÕES DE MÉTRICA (INALTERADAS) ---

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