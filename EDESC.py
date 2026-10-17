from __future__ import print_function, division

import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import LoadDataset, cluster_acc
import tensorflow as tf
import keras.backend as K
import warnings
from AutoEncoder import AE
from InitializeD import Initialization_D
from Constraint import D_constraint1, D_constraint2
import time

warnings.filterwarnings("ignore")

class EDESC(nn.Module):
    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 num_sample,
                 pretrain_path='data/reuters.pkl'):
        super(EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z, n_clusters))

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # Load pre-trained weights
        self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        print('Load pre-trained model from', path)

    def forward(self, x):
        x_bar, z = self.ae(x)
        d = args.d
        s = None
        eta = args.eta

        # Calculate subspace affinity
        for i in range(self.n_clusters):
            si = torch.sum(torch.pow(torch.mm(z,self.D[:,i*d:(i+1)*d]),2),1,keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s,si),1)
        s = (s+eta*d) / ((eta+1)*d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, beta):
        # Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)

        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target)

        # Constraints
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)

        # Total_loss
        total_loss = reconstr_loss + beta * kl_loss + loss_d1 + loss_d2
        return total_loss

def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()

def pretrain_ae(model):
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(50):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
    torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))

def train_EDESC():
    model = EDESC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        num_sample = args.num_sample,
        pretrain_path=args.pretrain_path).to(device)

    start = time.time()

    # Check if pretrain file exists, if not, perform pretraining
    if not os.path.exists(args.pretrain_path):
        print(f"Pretrain file not found at {args.pretrain_path}. Starting pretraining...")
        pretrain_ae(model.ae)
    else:
        # Load pre-trained model
        model.pretrain(model.pretrain_path)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=10)

    # Get clusters from K-means
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    print("Initial Cluster Centers: ", y_pred)

    # Initialize D
    D = Initialization_D(hidden, y_pred, args.n_clusters, args.d)
    D = torch.tensor(D).to(torch.float32)

    accmax = 0
    nmimax = 0
    arimax = 0
    y_pred_last = y_pred
    model.D.data = D.to(device)
    model.train()

    for epoch in range(200):
        x_bar, s, z = model(data)

        # Update refined subspace affinity
        tmp_s = s.data
        s_tilde = refined_subspace_affinity(tmp_s)

        # Evaluate clustering performance
        y_pred = tmp_s.cpu().detach().numpy().argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = y_pred

        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)
        ari = ari_score(y, y_pred)
        if acc > accmax:
            accmax = acc
        if nmi > nmimax:
            nmimax = nmi
        if ari > arimax:
            arimax = ari

        print(f'Iter {epoch} :Current Acc {acc:.4f} :Max Acc {accmax:.4f} , Current nmi {nmi:.4f} :Max nmi {nmimax:.4f} , Current ari {ari:.4f} :Max ari {arimax:.4f}')

        ############## Total loss function ######################
        loss = model.total_loss(data, x_bar, z, pred=s, target=s_tilde, dim=args.d, n_clusters = args.n_clusters, beta = args.beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.time()
    print('Running time: ', end-start)
    return accmax, nmimax, arimax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--d', default=2, type=int)
    parser.add_argument('--n_z', default=20, type=int)
    parser.add_argument('--eta', default=2, type=int)
    parser.add_argument('--dataset', type=str, default='reuters',
                        choices=['reuters', 'mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'stl10', 'ssfreg', 'ssf', 'pampa', 'cerrados'])
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--beta', default=1, type=float, help='coefficient of subspace affinity loss')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # Auto-configure parameters based on dataset
    if args.dataset == 'reuters':
        if not args.pretrain_path:
            args.pretrain_path = 'data/reuters.pkl'
        args.n_clusters = 4
        args.n_input = 2000
        args.num_sample = 10000
    elif args.dataset == 'mnist':
        if not args.pretrain_path:
            args.pretrain_path = 'data/mnist.pkl'
        args.n_clusters = 10
        args.n_input = 28*28
        args.num_sample = 70000
    elif args.dataset == 'fashion_mnist':
        if not args.pretrain_path:
            args.pretrain_path = 'data/fashion_mnist.pkl'
        args.n_clusters = 10
        args.n_input = 28*28
        args.num_sample = 70000
    elif args.dataset == 'cifar10':
        if not args.pretrain_path:
            args.pretrain_path = 'data/cifar10.pkl'
        args.n_clusters = 10
        args.n_input = 2*32*32
        args.num_sample = 60000
    elif args.dataset == 'cifar100':
        if not args.pretrain_path:
            args.pretrain_path = 'data/cifar100.pkl'
        args.n_clusters = 20
        args.n_input = 2*32*32
        args.num_sample = 60000
    elif args.dataset == 'stl10':
        if not args.pretrain_path:
            args.pretrain_path = 'data/stl10.pkl'
        args.n_clusters = 10
        args.n_input = 3*96*96
        args.num_sample = 13000
    elif args.dataset == 'ssf_filtered':
        if not args.pretrain_path:
            args.pretrain_path = 'data/EDESC/ssf_filtered/ssf_filtered.npz'
        args.n_clusters = 8
        args.n_input = 279
        args.num_sample = 8763
    elif args.dataset == 'ssf':
        if not args.pretrain_path:
            args.pretrain_path = 'data/EDESC/ssf/ssf.npz'
        args.n_clusters = 8
        args.n_input = 279
        args.num_sample = 10000
    elif args.dataset == 'pampa':
        if not args.pretrain_path:
            args.pretrain_path = 'data/EDESC/pampa/pampa.npz'
        args.n_clusters = 9
        args.n_input = 184
        args.num_sample = 6746
    elif args.dataset == 'pampa_filtered':
        if not args.pretrain_path:
            args.pretrain_path = 'data/EDESC/pampa_filtered/pampa_filtered.npz'
        args.n_clusters = 9
        args.n_input = 184
        args.num_sample = 5886
    elif args.dataset == 'cerrado_sentinel':
        if not args.pretrain_path:
            args.pretrain_path = 'data/EDESC/cerrado_sentinel/cerrado_sentinel.npz'
        args.n_clusters = 5
        args.n_input = 286
        args.num_sample = 33544
    elif args.dataset == 'cerrado_modis':
        if not args.pretrain_path:
            args.pretrain_path = 'data/EDESC/cerrado_modis/cerrado_modis.npz'
        args.n_clusters = 12
        args.n_input = 92
        args.num_sample = 50160
    elif args.dataset == 'cerrado_filtered':
        if not args.pretrain_path:
            args.pretrain_path = 'data/EDESC/cerrado_filtered/cerrado_filtered.npz'
        args.n_clusters = 12
        args.n_input = 92
        args.num_sample = 43910

    dataset = LoadDataset(args.dataset)

    bestacc = 0
    bestnmi = 0
    bestari = 0
    for i in range(10):
        acc, nmi, ari = train_EDESC()
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
        if ari > bestari:
            bestari = ari

    print(f'Best ACC {bestacc:.4f}  Best NMI {bestnmi:.4f}  Best ARI {bestari:.4f}')