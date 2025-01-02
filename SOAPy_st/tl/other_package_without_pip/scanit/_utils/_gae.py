import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE

import pickle
import numpy as np

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def rep_gae(n_h, X, A, mdl_type="VGAE"):

    n_nd, n_f = X.shape

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Encoder, self).__init__()
            self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
            if mdl_type == 'GAE':
                self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
            elif mdl_type == 'VGAE':
                self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
                self.conv_logvar = GCNConv(
                    2 * out_channels, out_channels, cached=True)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            if mdl_type == 'GAE':
                return self.conv2(x, edge_index)
            elif mdl_type == 'VGAE':
                return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

    channels = n_h
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mdl_type == 'GAE':
        model = GAE(Encoder(n_f, channels)).to(device)
    elif mdl_type == 'VGAE':
        model = VGAE(Encoder(n_f, channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    X = torch.FloatTensor(X)
    X = X.to(device)
    edge_list = sparse_mx_to_torch_edge_list(A)
    edge_list = edge_list.to(device)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(X, edge_list)
        loss = model.recon_loss(z, edge_list)
        if mdl_type == 'VGAE':
            loss = loss + (1 / n_nd) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    for epoch in range(1000):
        loss = train()
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
    
    z,_,_ = model(X, edge_list)
    return z.cpu().detach().numpy()

def main():
    f = open("./X.pkl", 'rb')
    X = pickle.load(f)
    X = X.todense()
    f = open("./A.pkl", 'rb')
    A = pickle.load(f)
    rep_gae(100, X, A, mdl_type='VGAE')

if __name__=="__main__":
    main()
