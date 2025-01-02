import random
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax

import numpy as np
from scipy import sparse
import pickle


def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def rep_dgi(
    n_h, 
    X, 
    A, 
    n_epoch=1000, 
    lr=0.001, 
    print_step=500, 
    torch_seed=None, 
    python_seed=None,
    numpy_seed=None,
    device=None
):
    # torch.set_deterministic(True)
    if not torch_seed is None:
        torch.manual_seed(torch_seed)
    if not python_seed is None:
        random.seed(python_seed)
    if not numpy_seed is None:
        np.random.seed(numpy_seed)


    n_f = X.shape[1]

    class Encoder(nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super(Encoder, self).__init__()
            self.conv = GCNConv(in_channels, hidden_channels, cached=False)
            self.prelu = nn.PReLU(hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
            self.prelu2 = nn.PReLU(hidden_channels)
        
        def forward(self, x, edge_index):
            x = self.conv(x, edge_index)
            x = self.prelu(x)
            x = self.conv2(x, edge_index)
            x = self.prelu2(x)
            return x
    
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    model = DeepGraphInfomax(
        hidden_channels = n_h, encoder = Encoder(n_f, n_h),
        summary = lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption = corruption).to(device)
    
    X = torch.FloatTensor(X)
    X = X.to(device)
    edge_list = sparse_mx_to_torch_edge_list(A)
    edge_list = edge_list.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    def train():
        model.train()
        optimiser.zero_grad()
        pos_z, neg_z, summary = model(X, edge_list)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimiser.step()
        return loss.item()

    for epoch in range(n_epoch):
        loss = train()
        if epoch % print_step == 0 or epoch+1 == n_epoch:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
    
    z,_,_ = model(X, edge_list)
    return z.cpu().detach().numpy()

def main():
    f = open("./X.pkl", 'rb')
    X = pickle.load(f)
    X = X.toarray()
    f = open("./A.pkl", 'rb')
    A = pickle.load(f)
    z = rep_dgi(32, X, A)
    print(z.shape)

if __name__=="__main__":
    main()
