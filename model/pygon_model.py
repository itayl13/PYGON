import math
import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleList
from torch.nn import functional
from torch.nn.parameter import Parameter


class PYGONModel(Module):
    def __init__(self, n_features, hidden_layers, dropout, activations, p=0.5, normalization="correct"):
        super(PYGONModel, self).__init__()
        hidden_layers = [n_features] + hidden_layers + [1]    # input_dim, hidden_layer0, ..., hidden_layerN, 1
        self._layers = ModuleList([GraphConvolution(first, second)
                                   for first, second in zip(hidden_layers[:-1], hidden_layers[1:])])
        self._p = p
        self._normalization = self.normalize if normalization == "correct" else self.normalize_no_correction
        self._alpha = Parameter(torch.tensor(0.), requires_grad=False)
        self._beta = Parameter(torch.tensor(0.), requires_grad=True)
        self._gamma = Parameter(torch.tensor(-1.), requires_grad=True)
        self._activations = activations  # Activation functions from input layer to last hidden layer
        self._dropout = dropout

    def forward(self, x, adj, get_representation=False):
        adj = self._normalization(adj)
        for i, layer in enumerate(self._layers[:-1]):
            x = self._activations[i](layer(x, adj))
            x = functional.dropout(x, self._dropout, training=self.training)
        x = self._layers[-1](x, adj)
        return torch.sigmoid(x)

    def normalize(self, a):
        a_tilde_diag = torch.eye(a.size(0), dtype=a.dtype, device=a.device) * self._gamma
        a_tilde_off_diag = torch.where(a > 0, (1 - self._p) / self._p * torch.exp(self._alpha), -torch.exp(self._beta))
        a_tilde_off_diag -= torch.diag(torch.diag(a_tilde_off_diag))
        a_tilde = (a_tilde_diag + a_tilde_off_diag) * (torch.pow(torch.tensor(a.size(0), dtype=torch.float64), -0.5))
        # final A: gamma for self-loops, e^alpha for existing edges, -e^beta for absent edges, all normalized by sqrt(N).
        return a_tilde

    def normalize_no_correction(self, a):
        a_tilde_diag = torch.eye(a.size(0), dtype=a.dtype, device=a.device) * self._gamma
        a_tilde_off_diag = torch.where(a > 0, torch.exp(self._alpha), -torch.exp(self._beta))
        a_tilde_off_diag -= torch.diag(torch.diag(a_tilde_off_diag))
        a_tilde = (a_tilde_diag + a_tilde_off_diag) * (torch.pow(torch.tensor(a.size(0), dtype=torch.float64), -0.5))
        # final A: gamma for self-loops, e^alpha for existing edges, -e^beta for absent edges, all normalized by sqrt(N).
        return a_tilde


class GraphConvolution(Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(torch.zeros((self.in_features, self.out_features), dtype=torch.double),
                                requires_grad=True)
        self.bias = Parameter(torch.zeros(self.out_features, dtype=torch.double), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # A * x * W
        support = torch.mm(adj, x)
        output = torch.mm(support, self.weight)
        if self.bias is None:
            return output
        return output + self.bias

    def __repr__(self):
        return f"{type(self)} ({self._in_features} -> {self._out_features})"
