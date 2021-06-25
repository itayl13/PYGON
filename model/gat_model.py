import numpy as np
import torch
from itertools import product
from multiprocessing import Pool
from torch.nn.modules.module import Module
from torch.nn import Linear, LeakyReLU, Parameter, ModuleList, functional, init


class GATModel(Module):
    def __init__(self, n_features, hidden_layers, dropout, activations):
        """
        Based on https://github.com/gordicaleksa/pytorch-GAT
        """
        super(GATModel, self).__init__()
        self.dropout = dropout
        self.activations = activations
        layers_list = GATModel._build_layer_sizes(n_features, hidden_layers, dropout, activations)
        self._layers = ModuleList(layers_list)

    @staticmethod
    def _build_layer_sizes(n_features, hidden_layers, dropout, activations):
        hidden_layers = [n_features] + hidden_layers + [1]
        n_heads = [1] + [2] * len(activations)
        layers_list = []
        for layer_idx in range(len(activations)):
            layers_list.append(
                GATLayer(n_heads[layer_idx] * hidden_layers[layer_idx], hidden_layers[layer_idx + 1],
                         n_heads[layer_idx + 1], activations[layer_idx], dropout))
        layers_list.append(
            GATLayer(n_heads[-1] * hidden_layers[-1], 1, 1, None, dropout)
        )
        return layers_list

    def forward(self, x, adj):
        for i, layer in enumerate(self._layers[:-1]):
            x = self.activations[i](layer(x, adj))
            x = functional.dropout(x, self.dropout, training=self.training)
        x = self._layers[-1](x, adj)
        return torch.sigmoid(x)


class GATLayerBase(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_heads, activation, dropout):
        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.activation = activation
        self.dropout = dropout

        self.linear_proj = Linear(num_in_features, num_of_heads * num_out_features, bias=False).double()

        self.scoring_fn_target = Parameter(torch.empty((1, num_of_heads, num_out_features)), requires_grad=True)
        self.scoring_fn_source = Parameter(torch.empty((1, num_of_heads, num_out_features)), requires_grad=True)

        self.bias = Parameter(torch.empty((num_of_heads * num_out_features), dtype=torch.double), requires_grad=True)

        self.skip_proj = Linear(num_in_features, num_of_heads * num_out_features, bias=False).double()

        self.leakyReLU = LeakyReLU(0.2)

        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        init.xavier_uniform_(self.linear_proj.weight)
        init.xavier_uniform_(self.scoring_fn_target)
        init.xavier_uniform_(self.scoring_fn_source)
        init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
            out_nodes_features += in_nodes_features.unsqueeze(1)
        else:
            out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)

        out_nodes_features += self.bias
        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayer(GATLayerBase):
    def __init__(self, num_in_features, num_out_features, num_of_heads, activation, dropout):
        super().__init__(num_in_features, num_out_features, num_of_heads, activation, dropout)

    def forward(self, x, adj):
        out_nodes_features = self.attend_by_adj_only(x, adj) if self.num_of_heads == 1 else \
            self.attend_by_adj_and_comp_adj(x, adj)

        return out_nodes_features
    
    def attend_by_adj_only(self, x, adj_and_comp_adj):
        adj, _ = adj_and_comp_adj
        x, graph_size, scores_source, scores_target, nodes_features_proj = self.prepare_for_attention(x, adj)

        out_nodes_features = self.attention_one_head(scores_source, scores_target, nodes_features_proj, adj,
                                                     x, graph_size)

        out_nodes_features = self.skip_concat_bias(x, out_nodes_features)
        return out_nodes_features

    def attend_by_adj_and_comp_adj(self, x, adj_and_comp_adj):
        adj, comp_adj = adj_and_comp_adj
        x, graph_size, scores_source, scores_target, nodes_features_proj = self.prepare_for_attention(x, adj)

        scores_source_original_graph, scores_source_complement_graph = scores_source.split(1, 1)
        scores_target_original_graph, scores_target_complement_graph = scores_target.split(1, 1)
        nodes_features_proj_original_graph, nodes_features_proj_complement_graph = nodes_features_proj.split(1, 1)

        out_features = (
            self.attention_one_head(
                scores_source_original_graph, scores_target_original_graph, nodes_features_proj_original_graph,
                adj, x, graph_size),
            self.attention_one_head(
                scores_source_complement_graph, scores_target_complement_graph, nodes_features_proj_complement_graph,
                comp_adj, x, graph_size)
                )

        out_nodes_features = torch.cat(out_features, dim=1)

        out_nodes_features = self.skip_concat_bias(x, out_nodes_features)
        return out_nodes_features
    
    def prepare_for_attention(self, x, adj):
        graph_size = x.shape[0]
        assert adj.shape[0] == 2, f'Expected edge index with shape=(2,E) got {adj.shape}'

        x = functional.dropout(x, self.dropout)
        nodes_features_proj = self.linear_proj(x).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = functional.dropout(nodes_features_proj, self.dropout)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        return x, graph_size, scores_source, scores_target, nodes_features_proj 

    def attention_one_head(self, scores_source, scores_target, nodes_features_proj, adj, in_features, graph_size):
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(
            scores_source, scores_target, nodes_features_proj, adj)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, adj[1], graph_size)
        attentions_per_edge = functional.dropout(attentions_per_edge, self.dropout)

        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, adj, in_features, graph_size)
        return out_nodes_features

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, graph_size):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(
            exp_scores_per_edge, trg_index, graph_size)

        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, graph_size):
        trg_index_broadcast = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[0] = graph_size
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(0, trg_index_broadcast, exp_scores_per_edge)

        return neighborhood_sums.index_select(0, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, graph_size):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[0] = graph_size
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        trg_index_broadcast = self.explicit_broadcast(edge_index[1], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(0, trg_index_broadcast, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    @staticmethod
    def lift(scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[0]
        trg_nodes_index = edge_index[1]

        scores_source = scores_source.index_select(0, src_nodes_index)
        scores_target = scores_target.index_select(0, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(0, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    @staticmethod
    def explicit_broadcast(this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        return this.expand_as(other)
