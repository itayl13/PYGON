import pickle
from sklearn.preprocessing import StandardScaler
import torch
from __init__ import *
from learning_params import LearningHyperParams
from utils import *
from graphs_and_features import GraphBuilder, FeatureCalculator
import pygon_runner as pygon


class PYGONMain:
    def __init__(self, v, p, sg, d, features, subgraph="clique", new_runs=0, nni=False, device=1):
        self._params = {
            'vertices': v,
            'probability': p,
            'subgraph': subgraph,
            'subgraph_size': sg,
            'directed': d,
            'features': features
        }
        """
        'subgraph' gets one of the following values: 
        'clique', 'dag-clique', 'k-plex', 'biclique' or 'G(k, q)' for G(k, q) with probability q (e.g. 'G(k, 0.9)').
        """
        self._new_runs = new_runs
        self._key_name = (subgraph, f"n_{v}_p_{p}_size_{sg}_{'d' if d else 'ud'}")
        self._head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                       subgraph, self._key_name[1] + '_runs')
        self._device = device
        self._load_data()
        self._nni = nni

    def _load_data(self):
        check_make_dir(self._head_path)
        graph_ids = os.listdir(self._head_path)
        if len(graph_ids) == 0 and self._new_runs == 0:
            raise ValueError(f"No runs of G({self._params['vertices']}, {self._params['probability']}) "
                             f"with a {self._params['subgraph']} subgraph of size {self._params['subgraph_size']} "
                             f"were saved, and no new runs were requested.")
        self._feature_matrices, self._adjacency_matrices, self._labels = [], [], []
        self._num_runs = len(graph_ids) + self._new_runs
        for run in range(self._num_runs):
            dir_path = os.path.join(self._head_path, self._key_name[1] + "_run_" + str(run))
            data = GraphBuilder(self._params, dir_path)
            gnx = data.graph()
            labels = data.labels()

            fc = FeatureCalculator(self._params, gnx, dir_path, self._params['features'], gpu=True, device=self._device)
            self._adjacency_matrices.append(fc.adjacency_matrix)
            self._feature_matrices.append(fc.feature_matrix)
            if type(labels) == dict:
                new_labels = [[y for x, y in labels.items()]]
                self._labels += new_labels
            else:
                self._labels += [labels]
        self._scale_matrices()

    def _scale_matrices(self):
        scaler = StandardScaler()
        all_matrix = np.vstack(self._feature_matrices)
        scaler.fit(all_matrix)
        for i in range(len(self._feature_matrices)):
            self._feature_matrices[i] = scaler.transform(self._feature_matrices[i].astype('float64'))

    def train(self, input_params=None, check='split'):
        if input_params is None:
            input_params = {
                "model": "PYGON",
                "edge_normalization": "correct"
            }
            update_params(input_params, self._params["subgraph_size"])

        learning_hyperparams = LearningHyperParams(input_params)
        _ = pygon.run_pygon(feature_matrices=self._feature_matrices, adj_matrices=self._adjacency_matrices,
                            labels=self._labels, learning_hyperparams=learning_hyperparams, iterations=2,
                            dumping_name=self._key_name + ('_'.join(map(str, learning_hyperparams.hidden_layers)), ),
                            graph_params=self._params, is_nni=self._nni, device=self._device, check=check,
                            purpose='checkout')
        return

    def train_performance(self, input_params, check='split'):
        learning_hyperparams = LearningHyperParams(input_params)
        aggregated_results = pygon.run_pygon(
            feature_matrices=self._feature_matrices,
            adj_matrices=self._adjacency_matrices,
            labels=self._labels,
            learning_hyperparams=learning_hyperparams,
            iterations=2, dumping_name=self._key_name,
            graph_params=self._params,
            check=check,
            device=self._device,
            purpose='performance')
        return aggregated_results

    def all_labels_to_pkl(self):
        pickle.dump(self._labels, open(os.path.join(self._head_path, 'all_labels.pkl'), 'wb'))

    @property
    def labels(self):
        return self._labels


if __name__ == "__main__":
    # Available features: Degree ('Degree'), In-Degree ('In-Degree'), Out-Degree ('Out-Degree'),
    #                     Betweenness Centrality ('Betweenness'), BFS moments ('BFS'), 3-motifs ('Motif_3') and
    #                     the extra features based on the motifs ('additional_features')

    params = {
        "model": "PYGON",
        "features": ['Motif_3'],
        "hidden_layers": [225, 175, 400, 150],
        "epochs": 1000,
        "dropout": 0.4,
        "lr": 0.005,
        "regularization": 0.0005,
        "coeffs": [1., 0., 0.],
        "unary": "bce",
        "optimizer": torch.optim.Adam,
        "activation": torch.relu,
        "edge_normalization": "correct",
        "early_stop": True
    }
    gg = PYGONMain(500, 0.5, 20, False, subgraph='clique', features=['Motif_3'], new_runs=20)
    gg.train(input_params=params)
