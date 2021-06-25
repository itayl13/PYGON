from torch import randn
import torch.optim as optim
import torch.nn.functional as functional


class LearningHyperParams:
    def __init__(self, learning_params_dict):
        self._learning_params_dict = learning_params_dict

        self.model = self._set_param("model", "PYGON")
        self.input_features = self._set_param("features", ["Degree"])
        self.hidden_layers = self._set_param("hidden_layers", [225, 175, 400, 150])
        self.optimizer = self._set_param("optimizer", optim.Adam)
        self.activation = self._set_param("activation", functional.relu)
        self.epochs = self._set_param("epochs", 1000)
        self.dropout = self._set_param("dropout", 0.4)
        self.learning_rate = self._set_param("lr", 0.005)
        self.l2_regularization = self._set_param("regularization", 0.0005)
        self.early_stop = self._set_param("early_stop", True)
        self.loss_coefficients = self._set_param("coeffs", [1., 0., 0.])
        self.unary_loss_type = self._set_param("unary", "bce")
        self.edge_normalization = self._set_param("edge_normalization", "correct")

        self._validate_params()

    def _set_param(self, learning_param_key, default):
        return self._learning_params_dict[learning_param_key] \
            if learning_param_key in self._learning_params_dict else default

    def _validate_params(self):
        self._validate_model()
        self._validate_features()
        self._validate_layers()
        self._validate_optimizer()
        self._validate_activation()
        self._validate_epochs()
        self._validate_dropout()
        self._validate_lr()
        self._validate_regularization()
        self._validate_early_stop()
        self._validate_loss_coefficients()
        self._validate_unary_loss()
        self._validate_edge_normalization()

        self._validate_params_matching_to_model()

    def _validate_model(self):
        assert self.model in ["PYGON", "GAT"], "Learning model must be either 'PYGON' or 'GAT'"

    def _validate_features(self):
        possible_input_features = {'Degree', 'In-Degree', 'Out-Degree', 'Betweenness', 'BFS',
                                   'Motif_3', 'additional_features', 'neighbors_degree_measures',
                                   'bidirectional_edges_count'}
        assert type(self.input_features) == list, "Input features must be a list"
        assert len(self.input_features) == len(set(self.input_features)), "Input features list must have unique items"
        assert set(self.input_features).issubset(possible_input_features), \
            f"Input features must be a subset of the following features: {possible_input_features}"

    def _validate_layers(self):
        assert type(self.hidden_layers) == list and \
               all(type(self.hidden_layers[i]) == int for i in range(len(self.hidden_layers))), \
               "Hidden layers must be a list of integers"

    def _validate_optimizer(self):
        assert issubclass(self.optimizer, (optim.Adam, optim.SGD)), "Incorrect value of optimizer"

    def _validate_activation(self):
        x = randn((5, 5))
        assert type(self.activation(x)) == type(x) and self.activation(x).size() == x.size(), \
            "Incorrect value of activation"

    def _validate_epochs(self):
        assert type(self.epochs) == int, "Number of epochs must be an integer"

    def _validate_dropout(self):
        assert type(self.dropout) == float, "Dropout rate must be of type float"

    def _validate_lr(self):
        assert type(self.learning_rate) == float, "Learning rate must be of type float"

    def _validate_regularization(self):
        assert type(self.l2_regularization) == float, "L2 regularization must be of type float"

    def _validate_early_stop(self):
        assert type(self.early_stop) == bool, "Early stopping must be a boolean"

    def _validate_loss_coefficients(self):
        assert type(self.loss_coefficients) == list and len(self.loss_coefficients) == 3 and \
            all(type(self.loss_coefficients[i]) == float for i in range(len(self.loss_coefficients))), \
            "Loss coefficients must be a list of three float numbers"

    def _validate_unary_loss(self):
        assert self.unary_loss_type in ["bce", "mse"], "Loss type must be either 'bce' or 'mse'"

    def _validate_edge_normalization(self):
        assert self.edge_normalization in ["correct", "none"], "Edge normalization must be either 'correct' or 'none'"

    def _validate_params_matching_to_model(self):
        if self.model == "PYGON":
            assert all(["coeffs" in self._learning_params_dict, "unary" in self._learning_params_dict,
                        "edge_normalization" in self._learning_params_dict]), \
                "For using the PYGON model, one must specify 'coeffs', 'unary' and 'edge_normalization'"
