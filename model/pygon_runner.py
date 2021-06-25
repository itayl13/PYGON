import logging
from datetime import datetime
from itertools import product
from functools import partial
import torch
import nni
from utils import *
from pygon_model import PYGONModel
from gat_model import GATModel
from loggers import PrintLogger, multi_logger, FileLogger
from sklearn.metrics import roc_auc_score


class ModelRunner:
    def __init__(self, conf, logger, weights, graph_params, early_stop=True, is_nni=False, tmp_path="", device=1):
        self._logger = logger
        self._conf = conf
        self._weights_dict = weights
        self._early_stop = early_stop
        self._graph_params = graph_params
        self._is_nni = is_nni
        self._device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
        self._tmp_path = tmp_path  # Path where the logs are saved. We save the best model state there.

    def _build_weighted_loss(self, labels):
        weights_list = []
        for i in range(labels.shape[0]):
            weights_list.append(self._weights_dict[labels[i].data.item()])
        weights_tensor = torch.tensor(weights_list, dtype=torch.double, device=labels.device)
        if self._conf["unary"] == "bce":
            self._criterion = torch.nn.BCELoss(weight=weights_tensor).to(labels.device)
        else:
            self._criterion = partial(self._weighted_mse, weights_tensor=weights_tensor)

    @staticmethod
    def _weighted_mse(outputs, labels, weights_tensor):
        return torch.mean(weights_tensor * torch.pow(outputs - labels, torch.tensor([2], device=labels.device)))

    @staticmethod
    def _pairwise_loss(flat_x, flat_adj):
        return - torch.mean((1 - flat_adj) * torch.log(
            torch.where(1 - flat_x <= 1e-8, torch.tensor([1e-8], dtype=torch.double, device=flat_x.device),
                        1 - flat_x)) +
                            flat_adj * torch.log(
            torch.where(flat_x <= 1e-8, torch.tensor([1e-8], dtype=torch.double, device=flat_x.device), flat_x)))

    def _binomial_reg(self, y_hat):
        return - torch.mean(y_hat * np.log(self._graph_params["subgraph_size"] / self._graph_params["vertices"]) +
                            (1 - y_hat) * np.log(
            1 - self._graph_params["subgraph_size"] / self._graph_params["vertices"]))

    @property
    def logger(self):
        return self._logger

    def _get_model(self):
        if self._conf["model"] == "PYGON":
            model = PYGONModel(n_features=self._conf["training_mat"][0].shape[1],
                               hidden_layers=self._conf["hidden_layers"],
                               dropout=self._conf["dropout"], activations=self._conf["activations"],
                               p=self._graph_params["probability"], normalization=self._conf["edge_normalization"])
        else:
            model = GATModel(n_features=self._conf["training_mat"][0].shape[1],
                             hidden_layers=self._conf["hidden_layers"],
                             dropout=self._conf["dropout"], activations=self._conf["activations"])
        opt = self._conf["optimizer"](model.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])

        training_mats, training_adjs, training_labels, eval_mats, eval_adjs, eval_labels, \
            test_mats, test_adjs, test_labels = [], [], [], [], [], [], [], [], []
        model_dict = {"model": model, "optimizer": opt}
        for element, single_name, dict_name in zip(
                [training_mats, training_adjs, training_labels, eval_mats, eval_adjs, eval_labels,
                 test_mats, test_adjs, test_labels],
                ["training_mat", "training_adj", "training_labels", "eval_mat", "eval_adj", "eval_labels",
                 "test_mat", "test_adj", "test_labels"],
                ["training_mats", "training_adjs", "training_labels", "eval_mats", "eval_adjs", "eval_labels",
                 "test_mats", "test_adjs", "test_labels"]):
            element = [self._conf[single_name][idx] for idx in range(len(self._conf[single_name]))]
            model_dict[dict_name] = element
        return model_dict

    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, verbose=2):
        if self._is_nni:
            self._logger.debug(f"Model: \nhidden layers: [{', '.join(map(str, self._conf['hidden_layers']))}] \n"
                               f"dropout: {self._conf['dropout']:.4f} \nlearning rate: {self._conf['lr']:.4f} \n"
                               f"L2 regularization: {self._conf['weight_decay']:.4f}")
            verbose = 0
        model = self._get_model()
        epochs_done, training_output_labels, intermediate_training_results, intermediate_eval_results, \
            intermediate_test_results = self.train(self._conf["epochs"], model=model, verbose=verbose)
        # Final evaluation for all sets, based on the best model from the training process.
        train_res = self.evaluate(model=model, what_set="training", verbose=verbose if not self._is_nni else 0)
        eval_res = self.evaluate(model=model, what_set="eval", verbose=verbose if not self._is_nni else 0)
        result = self.evaluate(model=model, what_set="test", verbose=verbose if not self._is_nni else 0)

        intermediate_results = {}
        for measure, what_set in product(["auc", "all_loss", "unary_loss", "pairwise_loss", "binom_reg"],
                                         zip(["train", "eval", "test"],
                                             [intermediate_training_results, intermediate_eval_results,
                                              intermediate_test_results])):
            intermediate_results[f"{measure}_{what_set[0]}"] = what_set[1][measure]

        final_results = {
            "training_output_labels": train_res["output_labels"],
            'eval_output_labels': eval_res["output_labels"],
            "test_output_labels": result["output_labels"],
            "epochs_done": epochs_done
        }
        for measure, what_set in product(["auc", "all_loss", "unary_loss", "pairwise_loss", "binom_reg"],
                                         zip(["train", "eval", "test"],
                                             [train_res, eval_res, result])):
            final_results[f"{measure}_{what_set[0]}"] = what_set[1][measure]

        if self._is_nni or verbose != 0:
            for message, value in zip(
                    ['epochs done: {}', 'Final loss train: {:3.4f}', 'Final AUC train: {:3.4f}',
                     'Final loss eval: {:3.4f}', 'Final AUC eval: {:3.4f}', 'Final loss test: {:3.4f}',
                     'Final AUC test: {:3.4f}'],
                    ["epochs_done", "all_loss_train", "auc_train", "all_loss_eval", "auc_eval", "all_loss_test",
                     "auc_test"]):
                self._logger.info(message.format(final_results[value]))

        return intermediate_results, final_results

    def train(self, epochs, model=None, verbose=2):
        epochs_done = 0.
        output = 0.
        training_labels = 0.
        training_results = {"all_loss": [], "unary_loss": [], "pairwise_loss": [], "binom_reg": [],
                            "auc": []}  # All results by epoch
        eval_results = {"all_loss": [], "unary_loss": [], "pairwise_loss": [], "binom_reg": [], "auc": []}
        test_results = {"all_loss": [], "unary_loss": [], "pairwise_loss": [], "binom_reg": [], "auc": []}
        counter = 0  # For early stopping
        min_loss = None
        for epoch in range(epochs):
            epochs_done += 1
            output, training_labels, auc_train, all_loss, unary_loss, pairwise_loss, binom_reg = self._train(epoch,
                                                                                                             model,
                                                                                                             verbose)
            for key, value in zip(["auc", "all_loss", "unary_loss", "pairwise_loss", "binom_reg"],
                                  [auc_train, all_loss, unary_loss, pairwise_loss, binom_reg]):
                training_results[key].append(value)
            # -------------------------- EVAL & TEST --------------------------
            eval_res = self.evaluate(model, what_set="eval", verbose=verbose if not self._is_nni else 0)
            for key in ["auc", "all_loss", "unary_loss", "pairwise_loss", "binom_reg"]:
                eval_results[key].append(eval_res[key])
            if epoch % 5 == 0:
                test_res = self.evaluate(model, what_set="test", verbose=verbose if not self._is_nni else 0)
                for key in ["auc", "all_loss", "unary_loss", "pairwise_loss", "binom_reg"]:
                    test_results[key].append(test_res[key])

            if epoch >= 10 and self._early_stop:  # Check for early stopping during training.
                if min_loss is None:
                    min_loss = min(eval_results["all_loss"])
                    torch.save(model["model"].state_dict(),
                               os.path.join(self._tmp_path, "tmp.pt"))  # Save the best state.
                elif eval_res["all_loss"] < min_loss:
                    min_loss = min(eval_results["all_loss"])
                    torch.save(model["model"].state_dict(),
                               os.path.join(self._tmp_path, "tmp.pt"))  # Save the best state
                    counter = 0
                else:
                    counter += 1
                    if counter >= 40:  # Patience for learning
                        break
        # After stopping early, our model is the one with the best eval loss.
        model["model"].load_state_dict(torch.load(os.path.join(self._tmp_path, "tmp.pt")))
        os.remove(os.path.join(self._tmp_path, "tmp.pt"))
        return epochs_done, np.vstack((output, training_labels)), training_results, eval_results, test_results

    def _train(self, epoch, model, verbose=2):
        model_ = model["model"]
        model_ = model_.to(self._device)
        optimizer = model["optimizer"]
        n_training_graphs = len(model["training_labels"])
        graph_size = self._graph_params["vertices"]
        graphs_order = np.arange(n_training_graphs)
        np.random.shuffle(graphs_order)
        outputs = torch.zeros(graph_size * n_training_graphs, dtype=torch.double)
        output_xs = torch.zeros((graph_size ** 2) * n_training_graphs, dtype=torch.double)
        training_adj_flattened = torch.tensor(
            np.hstack([model["training_adjs"][idx].flatten() for idx in graphs_order]))
        for i, idx in enumerate(graphs_order):
            training_mat = torch.tensor(model["training_mats"][idx], device=self._device)
            training_adj = self._get_adj_matrix(model["training_adjs"][idx])
            labels = torch.tensor(data=model["training_labels"][idx], dtype=torch.double, device=self._device)
            model_.train()
            optimizer.zero_grad()
            output = model_(training_mat, training_adj)
            output_matrix_flat = (torch.mm(output, output.transpose(0, 1)) + 1 / 2).flatten()
            output_xs[i * graph_size ** 2:(i + 1) * graph_size ** 2] = output_matrix_flat.cpu()
            outputs[i * graph_size:(i + 1) * graph_size] = output.view(output.shape[0]).cpu()
            self._build_weighted_loss(labels)
            loss_train = self._conf["loss_coeffs"][0] * self._criterion(output.view(output.shape[0]), labels) + \
                self._conf["loss_coeffs"][1] * self._pairwise_loss(
                output_matrix_flat, torch.tensor(
                    model["training_adjs"][idx], dtype=torch.double, device=self._device).flatten()) + \
                self._conf["loss_coeffs"][2] * self._binomial_reg(output)
            loss_train.backward()
            optimizer.step()
        all_training_labels = torch.tensor(np.hstack([model["training_labels"][idx] for idx in graphs_order]),
                                           dtype=torch.double)
        self._build_weighted_loss(all_training_labels)
        unary_loss = self._criterion(outputs, all_training_labels)
        pairwise_loss = self._pairwise_loss(output_xs, training_adj_flattened)
        regularization = self._binomial_reg(outputs)
        all_training_loss = self._conf["loss_coeffs"][0] * unary_loss + self._conf["loss_coeffs"][1] * pairwise_loss + \
            self._conf["loss_coeffs"][2] * regularization
        auc_train = self.auc(outputs, all_training_labels)

        if verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self._logger.debug(
                f"Epoch: {epoch + 1:04d} loss_train: {all_training_loss.data.item():.4f} auc_train: {auc_train:.4f}")
        return outputs.tolist(), all_training_labels.tolist(), auc_train, all_training_loss.data.item(), \
            unary_loss.data.item(), pairwise_loss.data.item(), regularization.data.item()

    def evaluate(self, model=None, what_set="eval", verbose=2):
        model_ = model["model"]
        n_graphs = len(model[f"{what_set}_labels"])
        graph_size = self._graph_params["vertices"]
        graphs_order = np.arange(len(model[f"{what_set}_labels"]))
        np.random.shuffle(graphs_order)
        outputs = torch.zeros(graph_size * n_graphs, dtype=torch.double)
        output_xs = torch.zeros(graph_size ** 2 * n_graphs, dtype=torch.double)
        adj_flattened = torch.tensor(np.hstack([model[f"{what_set}_adjs"][idx].flatten() for idx in graphs_order]))
        for i, idx in enumerate(graphs_order):
            mat = torch.tensor(model[f"{what_set}_mats"][idx], device=self._device)
            adj = self._get_adj_matrix(model[f"{what_set}_adjs"][idx])
            model_.eval()
            output = model_(*[mat, adj])
            output_matrix_flat = (torch.mm(output, output.transpose(0, 1)) + 1 / 2).flatten()
            output_xs[i * graph_size ** 2:(i + 1) * graph_size ** 2] = output_matrix_flat.cpu()
            outputs[i * graph_size:(i + 1) * graph_size] = output.view(output.shape[0]).cpu()
        all_labels = torch.tensor(np.hstack([model[f"{what_set}_labels"][idx] for idx in graphs_order]),
                                  dtype=torch.double)
        self._build_weighted_loss(all_labels)
        unary_loss = self._criterion(outputs, all_labels)
        pairwise_loss = self._pairwise_loss(output_xs, adj_flattened)
        regularization = self._binomial_reg(outputs)
        loss = self._conf["loss_coeffs"][0] * unary_loss + self._conf["loss_coeffs"][1] * pairwise_loss + \
            self._conf["loss_coeffs"][2] * regularization
        auc = self.auc(outputs, all_labels)
        if verbose != 0:
            self._logger.info(f"{what_set}: loss= {loss.data.item():.4f}, auc= {auc:.4f}")
        result = {"all_loss": loss.data.item(), "unary_loss": unary_loss.data.item(),
                  "pairwise_loss": pairwise_loss.data.item(),
                  "binom_reg": regularization.data.item(), "auc": auc,
                  "output_labels": np.vstack((outputs.tolist(), all_labels.tolist()))}
        return result

    def _get_adj_matrix(self, plain_matrix):
        if self._conf["model"] == "PYGON":
            return torch.tensor(plain_matrix, dtype=torch.double, device=self._device)

        source_nodes_ids, target_nodes_ids = [], []
        comp_source_nodes_ids, comp_target_nodes_ids = [], []

        for src_node, trg_node in product(range(plain_matrix.shape[0]), range(plain_matrix.shape[1])):
            if plain_matrix[src_node, trg_node] > 0:
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)
            elif src_node != trg_node:
                comp_source_nodes_ids.append(src_node)
                comp_target_nodes_ids.append(trg_node)

        edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))
        comp_edge_index = np.row_stack((comp_source_nodes_ids, comp_target_nodes_ids))

        return (torch.tensor(edge_index, dtype=torch.long, device=self._device),
                torch.tensor(comp_edge_index, dtype=torch.long, device=self._device))

    @staticmethod
    def auc(output, labels):
        preds = output.data.type_as(labels)
        return roc_auc_score(labels.cpu(), preds.cpu())


def execute_runner(runners, is_nni=False):
    res = [runner.run(verbose=2) for runner in runners]
    all_final_results = [r[1] for r in res]
    if is_nni:
        # NNI reporting. now reporting -losses, trying to maximize this. It can also be done for AUCs.
        final_loss = np.mean([all_final_results[it]["all_loss_test"] for it in range(len(all_final_results))])
        nni.report_final_result(np.exp(-final_loss))

        # Reporting results to loggers
        aggr_final_results = {}
        for new_name, old_name in zip(
                ["auc_train", "loss_train", "auc_eval", "loss_eval", "auc_test", "loss_test", "epochs_done"],
                ["auc_train", "all_loss_train", "auc_eval", "all_loss_eval", "auc_test", "all_loss_test",
                 "epochs_done"]):
            aggr_final_results[new_name] = [d[old_name] for d in all_final_results]
        runners[-1].logger.info("\nAggregated final results:")
        for name, vals in aggr_final_results.items():
            runners[-1].logger.info("*" * 15 + f"mean {name}: {np.mean(vals):.4f}")
            runners[-1].logger.info("*" * 15 + f"std {name}: {np.std(vals):.4f}")
            runners[-1].logger.info("Finished")

    # If the NNI doesn't run, only the mean results dictionary will be built. No special plots.
    all_results = {
        "all_final_output_labels_train": [d["training_output_labels"] for d in all_final_results],
        "all_final_output_labels_eval": [d["eval_output_labels"] for d in all_final_results],
        "all_final_output_labels_test": [d["test_output_labels"] for d in all_final_results],
        "final_auc_train": np.mean([d["auc_train"] for d in all_final_results]),
        "final_loss_train": np.mean([d["all_loss_train"] for d in all_final_results]),
        "final_auc_eval": np.mean([d["auc_eval"] for d in all_final_results]),
        "final_loss_eval": np.mean([d["all_loss_eval"] for d in all_final_results]),
        "final_auc_test": np.mean([d["auc_test"] for d in all_final_results]),
        "final_loss_test": np.mean([d["all_loss_test"] for d in all_final_results]),
        "average_epochs_done": np.mean([d["epochs_done"] for d in all_final_results])
    }
    return all_results


def execute_runner_for_performance(runners):
    res = [runner.run(verbose=2) for runner in runners]
    intermediate_results = [r[0] for r in res]
    final_results = [r[1] for r in res]
    all_test_ranks, all_eval_ranks, all_train_ranks = [], [], []
    all_test_labels, all_eval_labels, all_train_labels = [], [], []
    all_training_losses, all_eval_losses, all_test_losses = [], [], []
    all_training_unary_losses, all_eval_unary_losses, all_test_unary_losses = [], [], []
    all_training_pairwise_losses, all_eval_pairwise_losses, all_test_pairwise_losses = [], [], []
    all_training_binom_regs, all_eval_binom_regs, all_test_binom_regs = [], [], []
    for res_dict in final_results:
        ranks_labels_test = res_dict["test_output_labels"]
        ranks_test = list(ranks_labels_test[0, :])
        labels_test = list(ranks_labels_test[1, :])
        all_test_ranks += ranks_test
        all_test_labels += labels_test
        ranks_labels_eval = res_dict["eval_output_labels"]
        ranks_eval = list(ranks_labels_eval[0, :])
        labels_eval = list(ranks_labels_eval[1, :])
        all_eval_ranks += ranks_eval
        all_eval_labels += labels_eval
        ranks_labels_train = res_dict["training_output_labels"]
        ranks_train = list(ranks_labels_train[0, :])
        labels_train = list(ranks_labels_train[1, :])
        all_train_ranks += ranks_train
        all_train_labels += labels_train
    for intermediate_dict in intermediate_results:
        for element, key in zip([all_training_losses, all_eval_losses, all_test_losses,
                                 all_training_unary_losses, all_eval_unary_losses, all_test_unary_losses,
                                 all_training_pairwise_losses, all_eval_pairwise_losses, all_test_pairwise_losses,
                                 all_training_binom_regs, all_eval_binom_regs, all_test_binom_regs],
                                ["all_loss_train", "all_loss_eval", "all_loss_test",
                                 "unary_loss_train", "unary_loss_eval", "unary_loss_test",
                                 "pairwise_loss_train", "pairwise_loss_eval", "pairwise_loss_test",
                                 "binom_reg_train", "binom_reg_eval", "binom_reg_test"]):
            element.append(intermediate_dict[key])
    return all_test_ranks, all_test_labels, all_eval_ranks, all_eval_labels, all_train_ranks, all_train_labels, \
        all_training_losses, all_eval_losses, all_test_losses, \
        all_training_unary_losses, all_eval_unary_losses, all_test_unary_losses, \
        all_training_pairwise_losses, all_eval_pairwise_losses, all_test_pairwise_losses, \
        all_training_binom_regs, all_eval_binom_regs, all_test_binom_regs


def build_model(training_data, training_adj, training_labels, eval_data, eval_adj, eval_labels,
                test_data, test_adj, test_labels, learning_hyperparams, class_weights,
                graph_params, dumping_name, is_nni=False, device=1):
    activations = [learning_hyperparams.activation] * (len(learning_hyperparams.hidden_layers) + 1)
    conf = {"model": learning_hyperparams.model, "hidden_layers": learning_hyperparams.hidden_layers,
            "dropout": learning_hyperparams.dropout, "lr": learning_hyperparams.learning_rate,
            "weight_decay": learning_hyperparams.l2_regularization, "training_mat": training_data,
            "training_adj": training_adj, "training_labels": training_labels,
            "eval_mat": eval_data, "eval_adj": eval_adj, "eval_labels": eval_labels,
            "test_mat": test_data, "test_adj": test_adj, "test_labels": test_labels,
            "optimizer": learning_hyperparams.optimizer, "epochs": learning_hyperparams.epochs,
            "activations": activations, "loss_coeffs": learning_hyperparams.loss_coefficients,
            "unary": learning_hyperparams.unary_loss_type,
            "edge_normalization": learning_hyperparams.edge_normalization}

    products_path = os.path.join(os.getcwd(), "logs", *dumping_name, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    check_make_dir(products_path)

    logger = multi_logger([
        PrintLogger("MyLogger", level=logging.DEBUG),
        FileLogger("results_" + dumping_name[1], path=products_path, level=logging.INFO)], name=None)

    runner = ModelRunner(conf, logger=logger, weights=class_weights, graph_params=graph_params,
                         early_stop=learning_hyperparams.early_stop, is_nni=is_nni, tmp_path=products_path,
                         device=device)
    return runner


def run_pygon(feature_matrices, adj_matrices, labels, graph_params, learning_hyperparams, iterations=3,
              dumping_name='', is_nni=False, device=1, check='CV', purpose='checkout'):
    """
    Implement PYGON model using the specified hyper-parameters and the input graphs.
    Here we use at least 20 graphs (an arbitrary choice). We either split the graphs into training (60%), eval (20%) and
    test (20%) randomly for each iteration (check='split') or run a 5-fold cross validation with these set sizes
    (check='CV').
    One can either choose purpose='checkout' for running PYGON for basic testing, or run a more thorough performance
    test (theoretically, purpose='performance', but any string will be OK).
    """
    class_weights = {0: (float(graph_params['vertices']) / (graph_params['vertices'] - graph_params['subgraph_size'])),
                     1: (float(graph_params['vertices']) / graph_params['subgraph_size'])}
    runners = []
    assert len(labels) >= 20, f"Not enough graphs. Expected 20, got {len(labels)}"
    if check == 'split':
        for it in range(iterations):
            rand_test_indices = np.random.choice(len(labels), round(len(labels) * 0.4), replace=False)  # Test + Eval
            train_indices = np.delete(np.arange(len(labels)), rand_test_indices)
            rand_eval_indices = rand_test_indices[:round(0.5 * len(rand_test_indices))]
            test_indices = rand_test_indices[round(0.5 * len(rand_test_indices)):]

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            eval_features = [feature_matrices[j] for j in rand_eval_indices]
            eval_adj = [adj_matrices[j] for j in rand_eval_indices]
            eval_labels = [labels[j] for j in rand_eval_indices]

            test_features = [feature_matrices[j] for j in test_indices]
            test_adj = [adj_matrices[j] for j in test_indices]
            test_labels = [labels[j] for j in test_indices]

            runner = build_model(training_features, training_adj, training_labels,
                                 eval_features, eval_adj, eval_labels,
                                 test_features, test_adj, test_labels,
                                 learning_hyperparams, class_weights, graph_params, dumping_name,
                                 is_nni=is_nni, device=device)
            runners.append(runner)
    elif check == 'CV':
        # 5-Fold CV, one fold (4 graphs) for test graphs, one fold for eval and the rest are training.
        # From choosing the folds, the choice of eval is done such that every run new graphs will become eval.
        all_indices = np.arange(len(labels))
        np.random.shuffle(all_indices)
        folds = np.array_split(all_indices, 5)
        for it in range(len(folds)):
            test_fold = folds[it]
            eval_fold = folds[(it + 1) % 5]
            train_indices = np.hstack([folds[(it + 2 + j) % 5] for j in range(3)])

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            eval_features = [feature_matrices[j] for j in eval_fold]
            eval_adj = [adj_matrices[j] for j in eval_fold]
            eval_labels = [labels[j] for j in eval_fold]

            test_features = [feature_matrices[j] for j in test_fold]
            test_adj = [adj_matrices[j] for j in test_fold]
            test_labels = [labels[j] for j in test_fold]

            runner = build_model(training_features, training_adj, training_labels,
                                 eval_features, eval_adj, eval_labels,
                                 test_features, test_adj, test_labels,
                                 learning_hyperparams, class_weights, graph_params, dumping_name,
                                 is_nni=is_nni, device=device)
            runners.append(runner)
    elif check == '2CV':
        all_indices = np.arange(len(labels))
        np.random.shuffle(all_indices)
        folds = np.array_split(all_indices, 5)
        for it in range(2):
            test_fold = folds[it]
            eval_fold = folds[(it + 1) % 5]
            train_indices = np.hstack([folds[(it + 2 + j) % 5] for j in range(3)])

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            eval_features = [feature_matrices[j] for j in eval_fold]
            eval_adj = [adj_matrices[j] for j in eval_fold]
            eval_labels = [labels[j] for j in eval_fold]

            test_features = [feature_matrices[j] for j in test_fold]
            test_adj = [adj_matrices[j] for j in test_fold]
            test_labels = [labels[j] for j in test_fold]

            runner = build_model(training_features, training_adj, training_labels,
                                 eval_features, eval_adj, eval_labels,
                                 test_features, test_adj, test_labels,
                                 learning_hyperparams, class_weights, graph_params, dumping_name,
                                 is_nni=is_nni, device=device)
            runners.append(runner)
    else:
        raise ValueError(f"Wrong value for 'check', {check}")
    if purpose == "checkout":
        res = execute_runner(runners, is_nni=is_nni)
    else:  # purpose = "performance"
        res = execute_runner_for_performance(runners)
    return res
