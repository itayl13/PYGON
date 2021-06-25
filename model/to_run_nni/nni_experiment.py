import nni
import logging
from torch.optim import Adam, SGD
from torch.nn.functional import relu, elu, tanh
import argparse

from __init__ import *
logger = logging.getLogger("NNI_logger")

activations_by_name = {
    "ReLU": relu,
    "ELU": elu,
    "tanh": tanh
}


def run_trial(params, v, p, sg_sz, d, subgraph, device=0):
    """
    Run a PYGON trial using the hyper-parameters chosen by the NNI tuner.
    NOTE: we assume that the graphs on which the calculations are done have already been created.
    """
    features = params["input_vec"]
    hidden_layers = [int(params["h1_dim"]), int(params["h2_dim"]), int(params["h3_dim"]), int(params["h4_dim"])]
    dropout = params["dropout"]
    reg_term = params["regularization"]
    lr = params["learning_rate"]
    optimizer = Adam if params["optimizer"] == "Adam" else SGD
    activation = activations_by_name[params["activation"]]
    input_params = {
'	"model": "PYGON",  # or "GAT"
        "hidden_layers": hidden_layers,
        "epochs": 1000,
        "dropout": dropout,
        "lr": lr,
        "regularization": reg_term,
        "early_stop": True,
        "optimizer": optimizer,
        "activation": activation,
        "edge_normalization": "correct"
    }
    model = PYGONMain(v, p, sg_sz, d, features=features, subgraph=subgraph, nni=True, device=device)
    model.train(input_params)


def main(v, p, sg_sz, d, subgraph, device):
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params, v, p, sg_sz, d, subgraph, device)
    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("-p", type=float, default=0.5)
    parser.add_argument("--size", type=int)
    parser.add_argument("-d", type=bool, default=False)
    parser.add_argument("--subgraph", type=str, default="clique")
    parser.add_argument("--device", type=int, default=0)
    args = vars(parser.parse_args())
    main(args['n'], args['p'], args['size'], args['d'], args['subgraph'], args['device'])
