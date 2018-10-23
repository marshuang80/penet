"""
predict.py
	load a model and output predictions on a specified dataset
"""
import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np

from utils import compute_probs_from_paths


def get_parser():
    parser = argparse.ArgumentParser()

    # Predict Parameters
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, default="test")

    return parser


def predict(model_paths, split, save=False):

    all_probs = []
    for model_path in model_paths:

        probs = compute_probs_from_paths(model_path, split)

        all_probs.append(probs)

    ensemble_probs = np.mean(all_probs, axis=0)

    if save:
        np.save('probs.npy', ensemble_probs)

    return ensemble_probs


if __name__ == "__main__":

    args = get_parser().parse_args()

    assert args.dataset in ['valid', 'test', 'radio-test-mini'], f"{args.dataset} data split not supported"

    predict(args.model_paths, args.dataset, save=True)

