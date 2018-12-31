"""
predict.py
	load a model and output predictions on a specified dataset
"""
import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import os
import numpy as np

from utils import compute_probs_from_paths


def get_parser():
    parser = argparse.ArgumentParser()

    # Predict Parameters
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, default="valid")

    return parser


def predict(model_paths, split, save=False, save_dir = "/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/", n=1):
    save=True
    print("predict: ")
    all_probs = []
    for model_path in model_paths:

        probs, _ , _ = compute_probs_from_paths(model_path, split)
        print("Probs shape: ",probs.shape)
        all_probs.append(probs)

    try:
        ensemble_probs = np.mean(all_probs, axis=0)
        #print("Ensemble probs: ", ensemble_probs.shape)
        if save:
            save_dir = '{}/{}_ensemble_{}_probs'.format(save_dir,split, n)
            print("Save dir for probs: ", save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save('{}/probs.npy'.format(save_dir), ensemble_probs)

        print("Done with predict function")
        return ensemble_probs
    except Exception as e:
        print("Predict exception: ",e)


if __name__ == "__main__":

    args = get_parser().parse_args()

    assert args.dataset in ['valid', 'test', 'radio-test-mini'], f"{args.dataset} data split not supported"

    predict(args.model_paths, args.dataset, save=True)

