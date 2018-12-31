import torch
from torch.nn import ReLU
import sys, json, time, cv2, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt

from utils import get_model_and_loader, transform_data
from evaluate.get_best_model import get_best_models
from guided_backprop import get_gbp
from gradcam import get_grad

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modeldir', type=str, required=True)
    parser.add_argument('--datadir', type=str, default="/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/30AprData/localized")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--outputdir', type=str, default="/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/cams")
    parser.add_argument('--mode', type=str, default='GBP')
    return parser


def load_models(model_paths, split):
    models, loaders, _ = zip(*[get_model_and_loader(model_path, split) for model_path in model_paths])
    return models, loaders


if __name__ == '__main__':
    args = get_parser().parse_args()

    best_models = get_best_models(args.modeldir, 1, verbose=False)
    print("Best models: ", best_models)
    model_paths = [path for loss, path in best_models]

    models, loaders = load_models(model_paths, args.split)

    label_names = ["PE"] # Change if using multilabel
    thresholds = np.array([[0.5]]) # Change if using multilabel

    outputdir = Path(args.outputdir)
    if not outputdir.exists():
        outputdir.mkdir()

    for model, loader in zip(models, loaders): 
        if args.mode == 'GBP':
            get_gbp(model, loader, label_names, thresholds, outputdir)
        elif args.mode == 'GRAD':
            get_grad(model, loader, label_names, thresholds, outputdir)


