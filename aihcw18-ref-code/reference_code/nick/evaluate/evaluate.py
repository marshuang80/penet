"""
evaluate.py
    input predictions output by predict.py and compute metrics against
    groundtruth labels
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, auc, f1_score, cohen_kappa_score

from data.loader import load_data
from utils import get_loader

def find_threshold(probs, datadir, metric, verbose):
    best_metric = 0.0
    for i in range(1, 100):
        th = float(i)/100
        new_metric = 0
        if metric == 'kappa':
            _, new_metric = radiologist_evaluate(probs, datadir, 'valid', verbose, th)
        else:
            new_metric = evaluate(probs, datadir, 'valid', metric, verbose, th)
        if new_metric > best_metric:
            best_metric = new_metric
            best_threshold = th
    print('Best {}: {:.3f}'.format(metric, best_metric))
    print('Best threshold: {:.2f}'.format(best_threshold))
    return best_threshold


def evaluate(probs, datadir, split, metric, verbose, th = 0.5):
    assert(np.all(probs >= 0)), "probabilities must be at least 0"
    assert(np.all(probs <= 1)), "probabilities must be at most 1"

    loader = get_loader(datadir, split)
    groundtruth = np.concatenate([labels.numpy() for _, labels in loader], axis=0)

    labels = ['normal', 'abnormal'] # TODO: get these from somewhere else?

    preds = (probs > th).astype(int)

    PR_AUCs = []
    ROC_AUCs = []
    F1s = []
    accs = []
    counts = []


    def compute_metrics_for_class(i):
        p, r, t = precision_recall_curve(groundtruth[:, i], probs[:, i])
        PR_AUC = auc(r, p)
        ROC_AUC = roc_auc_score(groundtruth[:, i], probs[:, i])
        F1 = f1_score(groundtruth[:, i], preds[:, i])
        acc = accuracy_score(groundtruth[:, i], preds[:, i])
        count = np.sum(groundtruth[:, i])
        return PR_AUC, ROC_AUC, F1, acc, count


    num_classes = groundtruth.shape[1]
    for i in range(num_classes):
        try:
            PR_AUC, ROC_AUC, F1, acc, count = compute_metrics_for_class(i)
        except ValueError:
            continue
        PR_AUCs.append(PR_AUC)
        ROC_AUCs.append(ROC_AUC)
        F1s.append(F1)
        accs.append(acc)
        counts.append(count)

        if verbose:
            # NOTE: this is a bit weird for binary classification.
            print(f'Class: {labels[i]} Count: {int(count)} PR AUC: {PR_AUC:.4f} ROC AUC: {ROC_AUC:.4f} F1: {F1:.3f} Acc: {acc:.3f}')

    avg_PR_AUC = np.average(PR_AUCs, weights=counts)
    avg_ROC_AUC = np.average(ROC_AUCs, weights=counts)
    avg_F1 = np.average(F1s, weights=counts)
    avg_acc = np.average(accs, weights=counts)

    if verbose:
        print(f'Avg PR AUC: {avg_PR_AUC:.3f}')
        print(f'Avg ROC AUC: {avg_ROC_AUC:.3f}')
        print(f'Avg F1: {avg_F1:.3f}')
        print(f'Avg acc: {avg_acc:.3f}')

    metric_dict = {'f1':      avg_F1,
                   'pr_auc':  avg_PR_AUC,
                   'roc_auc': avg_ROC_AUC,
                   'acc': avg_acc}

    return metric_dict[metric]


def radiologist_evaluate(probs, datadir, split, verbose, th = 0.5):

    original_loader = get_loader(datadir, split) # TODO: needs to be changed to 'test' when we get radio-labels
    rad_loader = get_loader(datadir, split)

    original_paths = original_loader.dataset.img_paths
    original_labels = original_loader.dataset.labels.flatten()
    rad_paths = rad_loader.dataset.img_paths
    rad_labels = rad_loader.dataset.labels.flatten()

    original_df = pd.DataFrame({"Paths": original_paths, "OrigLabels": original_labels})
    rad_df = pd.DataFrame({"Paths": rad_paths, "RadLabels": rad_labels})

    combined_df = rad_df.merge(original_df, on="Paths")
    combined_df["ModelLabels"] = (probs > th).astype(int)

    rad_score = cohen_kappa_score(combined_df["RadLabels"], combined_df["OrigLabels"])
    model_score = cohen_kappa_score(combined_df["ModelLabels"], combined_df["OrigLabels"])

    return rad_score, model_score

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--probs_path', default='probs.npy')
    parser.add_argument('-d', '--datadir', default='data')
    parser.add_argument('-s', '--split', default='valid')
    parser.add_argument('-m', '--metric', default='f1')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--max', default=None)

    return parser


if __name__=='__main__':
    
    args = get_parser().parse_args()
    assert args.split in ['valid', 'test', 'radio-test-mini'], f'{args.split} data split not supported'
    assert args.metric in ['f1', 'pr_auc', 'roc_auc', 'acc', 'kappa'], f'{args.metric} metric not supported'

    probs = np.load(args.probs_path)
    if args.max == None:
        th = 0.5
    else:
        th = find_threshold(probs, args.datadir, args.max, args.verbose)

    if args.split in ['valid', 'test']:

        evaluate(probs, args.datadir, args.split, args.metric, args.verbose, th)

    else:

        radiologist_evaluate(probs, args.datadir, args.split, args.verbose, th)
