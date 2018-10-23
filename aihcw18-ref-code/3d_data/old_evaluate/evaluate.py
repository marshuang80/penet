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
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, auc, f1_score, cohen_kappa_score, roc_curve


import matplotlib.pyplot as plt
from data.loader import load_data
from utils import get_loader


def evaluate(probs, datadir, split, metric, verbose):
    assert(np.all(probs >= 0)), "probabilities must be at least 0"
    assert(np.all(probs <= 1)), "probabilities must be at most 1"
    loader = get_loader(datadir, split)
    #groundtruth = np.concatenate([labels.numpy() for _, labels in loader], axis=0)
    labels = ['normal', 'abnormal'] # TODO: get these from somewhere else?
    groundtruth  = loader.dataset.labels.flatten()
    print("Starting")
    preds = (probs > 0.5).astype(int)
    PR_AUCs = []
    ROC_AUCs = []
    F1s = []
    accs = []
    counts = []
    kappas = []
    def compute_metrics_for_class(i):
        p, r, t = precision_recall_curve(groundtruth, probs[:, i])
        PR_AUC = auc(r, p)
        ROC_AUC = roc_auc_score(groundtruth, probs[:, i])
        F1 = f1_score(groundtruth, preds[:, i])
        acc = accuracy_score(groundtruth, preds[:, i])
        kappa = cohen_kappa_score(preds, groundtruth)
        count = np.sum(groundtruth)
        return PR_AUC, ROC_AUC, F1, acc, count,kappa

    num_classes = 1
    for i in range(num_classes):
        try:
            PR_AUC, ROC_AUC, F1, acc, count, kappa = compute_metrics_for_class(i)
        except ValueError:
            continue
        PR_AUCs.append(PR_AUC)
        ROC_AUCs.append(ROC_AUC)
        F1s.append(F1)
        accs.append(acc)
        counts.append(count)
        kappas.append(kappa)

        if verbose:
            # NOTE: this is a bit weird for binary classification.
            print(f'Class: {labels[i]} Count: {int(count)} PR AUC: {PR_AUC:.4f} ROC AUC: {ROC_AUC:.4f} F1: {F1:.3f} Acc: {acc:.3f}')
    print(PR_AUCs, counts)
    avg_PR_AUC = np.average(PR_AUCs, weights=counts)
    avg_ROC_AUC = np.average(ROC_AUCs, weights=counts)
    avg_F1 = np.average(F1s, weights=counts)
    avg_kappa = np.average(kappas, weights=counts)
    avg_acc = np.average(accs, weights=counts)
    if verbose:
        print(f'Avg PR AUC: {avg_PR_AUC:.3f}')
        print(f'Avg ROC AUC: {avg_ROC_AUC:.3f}')
        print(f'Avg F1: {avg_F1:.3f}')

    metric_dict = {'f1':      avg_F1,
                   'pr_auc':  avg_PR_AUC,
                   'roc_auc': avg_ROC_AUC,
                   'kappa': avg_kappa,
                   'acc': avg_acc}

    fpr, tpr, _ = roc_curve(groundtruth, probs[:,0])
    plt.plot(fpr, tpr)
    plt.plot(0.075, 0.75, 'ro')
    plt.savefig("/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/roc_orig_groundtruth.png")
    plt.close()


    return metric_dict[metric]


def radiologist_evaluate(probs, datadir, split, verbose):

    original_loader = get_loader(datadir, 'test')
    rad_loader = get_loader(datadir, split)

    original_paths = original_loader.dataset.img_paths
    original_labels = original_loader.dataset.labels.flatten()
    rad_paths = rad_loader.dataset.img_paths
    rad_labels = rad_loader.dataset.labels.flatten()

    original_df = pd.DataFrame({"Paths": original_paths, "OrigLabels": original_labels})
    rad_df = pd.DataFrame({"Paths": rad_paths, "RadLabels": rad_labels})

    combined_df = rad_df.merge(original_df, on="Paths")
    combined_df["ModelLabels"] = (probs > 0.5).astype(int)

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

    return parser


if __name__=='__main__':

    args = get_parser().parse_args()
    assert args.split in ['valid', 'test', 'radio-test-mini'], f'{args.split} data split not supported'
    assert args.metric in ['f1', 'pr_auc', 'roc_auc'], f'{args.metric} metric not supported'

    probs = np.load(args.probs_path)

    if args.split in ['valid', 'test', 'radio-test-mini']:
        evaluate(probs, args.datadir, args.split, args.metric, args.verbose)

    else:

        radiologist_evaluate(probs, args.datadir, args.split, args.verbose)
