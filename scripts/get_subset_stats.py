"""Get stats for subsets of the data.

Uses the results format produced by test.py.
Dictionary where keys are study_num, values are dicts with 'label', 'pred' keys.
"""
import argparse
import numpy as np
import pickle
import sklearn.metrics as sk_metrics

from collections import Counter


def main(args):
    with open(args.pkl_path, 'rb') as pkl_fh:
        ctpes = pickle.load(pkl_fh)

    with open(args.results_path, 'rb') as pkl_fh:
        results_dict = pickle.load(pkl_fh)

    print('Got {} results'.format(len(results_dict)))
    c = Counter()
    for pe_type in ('subsegmental', 'central', 'segmental'):
        probs, labels = [], []
        for study_num, result_dict in results_dict.items():
            ctpe = find(study_num, ctpes)
            c[ctpe.type] += 1
            if ctpe.type == pe_type:
                probs.append(result_dict['pred'])
                labels.append(int(ctpe.is_positive))

        print('Counts: {}'.format(dict(c)))

        print('Got {} total.'.format(len(probs)))
        for prob, label in zip(probs, labels):
            print('{:.3f}: {}'.format(prob, label))

        # Get stats
        probs = np.array(probs)
        labels = np.array(labels)
        print('*** Stats for {} PE ***'.format(pe_type))
        print('AUROC: {}'.format(sk_metrics.roc_auc_score(labels, probs)))
        print('AUPRC: {}'.format(sk_metrics.precision_score(labels, probs)))


def find(study_num, sl):
    for s in sl:
        if s.study_num == study_num:
            return s
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl_path', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/cc_data/series_list.pkl')
    parser.add_argument('--results_path', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/norahborus/results/pe1007_test_all_20181011_125744/preds.pickle')

    main(parser.parse_args())
