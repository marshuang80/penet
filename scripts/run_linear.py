"""Run linear regression to combine models' window-wise predictions."""

import argparse
import numpy as np
import pandas as pd
import pickle
import random
import util

from collections import defaultdict
from sklearn import linear_model
from sklearn import metrics as sk_metrics


def main(args):

    # Concatenate all examples into one big dataframe
    with open(args.csv_list, 'r') as fh:
        csv_paths = [l.strip() for l in fh.readlines() if l.strip()]
    df = pd.concat([pd.read_csv(csv_path) for csv_path in csv_paths])

    # Get list of predictions from the CSV
    id2prob = defaultdict(list)  # (series_idx, slice_idx) -> [prob1, ..., probN]
    id2label = {}
    for _, row in df.iterrows():
        row_dict = dict(row)
        # Get ID as tuple (series_num, slice_idx)
        row_id = (int(row_dict['series_num']), int(row_dict['slice_idx']))
        # Get prob
        id2prob[row_id].append(float(row_dict['prob']))
        # Get label
        label = int(row_dict['label'])
        if row_id in id2label and label != id2label[row_id]:
            raise RuntimeError('Mismatched labels for series {}, slice {}'.format(*row_id))
        id2label[row_id] = label

    x_test, y_test, id_test = [], [], []
    pos_series, neg_series = get_pos_neg_series(id2label)
    if not args.is_test:
        # Stratified random split into train and test set
        num_val = int((1 - args.train_frac) * (len(pos_series) + len(neg_series)))
        val_series = set(pos_series[:num_val // 2]) | set(neg_series[:num_val // 2])
        x_train, y_train = [], []

        for i in id2prob:
            if i[0] not in val_series:
                x_train.append(id2prob[i])
                y_train.append(id2label[i])
            else:
                x_test.append(id2prob[i])
                y_test.append(id2label[i])
                id_test.append(i)

        # Randomly sample hyperparameters
        alphas = log_sample(args.num_alphas, args.alpha_min_exp, args.alpha_max_exp)
        clf = None
        best_metric_val = float('-inf')
        for alpha in alphas:
            # Fit a linear classifier (Lasso, ElasticNet, or Ridge)
            clf = linear_model.__dict__[args.model](alpha)
            clf.fit(x_train, y_train)

            # Measure performance at the window level
            y_pred = clf.predict(x_test)
            metrics = {'R^2': clf.score(x_test, y_test),
                       'AUROC': sk_metrics.roc_auc_score(y_test, y_pred),
                       'AUPRC': sk_metrics.average_precision_score(y_test, y_pred)}

            print('Alpha: {}'.format(alpha))
            for k, v in metrics.items():
                print('Window-level {}: {}'.format(k, v))

            # Save model with best AUROC
            if metrics[args.best_metric_name] > best_metric_val:
                best_metric_val = metrics[args.best_metric_name]
                print('Saving model to {} (alpha={})'.format(args.model_path, alpha))
                with open(args.model_path, 'wb') as pkl_fh:
                    pickle.dump(clf, pkl_fh)
    else:
        for i in id2prob:
            x_test.append(id2prob[i])
            y_test.append(id2label[i])
            id_test.append(i)

    # Load best for series-level performance
    print('Loading test model from {}...'.format(args.model_path))
    with open(args.model_path, 'rb') as pkl_fh:
        clf = pickle.load(pkl_fh)
    print('Weights: {}, Bias: {}'.format(clf.coef_, clf.intercept_))

    # Predict
    y_pred = clf.predict(x_test)

    # Combine window-level predictions to series-level predictions
    print('Combining window-level predictions to series-level predictions...')
    series2maxprob = defaultdict(float)
    series2label = defaultdict(int)

    for i, (series_num, slice_idx) in enumerate(id_test):
        series2maxprob[series_num] = max(series2maxprob[series_num], y_pred[i])
        series2label[series_num] = max(series2label[series_num], y_test[i])

    # Measure performance at the series level
    y_pred_series, y_test_series = get_prob_list(series2maxprob, series2label)
    print('Series-level AUPRC: {}'.format(sk_metrics.average_precision_score(y_test_series, y_pred_series)))
    print('Series-level AUROC: {}'.format(sk_metrics.roc_auc_score(y_test_series, y_pred_series)))

    # Get best threshold using Youden's J score (use validation set only)
    if not args.is_test:
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_test_series, y_pred_series)

        for f, t, thresh in zip(fpr, tpr, thresholds):
            print('Threshold: {}, FP: {}, FN: {}, J: {}'
                  .format(thresh,
                          sum(1 for pred, gold in zip(y_pred_series, y_test_series)
                              if pred > thresh and gold == 0),
                          sum(1 for pred, gold in zip(y_pred_series, y_test_series)
                              if pred < thresh and gold == 1),
                          t - f))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print('Youden\'s J-score Best Threshold: {}'.format(optimal_threshold))
        print('Test set size: {}'.format(len(y_pred_series)))
        print('Num. positive: {}'.format(sum(1 for y in y_test_series if y == 1)))

    # Save predictions to CSV
    df_rows = [{'dset_path': '/series/{:05d}'.format(series_num),
                'probability': series2maxprob[series_num]}
               for series_num in series2maxprob]
    df = pd.DataFrame(df_rows)
    df[['dset_path', 'probability']].to_csv('results/ensemble_probs.csv', index=False)


def get_pos_neg_series(id2label):
    """Get list of positive and list of negative series numbers."""
    pos_series, neg_series = [], []

    series_nums = list(set(series_num for series_num, _ in id2label))
    for series_num in series_nums:
        # Check if any window in this series is labeled positive
        is_positive = False
        for (s, _), y in id2label.items():
            if s == series_num and y == 1:
                is_positive = True
                break

        if is_positive:
            pos_series.append(series_num)
        else:
            neg_series.append(series_num)

    return pos_series, neg_series


def log_sample(n, min_exp, max_exp):
    """Randomly sample with a logarithmic (base 10) search scale."""
    samples = []
    for _ in range(n):
        # Sample randomly along a logarithm search scale
        random_exp = min_exp + random.random() * (max_exp - min_exp)
        samples.append(10 ** random_exp)

    return samples


def get_prob_list(series2maxprob, series2label):
    y_pred_series, y_test_series = [], []
    for series_num in series2maxprob:
        y_pred_series.append(series2maxprob[series_num])
        y_test_series.append(series2label[series_num])

    return y_pred_series, y_test_series


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear classifier for ensemble')

    # Training set parameters
    parser.add_argument('--csv_list', type=str, required=True,
                        help='Path to file with one CSV path per line, each with model predictions.')
    parser.add_argument('--model_path', type=str, default='model.pkl',
                        help='Path to save model.')
    parser.add_argument('--is_test', type=util.str_to_bool, default=False,
                        help='Is test inference mode.')
    parser.add_argument('--model', default='ElasticNet', choices=('Lasso', 'ElasticNet', 'Ridge'),
                        help='Linear model to use.')
    parser.add_argument('--train_frac', default=0.83, type=float,
                        help='Fraction of total examples to use for training (rest for validation).')

    # Hyperparameter search
    parser.add_argument('--best_metric_name', default='R^2', choices=('AUROC', 'AUPRC', 'R^2'),
                        help='Metric to maximize when choosing best hyperparameters.')
    parser.add_argument('--alpha_min_exp', default=-7, type=int,
                        help='Minimum exponent (base 10) for alpha search space.')
    parser.add_argument('--alpha_max_exp', default=-3, type=int,
                        help='Maximum exponent (base 10) for alpha search space.')
    parser.add_argument('--num_alphas', default=100, type=int,
                        help='Number of alpha hyperparameters to try.')

    random.seed(7)

    main(parser.parse_args())
