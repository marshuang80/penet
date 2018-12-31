import argparse
import numpy as np
import os
import pandas as pd
import sklearn.metrics as sk_metrics
import util
import xgboost as xgb

from sklearn.datasets import load_svmlight_file


def train_classifier(args):
    """Train an XGBoost classifier to get classification probabilities from segmentation outputs.
    Args:
        args: Command-line arguments.
    """

    train_filepath, ext = os.path.splitext(args.train_path)
    train_dir = os.path.dirname(args.train_path)
    val_dir = os.path.dirname(args.val_path)

    if ext == '.svmlight':
        train_inputs, train_labels = load_svmlight_file(args.train_path)
        val_inputs, val_labels = load_svmlight_file(args.val_path)
        train_inputs = train_inputs.todense()
        val_inputs = val_inputs.todense()
    elif ext == '.npy':
        train_inputs = np.load(args.train_path)
        train_labels = np.load(os.path.join(train_dir, 'labels.npy'))
        val_inputs = np.load(args.val_path)
        val_labels = np.load(os.path.join(val_dir, 'labels.npy'))
    else:
        train_inputs = pd.read_csv(args.train_path)
        train_labels = pd.read_csv(os.path.join(train_dir, 'labels.csv'))
        val_inputs = pd.read_csv(args.val_path)
        val_labels = pd.read_csv(os.path.join(val_dir, 'labels.csv'))

    # Pad to max length of train and test inputs
    max_features = max(train_inputs.shape[1], val_inputs.shape[1])
    train_inputs = np.pad(train_inputs, [(0, 0), (0, max_features - train_inputs.shape[1])], mode='constant')
    val_inputs = np.pad(val_inputs, [(0, 0), (0, max_features - val_inputs.shape[1])], mode='constant')

    train_data = xgb.DMatrix(train_inputs, label=train_labels, missing=0.)
    val_data = xgb.DMatrix(val_inputs, label=val_labels, missing=0.)

    eval_list = [(train_data, 'train'), (val_data, 'eval')]

    parameters = {'max_depth': args.max_depth,
                  'eta': args.learning_rate,
                  'silent': (not args.verbose),
                  'lambda': args.weight_decay,
                  'eval_metric': args.eval_metric,
                  'gamma': args.min_split_loss,
                  'min_child_weight': args.min_child_weight,
                  'subsample': args.subsample,
                  'colsample_bytree': (1 - args.dropout)}
    num_round = args.num_iters

    # If saved model exists, load and continue training
    saved_model = None
    if not args.overwrite and os.path.isfile(os.path.join(args.model_dir, args.name + '.model')):
        saved_model = xgb.Booster()
        saved_model.load_model(os.path.join(args.model_dir, args.name + '.model'))

    classifier = xgb.train(parameters, train_data, num_round, eval_list,
                           early_stopping_rounds=args.early_stopping_rounds,
                           xgb_model=saved_model)

    # Dump model in human readable form
    classifier.dump_model(os.path.join(args.model_dir, 'model.raw.txt'))

    # Save model
    classifier.save_model(os.path.join(args.model_dir, args.name + '.model'))


def run_classifier(args):
    """Run a trained XGBoost model and get metrics."""

    test_filepath, ext = os.path.splitext(args.test_path)
    test_dir = os.path.dirname(args.test_path)

    if ext == '.svmlight':
        test_inputs, test_labels = load_svmlight_file(args.test_path)
        test_inputs = test_inputs.todense()
    elif ext == '.npy':
        test_inputs = np.load(args.test_path)
        test_labels = np.load(os.path.join(test_dir, 'labels.npy'))
    else:
        test_inputs = pd.read_csv(args.test_path)
        test_labels = pd.read_csv(os.path.join(test_dir, 'labels.csv'))

    test_data = xgb.DMatrix(test_inputs, label=test_labels, missing=0.)

    if not os.path.exists(os.path.join(args.model_dir, args.name + '.model')):
        raise IOError("Could not load model from path {}.".format(os.path.join(args.model_dir, args.name + '.model')))

    model = xgb.Booster()
    model.load_model(os.path.join(args.model_dir, args.name + '.model'))
    test_probs = model.predict(test_data)

    metrics = {'Accuracy': sk_metrics.accuracy_score(test_labels, test_probs > 0.5),
               'AUROC': sk_metrics.roc_auc_score(test_labels, test_probs),
               'AUPRC': sk_metrics.average_precision_score(test_labels, test_probs)}

    util.print_err('Performance of model at {}:'.format(os.path.join(args.model_dir, args.name + '.model')))
    for k, v in metrics.items():
        print('{} = {:.4f}'.format(k, v))

    test_probs_path = os.path.join(os.path.dirname(args.test_path), '{}_preds.npy'.format(args.name))
    util.print_err('Saving predictions to {}...'.format(test_probs_path))
    np.save(test_probs_path, test_probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str,
                        default='/sailhome/tanay/results_train/dirty5kin/final_tests_20181210_150457/xgb/inputs.npy',
                        help='Path to the file with training data.')
    parser.add_argument('--val_path', type=str,
                        default='/sailhome/tanay/results/dirty5kin/final_tests_20181209_014003/xgb/inputs.npy',
                        help='Path to the file with val data.')
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Maximum tree depth for XGBoost base learners.')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Boosting learning rate that controls the weighting of new trees.')
    parser.add_argument('--min_split_loss', type=float, default=0.,
                        help='Minimum loss reduction required to make a further partition. \
                        Value should be in [0, +inf]. The larger, the more conservative the model will be.')
    parser.add_argument('--min_child_weight', type=float, default=1.,
                        help='Minimum sum of instance weight required to make a further partition. \
                        Value should be in [0, +inf]. The larger, the more conservative the model will be.')
    parser.add_argument('--subsample', type=float, default=1.,
                        help='Ratio of examples to randomly subsample.')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Ratio of columns to drop when constructing each tree.')
    parser.add_argument('--eval_metric', type=str, choices=('auc', 'error', 'logloss'), default='auc',
                        help='Evaluation metric to use. "auc" is area under the curve. \
                        "error" is the binary classification error rate using a 0.5 threshold. \
                        "logloss" is the negative log-likelihood.')
    parser.add_argument('--num_iters', type=int, default=20,
                        help='Number of training iterations/boosted trees to fit.')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='L2 regularization term on weights.')
    parser.add_argument('--verbose', type=util.str_to_bool, default=True,
                        help='Print messages while training.')
    parser.add_argument('--early_stopping_rounds', type=int, default=5,
                        help='If validation error does not decrease for given number of iterations, stop training.')
    parser.add_argument('--overwrite', type=util.str_to_bool, required=True,
                        help='If true, overwrite and retrain the model with the same experiment name.')
    parser.add_argument('--name', type=str, required=True,
                        help='Filename of stored xgb model to be stored/loaded.')
    parser.add_argument('--model_dir', type=str, default='/sailhome/tanay/results_train/dirty5kin/xgb_classifier/',
                        help='Directory to save model and results.')
    parser.add_argument('--run_classifier', type=util.str_to_bool,
                        help='Run a trained classifier and get metrics.')
    parser.add_argument('--test_path', type=str,
                        help='Path to the file with test data.')

    args_ = parser.parse_args()

    if args_.run_classifier:
        if not hasattr(args_, "test_path"):
            raise RuntimeError("Must specify path for dataset to run a trained model on.")
        run_classifier(args_)
    else:
        os.makedirs(args_.model_dir, exist_ok=True)
        train_classifier(args_)
