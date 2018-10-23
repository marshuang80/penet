import argparse

from get_best_model import get_best_models
from predict import predict
from evaluate import evaluate, radiologist_evaluate, find_threshold

if __name__ == "__main__":
	# Define command-line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',
                        help="path to get best from")
    parser.add_argument('-d', '--datadir', default='')
    parser.add_argument('-n', '--n', type=int, default=20)
    parser.add_argument('-s', '--split', default='valid')
    parser.add_argument('-m', '--metric', default='f1')
    parser.add_argument('--max', default=None)
    args = parser.parse_args()

    # Get best models based on validation loss
    best_models = get_best_models(args.folder, args.n, verbose=False)
    model_paths = [path for loss, path in best_models]

    # Get model probabilities on dataset
    model_probs = predict(model_paths, args.split, save=False)
    if args.max == None:
        th = 0.5
    else:
        model_probs_valid = predict(model_paths, 'valid', save=False)
        th = find_threshold(model_probs_valid, args.datadir, args.max, verbose=False)
    # Get model score: since we don't have radiologist labels yet, just compute metric
    if args.metric == 'kappa':
        _, model_score = radiologist_evaluate(model_probs, args.datadir, args.split, verbose=False, th=th)
        print(f'threshold: {th:.2f}')
        print(f'Model score: {model_score:.3f}')
    else:
        score = evaluate(model_probs, args.datadir, args.split, args.metric, verbose=False, th=th)
        print(f'threshold: {th:.2f}')
        print(f'{args.metric}: {score:.3f}')

    '''
    # Get the model score
    if args.split == 'radio-test-mini':
        rad_score, model_score = radiologist_evaluate(model_probs, args.datadir, args.split, verbose=False)
        
        print(f'Rad score: {rad_score:.3f}')
        print(f'Model score: {model_score:.3f}')
    else:
        score = evaluate(model_probs, args.datadir, args.split, args.metric, verbose=False)

        print(f'{args.metric}: {score:.3f}')
    '''