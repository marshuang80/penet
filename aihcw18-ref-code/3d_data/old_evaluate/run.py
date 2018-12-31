import argparse

from get_best_model import get_best_models
from predict import predict
from evaluate import evaluate, radiologist_evaluate

if __name__ == "__main__":
	# Define command-line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',
                        help="path to get best from")
    parser.add_argument('-d', '--datadir', default='')
    parser.add_argument('-n', '--n', type=int, default=20)
    parser.add_argument('-s', '--split', default='valid')
    parser.add_argument('-m', '--metric', default='f1')
    args = parser.parse_args()

    # Get best models based on validation loss
    print("getting models")
    best_models = get_best_models(args.folder, args.n, verbose=False)
    model_paths = [path for loss, path in best_models]
    print("predicting")
    # Get model probabilities on dataset
    model_probs = predict(model_paths, args.split, save=False)
    print("getting scores")
    # Get the model score
    if args.split == 'radio-test-mini':
        model_score = evaluate(model_probs, args.datadir, args.split,args.metric, verbose=False)

        print(f'Model score: {model_score:.3f}')
    else:
        score = evaluate(model_probs, args.datadir, args.split, args.metric, verbose=False)

        print(f'{args.metric}: {score:.3f}')
