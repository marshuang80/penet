import argparse

from get_best_model import get_best_models
from predict import predict
from evaluate import evaluate, radiologist_evaluate

if __name__ == "__main__":
	# Define command-line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',
                        help="path to get best from")
    #parser.add_argument('-d', '--datadir', default='/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/localized_New4ChestData/300_crop/lung_percent_threshold_12/HU_info_lt_-874_ut_-524_offset_1024')
    parser.add_argument('-d', '--datadir', default='/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/HU_New4ChestData/contrast_fluid/')
    parser.add_argument('-n', '--n', type=int, default=20)
    parser.add_argument('-s', '--split', default='radio-test-mini')
    parser.add_argument('-m', '--metric', default='roc_auc')
    args = parser.parse_args()

    # Get best models based on validation loss
    print("getting models")
    best_models = get_best_models(args.folder, args.n, verbose=False)
    model_paths = [path for loss, path in best_models]
    print("predicting")
    # Get model probabilities on dataset
    model_probs = predict(model_paths, args.split, save=False, save_dir=args.folder,n=args.n)
    print(model_probs.shape)
    print("getting scores")
    # Get the model score
    if args.split == 'radio-test-mini':
        model_score = evaluate(model_probs, args.datadir, args.split,args.metric, verbose=False, save_dir=args.folder,n=args.n)

        print(f'Model score: {model_score:.3f}')
    else:
        score = evaluate(model_probs, args.datadir, args.split, args.metric, verbose=False)

        print(f'{args.metric}: {score:.3f}')
