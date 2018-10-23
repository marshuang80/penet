"""
get_best_model.py
	input an experiment path /deep/group/{modality_task}/models/{experiment_type}/
	input number of models n
	output the n best models based on validation loss (found in the path, search over epochs)
"""
import glob, sys, argparse

def print_helper(seq, verbose, n_print):
    print(f"Best {n_print} model(s):")
    for i, model_info in enumerate(sorted(seq, reverse=True)):
        if verbose:
            print(model_info)
        else:
            print(model_info[1])
        if i == n_print:
            break
	
def get_best_models(path, n, verbose):
    # from https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    def find_between(s, first, last):
        try:
            start = s.index(first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""

    models = []
    for checkpoint_path in glob.glob(path + "/**/*epoch*"):
        val_loss = float(find_between(
            checkpoint_path, 'val', '_train'))
        models.append((val_loss, checkpoint_path))
    
    return sorted(models)[:n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', 
        help="Path to experiment")
    parser.add_argument('-n', '--n', type=int, default=20)
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()
    models = get_best_models(args.path, args.n, args.verbose)

    print_helper(models, args.verbose, args.n)
