"""
    Inputs models to compute worst predictions for each class and run visualization.
"""
import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torchvision.transforms as transforms

from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch.utils.data.dataloader import default_collate

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import get_model_and_loader, compute_probs_from_objects, transform_inputs

from joblib import Memory
memory = Memory(cachedir='./cache', verbose=0)


def get_predict_fn(loader):
    
    def _predict_fn(images):

        transformed_images = default_collate([loader.dataset.transform(Image.fromarray(image))\
                                              for image in images])

        probs_torch = model(transform_inputs(transformed_images, use_gpu=True))
        probs_npy = torch.sigmoid(probs_torch).cpu().data.numpy()

        # return np.stack([1-probs, probs], axis=1)
        return probs_npy

    return _predict_fn


@memory.cache
def get_worst_predictions(model, loader, N):
    """
    Computes the indices of the N worst predictions for each class in loader.
    Returns a list, where each element corresponds to the worst predictions for a class.
    Note: loader should be valid (not train due to dataset shuffling).
          Also does not (yet) support multilabels.
    """
    probs = compute_probs_from_objects(model, loader)

    labels = loader.dataset.labels

    classes = np.unique(labels)

    absolute_differences = np.absolute(probs - labels)

    worst_predictions = []
    
    for c in classes:

        absolute_differences_per_class = absolute_differences[labels == c]

        worst_predictions_per_class = np.argpartition(absolute_differences_per_class, -N)[-N:]

        worst_predictions.append(worst_predictions_per_class)

    return worst_predictions


if __name__ == "__main__":
    # Define command-line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', default='model')
    parser.add_argument('-d', '--datadir', default='data')
    parser.add_argument('-N', '--N', type=int, default=5)
    parser.add_argument('-s', '--split', default='valid')
    args = parser.parse_args()

    output_dir = Path('visualizations')
    if not output_dir.exists():
        output_dir.mkdir()

    # Get model and loader
    model, loader = get_model_and_loader(args.model_path, args.split)

    # Get worst predictions for each class
    worst_predictions = get_worst_predictions(model, loader, args.N)

    img_paths = loader.dataset.img_paths

    predict_fn = get_predict_fn(loader)

    explainer = lime_image.LimeImageExplainer()

    for c in range(len(worst_predictions)):

        worst_predictions_per_class = worst_predictions[c]

        worst_img_paths = [img_paths[i] for i in worst_predictions_per_class]

        for worst_img_path in worst_img_paths:

            worst_img = np.array(Image.open(worst_img_path).resize((224, 224), Image.ANTIALIAS).convert('RGB'))

            explanation = explainer.explain_instance(worst_img, predict_fn, hide_color=0, num_samples=1000)

            temp, mask = explanation.get_image_and_mask(c, positive_only=True, num_features=5, hide_rest=True)
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

            plt.savefig(f'{output_dir}/{c}_{str(worst_img_path).split("/")[-1]}')
            plt.close()
            assert False



        