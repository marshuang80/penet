import sys, json, time, cv2, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt

import torch

from utils import get_model_and_loader, transform_data
from evaluate.get_best_model import get_best_models


features_blobs = [None]


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modeldir', type=str, required=True)
    parser.add_argument('--datadir', type=str, default="/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/30AprData/localized")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--outputdir', type=str, default="/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/cams")

    return parser


def load_models(model_paths, split):
    
    def get_model_params(model_path):
        params = json.load(open(
            Path(model_path).parent / 'args.json', 'r'))
        return params

    def hook_feature(module, input, output):
        features_blobs[0] = output.data.cpu().numpy()


    models, loaders, _ = zip(*[get_model_and_loader(model_path, split) for model_path in model_paths])

    for model in models:
        # Will probably have to change this for models other than SeqAvgAlexNetGAP.
        model.cuda()
        model.model._modules.get('features').register_forward_hook(hook_feature)

    return models, loaders


def returnCAM(feature_conv, weight_softmax, size_upsample, class_idx):
    seq_len, nc, h, w = feature_conv.shape
    output_cams = []
    for seq_id in range(seq_len):
        cam = weight_softmax[class_idx].dot(feature_conv[seq_id].reshape((nc, h*w)))
        cam_img = cam.reshape(h, w)
        output_cams.append(cv2.resize(cam_img, size_upsample))
    return np.stack(output_cams)


def get_prediction_and_cams(model, inputs):

    start = time.time()

    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy()

    logit = model(inputs)

    probability = torch.sigmoid(logit).data.cpu().numpy().squeeze()
    num_classes = 1
    CAMs = [returnCAM(features_blobs[0], weight_softmax, inputs.data.cpu().numpy().shape[-2:], i) for i in range(num_classes)]

    print("Inference time: ", time.time() - start)
    return probability, CAMs


def get_cams(loader, models, label_names, thresholds, outputdir, save_orig=True):

    for i, batch in enumerate(loader):
        inputs, labels = transform_data(batch[:2], True, False)

        labels_npy = labels.data.cpu().numpy().squeeze()
        positive_inds = labels_npy == 1 # keep vectorized for multilabel models
        positive_names = np.array(label_names)[positive_inds]
        # Only run CAMS on positive examples.
        if not positive_names:
            continue

        ensemble_cams = []
        ensemble_predictions = []
        for model in models:
            prediction, cams = get_prediction_and_cams(model, inputs)
            ensemble_cams.append(cams)
            ensemble_predictions.append(prediction)
        ensemble_cams = np.mean(ensemble_cams, axis=0)
        ensemble_predictions = np.mean(ensemble_predictions, axis=0, keepdims=True)
        
        img_seq_npy = inputs.data.cpu().numpy().squeeze()
        img_seq_npy = (img_seq_npy - np.min(img_seq_npy))
        img_seq_npy = img_seq_npy / np.max(img_seq_npy)
        print (img_seq_npy.shape)
        if len(img_seq_npy.shape) < 4:
            continue
        img_seq_npy = np.transpose(img_seq_npy, (0, 2, 3, 1))

        cmap = plt.get_cmap('magma')
        for name in positive_names:
            label_index = label_names.index(name)
            if ensemble_predictions[label_index] > thresholds[label_index]:
                class_cam = ensemble_cams[label_index]
                # Normalize across the sequence.
                class_cam = class_cam - np.min(class_cam)
                class_cam = (class_cam / np.max(class_cam) * 255).astype('int')

                class_cams = [cmap(c)[:, :, :3] for c in class_cam]
                for j, cam in enumerate(class_cams):
                    merge = 0.3 * img_seq_npy[j] + 0.7 * cam
                    merged_normalized = merge / merge.max()
                    plt.imsave(str(outputdir / f'{name.squeeze()}_cam_{i}_{j}.jpg'), merged_normalized)

                    if save_orig:
                        print (np.min(img_seq_npy[j]), np.max(img_seq_npy[j]))
                        plt.imsave(str(outputdir / f'orig_{i}_{j}.jpg'), img_seq_npy[j])


def main():

    print("MAKE SURE LIST TRANSFORMS IS CORRECTLY CLIPPED VALUES.")

    args = get_parser().parse_args()

    best_models = get_best_models(args.modeldir, 1, args.datadir, verbose=False)
    print("best models: ", best_models)
    model_paths = [path for loss, path in best_models]

    models, loaders = load_models(model_paths, args.split)

    label_names = ["PE"] # Change if using multilabel
    thresholds = np.array([[0.5]]) # Change if using multilabel

    loader = loaders[0] # All loaders should be identical.

    outputdir = Path(args.outputdir)
    if not outputdir.exists():
        outputdir.mkdir()

    get_cams(loader, models, label_names, thresholds, outputdir)


if __name__ == "__main__":
    main()
    
