"""
    Inputs models to compute worst predictions for each class and run visualization.
"""
import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import os, json
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from lime import lime_image
import cv2
import sklearn
from skimage.segmentation import mark_boundaries
from torch.utils.data.dataloader import default_collate
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import matplotlib.pylab as pl
from skimage.segmentation import slic
from utils import get_loader, get_model_and_loader, compute_probs_from_objects, transform_inputs
from data.loader import load_data
from joblib import Memory
memory = Memory(cachedir='./cache', verbose=0)


def get_predict_fn(loader):
    print('getting predict function...')    
    def _predict_fn(images):

        transformed_images = default_collate([loader.dataset.transform(Image.fromarray(image))\
                                              for image in images])

        probs_torch = model(transform_inputs(transformed_images, use_gpu=True))
        probs_npy = torch.sigmoid(probs_torch).cpu().data.numpy()
        preds_npy = (probs_npy > 0.5).astype(int)
        # return np.stack([1-probs, probs], axis=1)
        return preds_npy
    print('done.')
    return _predict_fn


@memory.cache
def get_worst_predictions(model, loader, N):
    """
    Computes the indices of the N worst predictions for each class in loader.
    Returns a list, where each element corresponds to the worst predictions for a class.
    Note: loader should be valid (not train due to dataset shuffling).
          Also does not (yet) support multilabels.
    """
    print('getting worst predictions...')
    probs = compute_probs_from_objects(model, loader)

    labels = loader.dataset.labels

    classes = np.unique(labels)

    absolute_differences = np.absolute(probs - labels)

    best_predictions = []
    worst_predictions = []
    
    for c in classes:

        absolute_differences_per_class = absolute_differences[labels == c]

        worst_predictions_per_class = np.argpartition(absolute_differences_per_class, -N)[-N:]
        best_predictions_per_class = np.argpartition(absolute_differences_per_class, -N)[:N]
        worst_predictions.append(worst_predictions_per_class)
    print('done.')
    return worst_predictions

def get_saliency_map(img, model, label, from_file = True):
    """
    image: directory of image to predict or image in numpy(batch_size x num_channels x num_slices x im_height x im_width)
    model to run:
    """
    if from_file:
        image_npy = np.load(img)
        image = np.expand_dims(image_npy, axis=0)
        image = np.stack((image,)*3, axis=1) # Converting from grayscale to RGB
    
    model = model.cuda()
    model.zero_grad()
    model.eval()

    im_var = torch.from_numpy(image).type(torch.FloatTensor)
    im_var = Variable(im_var.cuda(), requires_grad = True)
    # pred = model(im_var)
    pred = torch.sigmoid(model(im_var))
    print('pred :',pred)
    print('pred type: ', type(pred))

    """
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(pred, label)
    loss.backwards(retain_variables=True)
    """
    if label == 1:
        lab = torch.ones((1,1)).type(torch.FloatTensor).cuda()
    else:
        lab = (-torch.ones((1,1))).type(torch.FloatTensor).cuda()
    pred.backward(lab, retain_variables=True)
    gradient = im_var.grad.data.cpu().numpy()
    print(gradient)
    print('grad shape: ', gradient.shape)
    max_elem = np.amax(gradient)
    print('max: ', max_elem)
    min_elem = np.amin(gradient)
    print('min: ', min_elem)
    max_norm = np.max([max_elem, -min_elem])
    gradient /= max_norm
    gradient = 0.5 * gradient + 0.5 #Maps from 0 to 1
    print('gradient: ', gradient)
    print("Max after normalization: {}, Min after normalization: {}".format(np.amax(gradient), np.amin(gradient)))
    height, width = image_npy.shape[1:]
    
    gradient = gradient[:,0,:,:,:]
    gradient = np.squeeze(gradient, axis=0)
    print('grad shape: ', gradient.shape)
    heatmaps = []
    for i in range(gradient.shape[0]):
        cam = cv2.resize(gradient[i,:,:],(width, height))
        print('cam shape: ', cam.shape)
        cam = sklearn.preprocessing.minmax_scale(cam)
        cmap = plt.get_cmap("gnuplot2")
        cam = cmap(cam) * 255
        print('cam shape: ', cam.shape)
        heatmaps.append(cam)
    return heatmaps

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
    # best_models = get_best_models(args.folder, args.n, verbose=False)
    # model_paths = [path for loss, path in best_models]
    print('loading model...')
    # train_loader = get_loader(args.datadir, 'train')
    # train = train_loader.dataset
    model, loader = get_model_and_loader(args.model_path, args.split)
    # model.eval()
    # Get worst predictions for each class
    img_path = '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/data-normalized/images/71035_1.npy'
    png_path = '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/images-scaled/images/71035_1_14.png'
    heatmaps = get_saliency_map(img_path, model, 0)
    heatmap = heatmaps[15]
    png = Image.open(png_path)
    png.save('/deep/group/aihc-bootcamp-winter2018/ajypark/mr.png')
    np.save('/deep/group/aihc-bootcamp-winter2018/ajypark/heatmap', heatmap)
    # worst_predictions = get_worst_predictions(model, loader, args.N)
    '''
    img_paths = loader.dataset.img_paths
    for c in range(len(worst_predictions)):

        worst_predictions_per_class = worst_predictions[c]

        worst_img_paths = [img_paths[i] for i in worst_predictions_per_class]
        print('worst_img_paths :', worst_img_paths)
        for worst_img_path in worst_img_paths:
            print('getting saliency map...')
            get_saliency_map(worst_img_path, model)
            print('done')
    '''
    # predict_fn = get_predict_fn(loader)

    # explainer = shap.KernelExplainer(predict_fn, np.zeros((1,50)))
    # shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1)
    # explainer = lime_image.LimeImageExplainer()
    print('done.')
    '''
    topil = transforms.ToPILImage()
    for c in range(len(worst_predictions)):

        worst_predictions_per_class = worst_predictions[c]

        worst_img_paths = [img_paths[i] for i in worst_predictions_per_class]
        print('worst_img_paths :', worst_img_paths)
        for worst_img_path in worst_img_paths:
            img = np.load(worst_img_path)
            img = np.expand_dims(img, axis=0)
            img_name = worst_img_path[96:-4] # Disclaimer: this is bad
            for i in range(img.shape[1]):    
                img_file = Image.open(Path(args.datadir).parent / 'images-scaled' / 'images' / img_name + '_' + str(i) + ".png")
            for i in range(img.shape[1]):
                img_save = np.squeeze(img[:,i,:,:], axis=0)
                topil(img_save).save(Path(args.model_path).parent / ("worst_" + str(worst_img_path)[96:-4] + str(i) + ".png"))
    '''
    '''
        for worst_img_path in worst_img_paths:
            print('loading images...')
            img = np.load(worst_img_path)
            img = np.expand_dims(img, axis=0)
            img = np.stack((img,)*3, axis=4) # 1, slice, 256, 256, 3
            img = np.squeeze(img, axis=0) # slice, 256, 256, 3
            # print(img.shape)
            print('done.')
            print('getting pred fn for images...')
            preds = predict_fn(img.copy())
            top_preds = np.argsort(-preds)
            print('done.')
            print('plotting...')

            colors = []
            for l in np.linspace(1,0,100):
                colors.append((245/255,39/255,87/255,l))
            for l in np.linspace(0,1,100):
                colors.append((24/255,196/255,93/255,l))
            cm = LinearSegmentedColormap.from_list("shap", colors)

            def fill_segmentation(values, segmentation):
                out = np.zeros(segmentation.shape)
                for i in range(len(values)):
                    out[segmentation == i] = values[i]
                return out

            # plot our explanations
            fig, axes = pl.subplots(nrows=1, ncols=4, figsize=(12,4))
            inds = top_preds[0]
            axes[0].imshow(img)
            axes[0].axis('off')
            max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
            for i in range(3):
                m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
                axes[i+1].set_title(feature_names[str(inds[i])][1])
                axes[i+1].imshow(img.convert('LA'), alpha=0.15)
                im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
                axes[i+1].axis('off')
            cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
            cb.outline.set_visible(False)
            pl.show()
            
            for i in range(img.shape[0]):
                # print('img[i,:,:,:] :', img[i,:,:,:].shape)
                # img = np.squeeze(img[i,:,:,:], axis=0)
                worst_img = img[i,:,:,:] # 256, 256, 3
                explanation = explainer.explain_instance(worst_img, predict_fn, hide_color=0, num_samples=1000)

                temp, mask = explanation.get_image_and_mask(c, positive_only=True, num_features=5, hide_rest=True)
                plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

                plt.savefig(f'{output_dir}/{c}_{str(worst_img_path).split("/")[-1]}')
                plt.close()
            assert False
            
    '''
