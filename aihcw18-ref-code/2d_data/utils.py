import json
from pathlib import Path
import os

import torch
from torch.autograd import Variable

import numpy as np

from data.loader import load_data
from model.models import *
import matplotlib
import matplotlib.pyplot as plt
from joblib import Memory
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, auc, f1_score, cohen_kappa_score, precision_score, recall_score

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def getPredMetrics(labels, preds, scores):
    '''
    Gets accuracy, f1, cohen kappa, precision, and recall for passed in labels and predictions (expects numpy arrays)
    '''
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    roc_auc = roc_auc_score(labels, scores)
    return roc_auc, accuracy, f1, kappa, precision, recall

# convert dictionary to object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def transform_inputs(inputs, use_gpu):
    inputs = inputs.cuda()
    inputs = Variable(inputs, requires_grad=False, volatile=True)
    return inputs


def transform_data(data, use_gpu, train=False, grad=False):
    inputs, labels = data
    labels = labels.type(torch.FloatTensor)
    if use_gpu is True:
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs = Variable(inputs, requires_grad=grad, volatile=not train)
    labels = Variable(labels, requires_grad=grad, volatile=not train)
    return inputs, labels

def get_loader(datadir, split):

    # Define the args to groundtruth loader.
    args_dict = {
        'datadir': datadir,
        'batch_size': 1, # change this depending on memory usage
        'workers': 8, # 8 is good default for workers
        'toy': False, # never subsample when evaluating
        'scale': 512, # doesn't matter, only using labels
        'horizontal_flip': False, # doesn't matter, only using labels
        'rotate': False, # doesn't matter, only using labels
        'weighted_loss': False, # doesn't matter, only using labels
        'verbose': False,
        'max_len': 40,
        'crop': 336,
        'invalid_files_lower_threshold': 16, 
        'mode': 'copy',
    }
    args = Struct(**args_dict)

    train_loader, val_loader, test_loader = load_data(args)
    loaders = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
    loader = loaders[split]

    return loader


def create_pngs(ct_scan_filepath, store_parent_dir, start=None,end=None):

    '''
    Input:
        ct_scan_filepath: (absolute path) of file containing nx512x512 array, where n represents # of image slices for one chest ct_scan
        store_parent_dir: parent directory for where pngs should be stored
        start: index of first image slice
        end: index of last image slice

    Goal: Create pngs of each image slice in the ct_scan numpy array and store the results in the relevant dir
    '''
    ct_scan_filename = os.path.basename(ct_scan_filepath)
    store_dir = store_parent_dir + "/" + ct_scan_filename + "/"
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    files_stored = os.listdir(store_dir)
    if len(files_stored) > 0:
        print("store_dir {} already populated".format(ct_scan_filename))
        return None

    print("PNGS will be stored in: {}".format(store_dir))
    ct_scan = np.load(ct_scan_filepath)
    if start == None:
        start = 0
    if end == None:
        end = len(ct_scan)

    print("Start: {}, End: {}".format(start, end))
    for index,matrix in enumerate(ct_scan):
        plt.imsave(store_dir + str(index) +".png", matrix, cmap='gray')


def get_model_and_loader(model_path, dataset):

    # Load model args
    with open(Path(model_path).parent / 'args.json') as args_f:
        model_args_dict = json.load(args_f)
    model_args = Struct(**model_args_dict)
    # Get loader from args
    train_loader, val_loader, test_loader = load_data(model_args)
    loaders = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
    loader = loaders[dataset]
    # Load model
    model = model_dict[model_args.model](model_args, loader.dataset.num_classes).cuda()
 

    model.load_state_dict(torch.load(model_path))
    '''    
    state = model.state_dict()
    loaded_state = torch.load(model_path)
    for k in state.keys():
        if k not in loaded_state and 'features' in k:
            new_k = k.replace('features.', '')
            state[k] = loaded_state[new_k]
        elif k not in loaded_state and ('features.' + k) in loaded_state:
            new_k = 'features.' + k
            state[k] = loaded_state[new_k]
        else:
            state[k] = loaded_state[k]
    model.load_state_dict(state) '''
    model.eval()
    csv_file_path = model_args.datadir +'/'+ dataset +".csv"

    return model, loader, csv_file_path



def compute_probs_from_objects(model, loader):
    all_losses = []
    all_labels = []
    all_probs = []

    num_batches = len(loader)

    for batch in loader: #TODO tqdm
        inputs, labels = transform_data(batch, True, train=False)
        predictions = model.forward(inputs.float())
        # BCEWithLogitsLoss averages over the batch by default.
        batch_loss = nn.BCEWithLogitsLoss()(predictions, labels)

        all_probs.append(predictions.data.cpu().numpy().flatten())
        all_labels.append(labels.data.cpu().numpy().flatten())
        all_losses.append(batch_loss.data.cpu().numpy().flatten())

    all_probs = sigmoid(np.concatenate(all_probs))

    all_predictions = (all_probs > 0.5).astype(int)
    all_labels = np.concatenate(all_labels).astype(int)
    all_losses = np.concatenate(all_losses)

    total = len(all_predictions)
    tot_pred_pos = np.sum(all_predictions == 1)
    tot_pred_neg = np.sum(all_predictions == 0)

    neg_fraction = tot_pred_neg / total
    pos_fraction = tot_pred_pos / total
    avg_loss = np.sum(all_losses) / num_batches

    roc_auc, accuracy, f1, kappa, precision, recall = getPredMetrics(all_labels, all_predictions, all_probs)
    print ("ROC", roc_auc)
    print ("Acc", accuracy)
    print ("loss", avg_loss)
    print ("precision", precision)
    print ("recall", recall)

    return all_probs, all_labels



def compute_probs_from_paths(model_path, dataset):

    model, loader, _ = get_model_and_loader(model_path, dataset)
    return compute_probs_from_objects(model, loader)

