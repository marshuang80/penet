import json
from pathlib import Path

import torch
from torch.autograd import Variable

import numpy as np

from data.loader import load_data
from model.models import *

from sklearn.metrics import precision_recall_curve, accuracy_score, auc, f1_score, cohen_kappa_score, precision_score, recall_score

from joblib import Memory
memory = Memory(cachedir='./cache', verbose=0)


# convert dictionary to object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
        
def getPredMetrics(multilabel, labels, preds):
    '''
    Gets accuracy, f1, cohen kappa, precision, and recall for passed in labels and predictions (expects numpy arrays)
    '''
    average = 'weighted'
    accuracy = accuracy_score(labels, preds)
    if multilabel:
        f1 = f1_score(labels, preds, average = average)
        precision = precision_score(labels, preds, average = average)
        recall = recall_score(labels, preds, average = average)
    else:
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)

    kappa = cohen_kappa_score(labels, preds)
    return accuracy, f1, kappa, precision, recall
    

def transform_inputs(inputs, use_gpu):
    inputs = inputs.cuda()
    inputs = Variable(inputs, requires_grad=False, volatile=True)
    return inputs


def transform_data(data, use_gpu, train=False):
    img_dict, labels = data
    for view, img in img_dict.items():
        if img is not None:
            img = img.type(torch.FloatTensor)
            if use_gpu:
                img = img.cuda()
            img_dict[view] = Variable(img, requires_grad=False, volatile=not train)
    

    labels = labels.type(torch.FloatTensor)
    if use_gpu is True:
        labels = labels.cuda()
    labels = Variable(labels, volatile=not train)
    return img_dict, labels
    
def get_loader(datadir, split):
    
    # Define the args to groundtruth loader.
    args_dict = {
        'datadir': datadir,
        'batch_size': 1, # change this depending on memory usage
        'workers': 8, # 8 is good default for workers
        'toy': False, # never subsample when evaluating
        'scale': 224, # doesn't matter, only using labels
        'horizontal_flip': False, # doesn't matter, only using labels
        'rotate': False, # doesn't matter, only using labels
        'weighted_loss': False, # doesn't matter, only using labels
        'verbose': False,
        'reverse': False,
        'fix_num_slices': False,
        'shift': False,
        'rgb': True,
        'no_fours': False,
        'normalize': 'knee'
    }
    args = Struct(**args_dict)

    train_loader, val_loader, test_loader, rad_loader = load_data(args)
    loaders = {'train': train_loader, 'valid': val_loader, 'test': test_loader, 'radio-test-mini': rad_loader}
    loader = loaders[split]

    return loader

def get_loader_for_model(model_path, dataset):
    # loads model with args from model args.json
    with open(Path(model_path).parent / 'args.json') as args_f:
        model_args_dict = json.load(args_f)
    model_args = Struct(**model_args_dict)
    print("---Retrieved model args")

    if not hasattr(model_args, 'reverse'):
        model_args.reverse = False
    if not hasattr(model_args, 'shift'):
        model_args.shift = False
    if not hasattr(model_args, 'no_fours'):
        model_args.no_fours = False
    # Get loader from args
    print("---Loading data")
    train_loader, val_loader, test_loader, rad_loader = load_data(model_args)
    loaders = {'train' : train_loader, 'valid': val_loader, 'test': test_loader, 'radio-test-mini': rad_loader}
    loader = loaders[dataset]
    print("---Data loading complete")
    return loader


@memory.cache
def get_model_and_loader(model_path, dataset):
    # Load model args
    with open(Path(model_path).parent / 'args.json') as args_f:
        model_args_dict = json.load(args_f)
    model_args = Struct(**model_args_dict)
    print("---Retrieved model args")

    if not hasattr(model_args, 'reverse'):
        model_args.reverse = False
    if not hasattr(model_args, 'shift'):
        model_args.shift = False
    if not hasattr(model_args, 'no_fours'):
        model_args.no_fours = False
    # Get loader from args
    print("---Loading data")
    _, val_loader, test_loader, rad_loader = load_data(model_args)
    loaders = {'valid': val_loader, 'test': test_loader, 'radio-test-mini': rad_loader}
    loader = loaders[dataset]
    print("---Data loading complete")
    
    # Load model
    if model_args.model == 'lrcn':
        if not hasattr(model_args, 'dropout'):
            model_args.dropout = 0
        model = model_dict[model_args.model](model_args, loader.dataset.num_classes, model_args.hidden_dim, model_args.dropout).cuda()
    else:
        model = model_dict[model_args.model](model_args, loader.dataset.num_classes).cuda()

    remove_keys = ('attn_combine.bias', 'attn.bias', 'attn_combine.weight', 'attn.weight')
    # state = model.state_dict()
    '''
    for k in remove_keys:
        state.pop(k, None)
    state.update(state)
    print(state)
    '''
    # model.load_state_dict(state)
    # model.load_state_dict(torch.load(model_path))

    #state = model.state_dict()
    #print(state)
    #state.update(partial)
    #model.load_state_dict(state)
    # model_dict2 = model.state_dict()
    '''
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict2}
    # 2. overwrite entries in the existing state dict
    model_dict2.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    '''
    print("---Loading state dict")
    state_dict = torch.load(model_path)
    for k in remove_keys:
        state_dict.pop(k, None)
    print("---Loading model")    
    model.load_state_dict(state_dict)
    model.eval()

    return model, loader


@memory.cache
def compute_probs_from_objects(model, loader):
    # NOTE: will have to change sigmoid to softmax for multiclass setting.
    
    probs = []
    for batch in loader:
        batch_inputs, _ = transform_data(batch, use_gpu=True)

        batch_logits = model(batch_inputs)
        batch_probs = torch.sigmoid(batch_logits)
        batch_probs_npy = batch_probs.cpu().data.numpy()

        probs.append(batch_probs_npy)

    probs_concat = np.concatenate(probs, axis=0)

    return probs_concat


@memory.cache
def compute_probs_from_paths(model_path, dataset):

    model, loader = get_model_and_loader(model_path, dataset)
    
    return compute_probs_from_objects(model, loader)

