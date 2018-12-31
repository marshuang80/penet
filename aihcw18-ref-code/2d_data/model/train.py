"""
train.py
    defines command line arguments, training procedure
    should import from model.py and loader.py
    models should save in the following structure:
    {modality}-{task}/
        models/
            {experiment_type}/
                {timestamp in s}/
                    args.json
                    log.txt
                    val{val_loss}_train{train_loss}_epoch{epoch_num}
                    ...
                ...
            ...
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse, time, json
import os
import pickle
from datetime import datetime
import warnings
from shutil import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
optimizers = {'adam': optim.Adam, 'rmsprop': optim.RMSprop, 'sgd':optim.SGD}
from cls import CyclicLR

import numpy as np
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from data.loader import load_data
from models import model_dict
from utils import transform_data, getPredMetrics, sigmoid



def get_parser():
    parser = argparse.ArgumentParser()
    # Experiment Parameters
    parser.add_argument('--datadir', type=str, default='')
    parser.add_argument('--rundir', type=str, default='')
    parser.add_argument('--invalid_files_lower_threshold', type=int, default=16)
    parser.add_argument('--seed', default=123, type=int)

    # Model Parameters
    parser.add_argument('--model', type=str, default='SeqAvgAlexNet')
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--weighted_loss', action="store_true")
    parser.add_argument('--pooling', type=str, default='avgmax')
    parser.add_argument('--load_model', type=str, default="")

    # Training Parameters
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--verbose', action="store_true", default=True)
    parser.add_argument('--plot', action="store_true", default=True)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--learning_rate',type=float, default=0.0001)
    parser.add_argument('--factor', default=0.3, type=float)

    # Data Augmentation/Loading Parameters
    parser.add_argument('--scale', type=int, default=512)
    parser.add_argument('--crop', type=int, default=336)
    parser.add_argument('--mode', type=str, default='copy')
    parser.add_argument('--horizontal_flip', action="store_true", default = False)
    parser.add_argument("--rotate", default=30, type=int)
    parser.add_argument("--num_channels", default=3, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument('--shift', default=0, type=int)
    parser.add_argument("--max_len", default=40, type=int)
    parser.add_argument('--comment', default="", type=str)
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--undersample", action="store_true")

    return parser

def run_model(model, loader, criterion, writer=None, epoch=None,
              train=False, optimizer=None, verbose=False):

    if train:
        model.train()
    else:
        model.eval()

    num_batches = len(loader)

    all_losses = []
    all_labels = []
    all_probs = []

    for batch in loader: #TODO tqdm
        # TO AVOID MEMORY LEAKS MAKE SURE THAT NO PYTORCH OBJECTS LEAVE THIS LOOP
        # CONVERT EVERYTHING TO NUMPY OR BUILT-IN TYPES

        if train:
            optimizer.zero_grad()
        inputs, labels = transform_data(batch, True, train)
        predictions = model.forward(inputs.float())
        # BCEWithLogitsLoss averages over the batch by default.
        batch_loss = criterion(predictions, labels)

        if train:
            batch_loss.backward()
            optimizer.step()

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

    if writer and epoch:
        phase_str = 'train' if train else 'val'
        writer.add_pr_curve(phase_str + '_pr_curve', all_labels, all_probs, global_step=epoch)
        writer.add_scalar('accuracy/' + phase_str, accuracy, epoch)
        writer.add_scalar('f1/' + phase_str, f1, epoch)
        writer.add_scalar('kappa/' + phase_str, kappa, epoch)
        writer.add_scalar('precision/' + phase_str, precision, epoch)
        writer.add_scalar('recall/' + phase_str, recall, epoch)
        writer.add_scalar('pos_fraction/'+ phase_str, pos_fraction, epoch)
        writer.add_scalar('auc_roc/' + phase_str, roc_auc, epoch)
    return roc_auc, avg_loss, accuracy, precision, recall, f1, kappa, neg_fraction, pos_fraction

def train(args, writer=None):

    # Use this for MultiGPU Data Parallelization
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    else:
        gpu_ids = None

    print("Number of epochs:", args.epochs)
    print("Maximum number of slices:", args.max_len)
    print("Batch size:", args.batch_size)

    if args.batch_size > 1:
        assert args.invalid_files_lower_threshold == args.max_len, "Max Len should equal Min Len (Invalid Files Lower Threshold)"

    train_loader, valid_loader, test_loader = load_data(args)

    if args.model not in model_dict:
       raise ValueError(f"{args.model} model not supported")

    assert train_loader.dataset.num_classes == \
           valid_loader.dataset.num_classes == \
           test_loader.dataset.num_classes, \
           "Different number of classes in data splits"

    model = model_dict[args.model](args, train_loader.dataset.num_classes)

    # Restore previous model
    # Need this for C3D: model/pretrained/c3d.pickle
    if args.load_model:
        print("Restoring Model:", args.load_model)
        state = model.state_dict()
        loaded_state = torch.load(args.load_model)
        if 'state_dict' in loaded_state: # There is nesting
            loaded_state = loaded_state['state_dict']

        # Loads only the relevant model states
        for k in state.keys():
            if k in loaded_state:
                state[k] = loaded_state[k]
            # This is if the model has all layers moved to a features state dict instead of the original one
            new_k = k.replace('features.', '')
            if new_k in loaded_state:
                state[k] = loaded_state[new_k]
        model.load_state_dict(state)

    model = model.cuda()
    
    if len(gpu_ids) > 1:
        model = DataParallel(model, device_ids=gpu_ids)

    # Initialize loss function
    if args.weighted_loss:
        # NOTE: think about weighted loss here - during validation,
        # should we use weights computed from training or validation?
        train_criterion = train_loader.dataset.weighted_loss
        valid_criterion = valid_loader.dataset.weighted_loss
    else:
        train_criterion = nn.BCEWithLogitsLoss()
        valid_criterion = nn.BCEWithLogitsLoss()

    # Initialize optimizer and learning rate annealer
    if args.optimizer not in optimizers:
        raise ValueError(f"{args.optimizer} optimizer not supported")
    if args.optimizer == 'sgd':
        optimizer = optimizers[args.optimizer](model.parameters(), args.learning_rate,
                                               weight_decay = args.decay, momentum=0.9)
    else:
        optimizer = optimizers[args.optimizer](model.parameters(), args.learning_rate,
                                               weight_decay = args.decay)
    #scheduler = CyclicLR(optimizer, base_lr=5e-7, max_lr=5e-4, mode='exp_range', step_size=8)#lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    #schedulerContinuous = lr_scheduler.StepLR(optimizer, step_size=3, gamma=args.factor)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.max_patience, factor=args.factor, threshold=1e-4)
    
    # Training loop
    train_losses = []
    valid_losses = []
    epochs = []

    best_val_loss = float('Inf')
    val_precisions= []
    val_recalls = []
    val_f1s = []
    val_kappas = []
    start_time = datetime.now()
    for epoch in range(args.epochs):
        change = datetime.now() - start_time
        print('-' * 60)
        print("Starting epoch {}. Model has been running for {}".format(epoch, str(change)))
        train_roc_auc, train_loss, train_accuracy, train_precision, train_recall, train_f1, train_kappa, train_neg_fraction, train_pos_fraction = run_model(model, train_loader, train_criterion,
                               train=True, optimizer=optimizer,
                               verbose=args.verbose, writer=writer, epoch=epoch)

        val_roc_auc, val_loss, val_accuracy, val_precision, val_recall,val_f1, val_kappa, val_neg_fraction, val_pos_fraction  = run_model(model, valid_loader, valid_criterion, writer=writer, epoch=epoch)
        lr = optimizer.param_groups[0]['lr']

        width = 12
        lr = optimizer.param_groups[0]['lr']

        print(f'{"":>{width}}   Train | Val  ')
        print(f'{"Average loss":>{width}}: {train_loss:0.4f} | {val_loss:0.4f}')
        print(f'{"ROC_AUC":>{width}}: {train_roc_auc:0.4f} | {val_roc_auc:0.4f}')
        print(f'{"Kappa":>{width}}: {train_kappa:0.4f} | {val_kappa:0.4f}')
        print()
        print(f'{"Accuracy":>{width}}: {train_accuracy:0.4f} | {val_accuracy:0.4f}')
        print(f'{"F1":>{width}}: {train_f1:0.4f} | {val_f1:0.4f}')
        print(f'{"Precision":>{width}}: {train_precision:0.4f} | {val_precision:0.4f}')
        print(f'{"Recall":>{width}}: {train_recall:0.4f} | {val_recall:0.4f}')
        print(f'{"Predicted 0":>{width}}: {train_neg_fraction:0.4f} | {val_neg_fraction:0.4f}')
        print(f'{"Predicted 1":>{width}}: {train_pos_fraction:0.4f} | {val_pos_fraction:0.4f}')
        print('Learning Rate %f' % lr)

        val_f1s.append(val_f1)
        val_kappas.append(val_kappa)

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        epochs.append(epoch)

        if writer:            
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('learning_rate/train', lr, epoch)

        if args.plot:

            plot_dir = Path(args.rundir) / "plots"
            plt.plot(epochs, train_losses, label="train")
            plt.plot(epochs, valid_losses, label="valid")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(str(plot_dir / "loss"))

            plt.clf()
            plt.plot(epochs, val_precisions, label="valid precision", color='r')
            plt.plot(epochs, val_recalls, label="valid recall", color='b')
            plt.xlabel('Epoch')
            plt.ylabel('Recall & Precision')
            plt.legend()
            plt.savefig(str(plot_dir / "val_precision_recall"))
            plt.close()

            plt.clf()
            plt.plot(epochs, val_f1s, label="valid f1", color='r')
            plt.plot(epochs, val_kappas, label="valid kappa", color='b')
            plt.xlabel('Epoch')
            plt.ylabel('Validation F1 & Kappa')
            plt.legend()
            plt.savefig(str(plot_dir / "val_f1_kappa"))
            plt.close()

        scheduler.step(val_loss)
        #schedulerContinuous.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        file_name = f'val{val_loss}_train{train_loss}_epoch{epoch+1}'
        save_path = Path(args.rundir) / file_name

        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    print(args)

    with open(Path(args.rundir) / "args.json", 'w') as out:
        json.dump(vars(args), out, indent=4)

    if args.plot:
        plot_dir = Path(args.rundir) / "plots"
        plot_dir.mkdir(exist_ok=True)

    if (args.oversample or args.undersample) and args.weighted_loss:
        warnings.warn("You are using weighted loss and oversampling.")

    tensorboard_dir = Path(args.rundir) / "tf_events"
    tensorboard_dir.mkdir(exist_ok=True)

    comment = args.comment if args.comment else ""
    writer = SummaryWriter(log_dir = tensorboard_dir, comment=comment)

    # Copy code files to rundir for easier visualization and tracking
    backup_code_dir = Path(args.rundir) / "code"
    backup_code_dir.mkdir(exist_ok=True)
    copy('model/train.py', backup_code_dir)
    copy('model/models.py', backup_code_dir)
    copy('data/loader.py', backup_code_dir)
    copy('data/list_transforms.py', backup_code_dir)
    train(args, writer)
