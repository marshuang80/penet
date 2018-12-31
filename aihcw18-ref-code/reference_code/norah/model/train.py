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
optimizers = {'adam': optim.Adam, 'rmsprop': optim.RMSprop}

import numpy as np
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from data.loader import load_data
from models import model_dict
from utils import transform_data, getPredMetrics, cluster_slices, store_preprocessed_images

def get_parser():
    parser = argparse.ArgumentParser()
    # Experiment Parameters
    parser.add_argument('--datadir', type=str, default='/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/best_localized_data/strict/HU_info_lt_-800_ut_-600_p_3.0%_ns_60')
    parser.add_argument('--rundir', type=str, default='/deep/group/aihc-bootcamp-winter2018/akos/Chest/experimental/test_run')

    parser.add_argument('--min_num_slices', type=int, default=15)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--bidirectional', action="store_true", default=False)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--learning_rate',type=float, default=0.0001)
    parser.add_argument('--lstm_dropout', type=float, default=0.2)

    # Model Parameters
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument('--model', type=str, default='SeqAvgAlexNet')
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--weighted_loss', action="store_true")
    parser.add_argument('--pooling', type=str, default='avgmax')

    # Training Parameters
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=50, type=int)
    parser.add_argument('--verbose', action="store_true", default=True)
    parser.add_argument('--plot', action="store_true", default=True)

    # Data Augmentation/Loading Parameters
    parser.add_argument('--scale', type=int, default=384)
    parser.add_argument('--crop', type=int, default=224)
    parser.add_argument('--horizontal_flip', action="store_true", default = False)
    parser.add_argument("--rotate", default=30, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--max_len", default=40, type=int)
    parser.add_argument('--stack_rgb', default='copy', type=str)
    parser.add_argument('--use_3D', action='store_true')
    parser.add_argument('--comment', default="", type=str)
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--undersample", action="store_true")
    # 'copy': Copy grayscale into each channel, 'seq': use sequential slices as the channels

    return parser

def run_model(model, loader, criterion, writer=None,epoch=None,
              train=False, optimizer=None, verbose=False):

    model.train(train)
    num_batches = len(loader)

    all_losses = []
    all_labels = []
    all_scores = []

    for batch in loader: #TODO tqdm
        # TO AVOID MEMORY LEAKS MAKE SURE THAT NO PYTORCH OBJECTS LEAVE THIS LOOP
        # CONVERT EVERYTHING TO NUMPY OR BUILT-IN TYPES

        if train:
            optimizer.zero_grad()
        inputs, labels = transform_data(batch, True, train)
        predictions = model(inputs.float())
        # BCEWithLogitsLoss averages over the batch by default.
        batch_loss = criterion(predictions, labels)

        if train:
            batch_loss.backward()
            optimizer.step()

        all_scores.append(predictions.data.cpu().numpy().flatten())
        all_labels.append(labels.data.cpu().numpy().flatten())
        all_losses.append(batch_loss.data.cpu().numpy().flatten())

    all_scores = np.concatenate(all_scores)
    all_predictions = (all_scores > 0).astype(int)
    all_labels = np.concatenate(all_labels).astype(int)
    all_losses = np.concatenate(all_losses)

    total = len(all_predictions)
    tot_pred_pos = np.sum(all_predictions == 1)
    tot_pred_neg = np.sum(all_predictions == 0)

    neg_fraction = tot_pred_neg / total
    pos_fraction = tot_pred_pos / total
    avg_loss = np.sum(all_losses) / num_batches

    roc_auc, accuracy, f1, kappa, precision, recall = getPredMetrics(all_labels, all_predictions, all_scores)

    if writer and epoch:
        phase_str = 'train' if train else 'val'
        writer.add_pr_curve(phase_str + '_pr_curve', all_labels, all_scores, global_step=epoch)
        writer.add_scalar('accuracy/' + phase_str, accuracy, epoch)
        writer.add_scalar('f1/' + phase_str, f1, epoch)
        writer.add_scalar('kappa/' + phase_str, kappa, epoch)
        writer.add_scalar('precision/' + phase_str, precision, epoch)
        writer.add_scalar('recall/' + phase_str, recall, epoch)
        writer.add_scalar('pos_fraction/'+ phase_str, pos_fraction, epoch)
        writer.add_scalar('auc_roc/' + phase_str, roc_auc, epoch)
    return roc_auc, avg_loss, accuracy, precision, recall, f1, kappa, neg_fraction, pos_fraction

def train(args, writer=None):
    print("Number of epochs:", args.epochs)
    print("Maximum number of slices:", args.max_len)
    print("Batch size:", args.batch_size)

    train_loader, valid_loader, test_loader, rad_loader = load_data(args)

    # Initialize desired model
    print (args.model, model_dict)
    #if args.model not in model_dict:
	#raise ValueError(f"{args.model} model not supported")

    assert train_loader.dataset.num_classes == \
           valid_loader.dataset.num_classes == \
           test_loader.dataset.num_classes, \
           "Different number of classes in data splits"

    model = model_dict[args.model](args, train_loader.dataset.num_classes)
    model = model.cuda()

    # Restore previous model
    if args.load_model:
        print("Restoring Model:", args.load_model)
        model.load_state_dict(torch.load(args.load_model))

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
    #if args.optimizer not in optimizers:
     #   raise ValueError(f"{args.optimizer} optmiizer not supported")
    optimizer = optimizers[args.optimizer](model.parameters(),
                                           args.learning_rate,
                                           weight_decay = args.decay)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    # Training loop
    train_losses = []
    valid_losses = []
    epochs = []

    best_val_loss = float('Inf')
    patience = 0
    val_precisions= []
    val_recalls = []
    val_f1s = []
    val_kappas = []
    start_time = datetime.now()
    for epoch in range(args.epochs):
        current_time = datetime.now()
        change = current_time - start_time
        print('-'*60)
        print("Starting epoch {}. Model has been running for {}".format(epoch, str(change)))
        train_roc_auc, train_loss, train_accuracy, train_precision, train_recall, train_f1, train_kappa, train_neg_fraction, train_pos_fraction = run_model(model, train_loader, train_criterion,
                               train=True, optimizer=optimizer,
                               verbose=args.verbose, writer=writer, epoch=epoch)

        val_roc_auc, val_loss, val_accuracy, val_precision, val_recall,val_f1, val_kappa, val_neg_fraction, val_pos_fraction  = run_model(model, valid_loader, valid_criterion,writer=writer, epoch=epoch)

        width = 12
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

        val_f1s.append(val_f1)
        val_kappas.append(val_kappa)

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        epochs.append(epoch)

        if writer:
            #TODO: guaranteed has at least one param group?
            lr = optimizer.param_groups[0]['lr']
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        file_name = f'val{val_loss}_train{train_loss}_epoch{epoch+1}'
        save_path = Path(args.rundir) / file_name

        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    args = get_parser().parse_args()
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
