import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import argparse, time, json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
from data.loader import load_data
from model.models import ResNet152, DenseNet121, Simple3D,SeqRNNResNet18, Resnet183D, ManyToOne, Attention, AttentionFCN, model_dict
from utils import transform_data, getPredMetrics, cluster_slices, store_preprocessed_images
import warnings
optimizers = {'adam': optim.Adam, 'rmsprop': optim.RMSprop}
from sklearn.cluster import KMeans
import os




def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--datadir', type=str, default='/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/best_localized_data/strict/HU_info_lt_-800_ut_-600_p_3.0%_ns_60')
	parser.add_argument('--rundir', type=str, default='/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/models/eval_test/')
	parser.add_argument('--state_dict', type=str, default='/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/models/cropped/removing_outliers/more_than_15_range/UpdatedNew4ChestDataMarch7/decay_0.0005/lr_0.0001/scale384_crop256/maxpool/nborus_seqmaxresnet18/no_pretrained/1520474291/val0.22467675805091858_train0.23427745699882507_epoch5')
	
	parser.add_argument('--batch_size', default=1, type=int)
	
	parser.add_argument('--model', type=str, default='SeqMaxResNet18')
	parser.add_argument('--verbose', action="store_true")
	
	
	parser.add_argument('--optimizer', default='adam', type=str)
	parser.add_argument('--scale', type=int, default=384)
	parser.add_argument('--crop', type=int, default=256)
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
	parser.add_argument('--weighted_loss', action='store_true', default=True)

	parser.add_argument('--pretrained', action="store_true")
	parser.add_argument('--hidden_dim', type=int, default=128)
	parser.add_argument("--remove_invalids", action='store_true')
	parser.add_argument('--min_num_slices', type=int, default=15)
	parser.add_argument('--bidirectional', action="store_true", default=False)
	parser.add_argument('--lstm_dropout', type=float, default=0.2)
	parser.add_argument('--decay', type=float, default=0.0005)
	parser.add_argument('--learning_rate',type=float, default=0.0001)

	return parser


def run_model(model, loader, criterion,
				train=False, optimizer=None, verbose=False):

	model.train(train)
	avg_loss = 0.0
	num_batches = 0
	complete_predictions = None
	# TODO: cleaner way to get all labels?
	complete_labels = None
	running_corrects = 0
	pos_correct = 0.0
	precision_denom = 0.0
	pred_neg = 0.0
	pred_pos = 0.0
	recall_denom = 0.0
	total = 0
	recall_denom = 0.0
	for batch in loader:
		if train:
			optimizer.zero_grad()
		inputs, labels = transform_data(batch, True, train)
		predictions = model(inputs.float())
		if complete_predictions is None:
			complete_predictions = predictions
			complete_labels = labels
		else:
			complete_predictions = torch.cat((complete_predictions, predictions))
			complete_labels = torch.cat((complete_labels, labels))

		#BCEWithLogitsLoss averages over the batch by default.
		batch_loss = criterion(predictions, labels)
		avg_loss += batch_loss
		if train:
			batch_loss.backward()
			optimizer.step()
		preds = (predictions.data > 0).long()
		preds = preds.view(-1, 1)
		running_corrects += torch.sum(preds == labels.long().data)
		total += 1
		if labels.long().data[0][0] == 1:
			recall_denom += 1
		if preds[0][0] == 1:
			precision_denom += 1
			pred_pos +=1 
		if preds[0][0] == 0:
			pred_neg += 1
		if labels.long().data[0][0] == 1 and preds[0][0] == 1:
			pos_correct += 1

		num_batches += 1


	neg_fraction = pred_neg / total
	pos_fraction = pred_pos / total
	avg_loss /= num_batches
	accuracy = running_corrects / num_batches
	roc_auc, accuracy, f1, kappa, precision, recall = getPredMetrics(complete_labels.data, complete_predictions.data > .5)
	return roc_auc, avg_loss.data[0], accuracy, precision, recall, f1, kappa, neg_fraction, pos_fraction

def test(args):
	train_loader, valid_loader, test_loader, rad_loader = load_data(args)
	dataloaders = [rad_loader, valid_loader, test_loader, train_loader]
	dataloader_names = ['rad', 'valid', 'test', 'train']
	all_valids= []
	for curr, name in enumerate(dataloader_names): 
		for img_filepath in dataloaders[curr].dataset.img_paths:
			img_file = os.path.basename(img_filepath) 
			all_valids.append(img_file + '\n')

	with open('../all_valids.csv', 'w') as f:
		f.writelines(all_valids)

	'''
	if args.model not in model_dict:
		raise ValueError(f"{args.model} model not supported")

	assert train_loader.dataset.num_classes == \
			valid_loader.dataset.num_classes == \
			test_loader.dataset.num_classes, \
			"Different number of classes in data splits"

	model = model_dict[args.model](args, test_loader.dataset.num_classes)
	model = model.cuda()
	# Initialize loss function
	print("Load model from: ", args.state_dict)
	model.load_state_dict(torch.load(args.state_dict))

	# Initialize loss function
	if args.weighted_loss:
		# NOTE: think about weighted loss here - during validation,
		# should we use weights computed from training or validation?
		test_criterion = test_loader.dataset.weighted_loss
	else:
		test_criterion = nn.BCEWithLogitsLoss()

	 # Initialize optimizer and learning rate annealer
	if args.optimizer not in optimizers:
		raise ValueError(f"{args.optimizer} optmiizer not supported")
	optimizer = optimizers[args.optimizer](model.parameters(),
											args.learning_rate,
											weight_decay = args.decay)
	test_roc_auc, test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa, test_neg_fraction, test_pos_fraction = run_model(model, test_loader, test_criterion,
								train=False, optimizer=optimizer,
								verbose=args.verbose)
	print(f"Test roc_auc score: {test_roc_auc:0.4f}")
	print(f"Test pred = 0: {test_neg_fraction:0.4f}  Test pred = 1: {test_pos_fraction:0.4f})")
	print(f"Average test loss {test_loss:0.4f} Test Accuracy: {test_accuracy:0.4f}")
	print(f"Test precision: {test_precision:0.4f} Test recall: {test_recall:0.4f} Test f1: {test_f1:0.4f} Test kappa: {test_kappa: 0.4f}")
	'''


if __name__ == "__main__":
	args = get_parser().parse_args()
	with open(Path(args.rundir) / "args.json", 'w') as out:
		json.dump(vars(args), out, indent=4)

	test(args)

