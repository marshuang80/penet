"""
dq_script.py
  if use_dq is True, submit jobs using dq.
  otherwise, print out commands to run from command-line.
"""
from subprocess import Popen, check_call
from collections import namedtuple
from pathlib import Path
import time
import argparse

# Experiment results will be saved in /aaa_mr_knee_experiments/{timestamp}_{task}_{view}_{cnn}_{pooling}_{keyword}
parser = argparse.ArgumentParser()
parser.add_argument('--use_dq', action="store_true")
parser.add_argument('--test', action="store_true")
# Experiment Parameters
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--view', type=str, required=True)
parser.add_argument('--cnn', default='resnet34', type=str)
parser.add_argument('--keyword', type=str, required=True)
parser.add_argument('--pooling', default='avgmax', type=str)
parser.add_argument('--model', default='cnn', type=str)
parser.add_argument('--add_relu', action="store_true")
parser.add_argument('--pretrained', action="store_false")
# Training Parameters
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--weighted_loss', action="store_true")
parser.add_argument('--epochs', default=35, type=int)
parser.add_argument('--oversample', action="store_true")
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--max_patience', default=5, type=int)
parser.add_argument('--factor', default=0.3, type=float)
# Data Loading Parameters
parser.add_argument('--toy', action="store_true")
parser.add_argument('--rgb', default=True, action="store_true")
parser.add_argument('--fix_num_slices', action="store_true")
parser.add_argument('--num_slices', default=52, type=int) # should be ignored if fix_num_slices = False
parser.add_argument('--fixing_method', default='inner', type=str) # should be ignored if fix_num_slices = False
parser.add_argument('--normalize', default='knee', type=str)
# Data Augmentation Parameters
parser.add_argument('--scale', default=256, type=int)
parser.add_argument('--horizontal_flip', default=True, action="store_true")
parser.add_argument('--rotate', default=15, type=int)
parser.add_argument('--shift', default=15, type=int)
parser.add_argument('--reverse', default=True, action="store_true")
args = parser.parse_args()

# NOTE: use your name as the first part of the experiment
# to distinguish your experiments from others!
modality = "mr"
subject = "knee_abnormality"
experiment = "alexnet_ftw"

if args.model in ['simple2d']:
  args.rgb = False

if args.model in ['alexnet', 'cnn']:
  assert hasattr(args, 'pooling'), "pooling can be avgmax, avgavg, maxmax, or maxavg"

if args.toy:
  experiment += '_toy'

if not args.oversample:
  args.weighted_loss = True

assert args.weighted_loss != args.oversample, 'oversampling and doing weighted loss at the same time'

if not args.test:
    
    HParams = namedtuple("HParams", ["modality", "subject", "view", "task", "cnn", "pooling", # Experiment Parameters
                                     "pretrained", "model", "weighted_loss", "oversample", "add_relu",  # Model Parameters
                                     "optimizer", "learning_rate", "weight_decay", # Training Parameters
                                     "dropout", "batch_size", "epochs", "max_patience", "factor", # Training Parameters
                                     "normalize", "seed", "toy",
                                     "rgb", "scale", "horizontal_flip", "rotate", "reverse", # Data Parameters
                                     "shift", "fix_num_slices", "num_slices", "fixing_method", # Data Parameters
                                     "dq_desc", "experiment"], verbose=False)

    settings = []

    for i, sd in enumerate([123, 124]):

        settings.append(HParams(modality=modality, subject=subject, view=args.view, task=args.task, cnn=args.cnn, pooling=args.pooling,
                                pretrained=args.pretrained, model=args.model, weighted_loss=args.weighted_loss, oversample=args.oversample, add_relu=args.add_relu,
                                optimizer=args.optimizer, learning_rate=args.lr, weight_decay=args.weight_decay, 
                                dropout=args.dropout, batch_size=args.batch_size, epochs=args.epochs, 
                                max_patience=args.max_patience, factor=args.factor,
                                normalize=args.normalize, seed=sd, toy=args.toy,
                                rgb=args.rgb, scale=args.scale, horizontal_flip=args.horizontal_flip, rotate=args.rotate, reverse=args.reverse,
                                shift=args.shift, fix_num_slices=args.fix_num_slices, num_slices=args.num_slices, fixing_method=args.fixing_method,
                                dq_desc=f'{args.model}_lr{args.lr}_{experiment}_train', experiment=experiment))

else:

    
    HParams = namedtuple("HParams", ["modality", "subject", # Experiment Parameters
                                     "n",
                                     "split",
                                     "metric",
                                     "dq_desc", "experiment"], verbose=False)     

    settings = [HParams(modality=modality, subject=subject,
                        n=1,
                        split="valid", # one of valid, test, radio-test-mini
                        metric="roc_auc", # one of roc_auc, pr_auc, f1. this param is ignored if radio-test-mini is used.
                        dq_desc=f'{experiment}_test', experiment=experiment)]

k = 0
for setting in settings:
    setting_dict = setting._asdict()

    parent_dir = Path("/deep/group/aihc-bootcamp-winter2018/medical-imaging/")
    task_modality_dir = parent_dir / f"{setting_dict['modality']}_{setting_dict['subject']}"
    modeldir = task_modality_dir / "models" / setting_dict['experiment']
    datadir = task_modality_dir / "data-final"

    t = int(time.time())
    
    setting_dict['dq_desc'] += "_" + str(t)

    if args.test:
        
        outputdir = Path("results") / str(t)
        
        if not outputdir.exists():
            outputdir.mkdir(parents=True)
        
        cmd = f"python evaluate/run.py {modeldir} --datadir {datadir}" +\
                                     f" --n {setting_dict['n']} " + \
                                     f"--split {setting_dict['split']} " + \
                                     f"--metric {setting_dict['metric']}"


        cmd += f" > {outputdir}/eval.txt"
    
    else:
        
        if args.model == 'cnn':
          rundir = modeldir / (f'{str(t)}_{args.task}_{args.view}_{args.cnn}_{args.keyword}')
        else:
          rundir = modeldir / (f'{str(t)}_{args.task}_{args.view}_{args.model}_{args.keyword}')
        cmd = f"python model/train.py --datadir {datadir} --rundir {rundir} --verbose"

        for hparam in setting_dict:
            if hparam in ["pretrained", "weighted_loss", "oversample", "horizontal_flip", "fix_num_slices", "rgb", "reverse", "toy", "add_relu"]:
                if setting_dict[hparam] == True:
                    cmd += f" --{hparam}"
            else:
                if hparam not in ['modality', 'subject', 'dq_desc', 'experiment']:
                    cmd += f" --{hparam} {setting_dict[hparam]}"
  
        cmd += f" > {rundir}/log.txt"
  
        if not rundir.exists():
            rundir.mkdir(parents=True)

    if args.use_dq:
        with open("tmp_dq_script.sh", "w") as fout:
            fout.write(cmd)
        # DO NOT CHANGE THE RANGE!
        check_call(f"DQ_DESC={setting_dict['dq_desc']} DQ_RANGE=16-5 dq-submit tmp_dq_script.sh",
                    shell=True)
    else:
        # NOTE: this command will background your process and redirect stdout/stderr to nohup.out
        # processes will live after exiting ssh session.
        cmd = f"export CUDA_VISIBLE_DEVICES={k % 4}; nohup {cmd} &"
        print(cmd)
        k += 1

        # For timestamps when not using dq.
        time.sleep(2)

