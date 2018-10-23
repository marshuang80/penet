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


parser = argparse.ArgumentParser()
parser.add_argument('--use_dq', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--comment', default="", type=str)
args = parser.parse_args()

# NOTE: use your name as the first part of the experiment
# to distinguish your experiments from others! 
modality = "ct"
task = "chest_pe"
#experiment = "cropped/removing_outliers/more_than_15_range/contrast_fluid_New4ChestData/decay_0.0005/lr_0.0001/scale384_crop256/lstmdropout_0.2/hiddendim_128/nborus_seqrnnresnet34/no_pretrained/"
#experiment = 'cropped/removing_outliers/more_than_15_range/contrast_fluid_New4ChestData/decay_0.0005/lr_0.0001/scale384_crop256/avgpool/nborus_seqavgresnet18/no_pretrained/'
#experiment = 'cropped/removing_outliers/more_than_15_range/localized_New4ChestData/decay_0.0005/lr_0.0001/scale384_crop224/avgpool/nborus_seqavgalexnet/pretrained/'
#experiment = 'kmeans_preprocess'
#experiment = 'eval_test/nborus_seqmaxresnet18/'
#experiment = '12_percent_threshold_300_crop/avgresnet18GAP_batchsize_10/'
experiment = 'contrast_fluid_localization/avgresnet18GAP_batchsize_5/'
if not args.test:
    
    HParams = namedtuple("HParams", ["modality", "task", # Experiment Parameters
                                     "model", "pretrained", "weighted_loss", # Model Parameters
                                     "optimizer", "learning_rate", "batch_size", "epochs", # Training Parameters
                                     "scale", "horizontal_flip", "rotate", # Data Parameters
                                     "dq_desc", "experiment", 'max_len', 'decay'], verbose=False)

    settings = []

    for i, lr in enumerate([0.01]):
        settings.append(HParams(modality=modality, task=task,
                                model="SeqAvgResNet18GAP", pretrained=False, weighted_loss=True,
                                optimizer="sgd", learning_rate=lr, batch_size=1, epochs=50,
                                scale=384,horizontal_flip=False, rotate=0, max_len=40, decay=0.0001, 
                                dq_desc=f'seqavgresnet18_lr{lr}_{experiment}_train', experiment=experiment))

else:

    
    HParams = namedtuple("HParams", ["modality", "task", # Experiment Parameters
                                     "n",
                                     "split",
                                     "metric",
                                     "dq_desc", "experiment"], verbose=False)     

    settings = [HParams(modality=modality, task=task,
                        n=1,
                        split="valid", # one of valid, test, radio-test-mini
                        metric="roc_auc", # one of roc_auc, pr_auc, f1. this param is ignored if radio-test-mini is used.
                        dq_desc=f'{experiment}_test', experiment=experiment)]

k = 0
for setting in settings:
    setting_dict = setting._asdict()

    parent_dir = Path("/deep/group/aihc-bootcamp-winter2018/nborus/")
    task_modality_dir = parent_dir / f"{setting_dict['modality']}_{setting_dict['task']}"
    modeldir = task_modality_dir / "models" / setting_dict['experiment']
    #datadir = task_modality_dir / "localized_New4ChestData/strict/HU_info_lt_-800_ut_-600_p_3.0%_ns_60/"
    #datadir = task_modality_dir / "localized_New4ChestData/300_crop/lung_percent_threshold_12/HU_info_lt_-874_ut_-524_offset_1024/"
    #datadir = task_modality_dir / "New4ChestData/"
    datadir = task_modality_dir / 'HU_New4ChestData/contrast_fluid/'

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
        
        rundir = Path("/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/models/") / experiment / str(t)
        comment = args.comment
        cmd = f"python model/train.py --weighted_loss --comment {comment} --datadir {datadir} --rundir {rundir} --verbose"

        for hparam in setting_dict:
            if hparam in ["pretrained", "weighted_loss", "horizontal_flip"]:
                if setting_dict[hparam] == True:
                    cmd += f" --{hparam}"
            else:
                if hparam not in ['modality', 'task', 'dq_desc', 'experiment']:
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

