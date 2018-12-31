from subprocess import Popen, check_call
from collections import namedtuple
from pathlib import Path
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--test', action="store_true")
args = parser.parse_args()

# NOTE: use your name as the first part of the experiment
# to distinguish your experiments from others!
modality = "ct"
task = "chest_pe"
experiment = 'tanay_30AprData_tests_7may'
if not args.test:

    HParams = namedtuple("HParams", ["modality", "task", # Experiment Parameters
                                     "model", "pretrained", "weighted_loss", "pooling", # Model Parameters
                                     "optimizer", "learning_rate", "batch_size", "epochs", # Training Parameters
                                     "mode", "scale", "horizontal_flip", "rotate", "crop", # Data Parameters
                                     "experiment", 'max_len', 'decay'], verbose=False)

    settings = []

    for i, lr in enumerate([0.00001]):
        settings.append(HParams(modality=modality, task=task,
                                model="DoublePoolAlexNet", pretrained=True, weighted_loss=True, pooling='avgmax',
                                optimizer="adam", learning_rate=lr, batch_size=1, epochs=50, mode='copy',
                                scale=512, horizontal_flip=False, rotate=5, crop=336, max_len=10, decay=0.005,
                                experiment=experiment))

else:
    HParams = namedtuple("HParams", ["modality", "task", # Experiment Parameters
                                     "n",
                                     "split",
                                     "metric",
                                     "experiment"], verbose=False)

    settings = [HParams(modality=modality, task=task,
                        n=1,
                        split="valid", # one of valid, test, radio-test-mini
                        metric="roc_auc", # one of roc_auc, pr_auc, f1. this param is ignored if radio-test-mini is used.
                        experiment=experiment)]

k = 0
for setting in settings:
    setting_dict = setting._asdict()

    parent_dir = Path("/deep/group/aihc-bootcamp-winter2018/medical-imaging/")
    task_modality_dir = parent_dir / f"{setting_dict['modality']}_{setting_dict['task']}"
    modeldir = task_modality_dir / "models" / setting_dict['experiment']
    datadir = '/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/30AprData/localized/'

    t = int(time.time())

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

        rundir = Path("/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/models/") / experiment / str(t)
        cmd = f"python model/train.py --datadir {datadir} --rundir {rundir} --verbose "

        for hparam in setting_dict:
            if hparam in ["pretrained", "weighted_loss", "horizontal_flip"]:
                if setting_dict[hparam] == True:
                    cmd += f" --{hparam}"
            else:
                if hparam not in ['modality', 'task', 'experiment']:
                    cmd += f" --{hparam} {setting_dict[hparam]}"

        cmd += f" > {rundir}/log.txt"

        if not rundir.exists():
            rundir.mkdir(parents=True)

    cmd = f"export CUDA_VISIBLE_DEVICES={k % 4}; {cmd} &"
    print(cmd)
    k += 1

    # For timestamps when not using dq.
    time.sleep(2)

