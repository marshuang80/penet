# Using dq_script.py to Run Experiments
`dq_script.py` will generate *two* scripts to run with the arguments you provided, one with seed 123 and one with seed 124.

Experiment results will be saved in 
`/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/aaa_mr_knee_experiments/{timestamp}_{task}_{view}_{cnn}_{pooling}_{keyword}` 
({model} will replace {cnn} if Alexnet is used)

Tensorboard logdir is: 
`/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/aaa_mr_knee_experiments/runs`

Toy set experiment results will be saved in 
`/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/aaa_mr_knee_experiments_toy/{timestamp}_{task}_{view}_{cnn}_{pooling}_{keyword}` 
({model} will replace {cnn} if Alexnet is used)

Tensorboard logdir for toy set experiments is:
`/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/aaa_mr_knee_experiments_toy/runs`


In your folder where your git clone is, run `python dq_script.py` specifying the following arguments:
## Required args:
 - task: abnormal, acl, meniscus
 - view: sagittal, axial, coronal
 - keyword: keyword that can distinguish the experiment (will be part of folder name)

## Model args:
If you want to use Alexnet:
 - model: alexnet
 - pooling: pooling layers to use in forward, avgavg, avgmax, maxavg, maxmax (Current default is avgmax)

If you want to use Resnet or Densenet: 
 - model: cnn (This is set as default, so don't have to specify)
 - cnn: resnet18, resnet34, resnet50, resnet101, resnet152, densenet 121, densenet161, densenet169, densenet201 (Current default is resnet50)
 - pooling: pooling layers to use in forward, avgavg, avgmax, maxavg, maxmax (Current default is avgmax)
 
## Training args:
 - lr: learning rate, default 1e-4
 - weighted_loss: currently set as default, turned off if oversample is true
 - epochs: default 35
 - oversample: needs double-checking, gives error with auroc
 - optimizer: default 'adam'
 - weight_decay: default 1e-2
 - max_patience: patience for lr scheduler to drop lr
 - factor: new lr = lr * factor

## Data Loading args:
 - toy
 - rgb: default true
 - fix_num_slices: whether to cap slices or not
 - num_slices: how many slices to cap at
 - fixing_method: inner, uniform
 - normalize: knee, imagenet, none

## Data Augmentation args:
 - scale: default 256
 - horizontal_flip: default true
 - rotate: default 0
 - shift: default 0
 - reverse: default true

## Example:
`python dq_script.py --task meniscus --view axial --cnn resnet18 --pooling avgmax --keyword test_dq_script --toy --epochs 5 --rotate 30`
