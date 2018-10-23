# Using TensorBoard for Data Visualization

To use, activate imaging-vis virtual environment
`source /deep/group/packages/miniconda3/bin/activate imaging-vis`

Tensorboard will store experiment data in 'runs' folder from where you run it, so run consistently from medical-imaging-starter-pack in order to compare experiments. You can specify a --comment option to briefly describe the experiment purpose.
Metrics will automatically save when train.py is run.

To view, execute
`tensorboard --logdir ['runs' folder path]`

Then, from your local terminal, execute
`ssh -N -f -L localhost:6006:localhost:6006 [username]@[remote host]`
to forward from to the first localhost address locally from the second (the remote localhost address)

You can then launch localhost:6006 from your browser. You will be able to compare metrics from any groupings of your experiments in real time.
Currently tracked metrics (train.py):
- loss
- accuracy
- f1
- kappa
- precision
- recall
- learning rate
- precision/recall curve by epoch

