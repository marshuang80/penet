## Head & Spine CT

### Usage

1. **Conda**
  - `conda create -n <ENV_NAME> -f requirements.txt`
  - `source activate <ENV_NAME>`
2. **Convert Data**
  - We use `python scripts/convert.py` (run with `-h` for options).
  - Takes a CSV file of annotations and a directory tree containing DICOM files.
  - Writes output to directory containing each individual series as a top-level subdir.
  - Writes a `series_list.{pkl,json}` file to the top of the output directory with metadata.
  The `DataLoader` uses this metadata file.
3. **Train**
  - *R(2+1)D*
        ```
        python train.py --name=<experiment_name> \
        --dataset=<head_or_spine> --data_dir=<path_to_data_dir> \
        --batch_size=8 --gpu_ids=0,1 \
        --num_slices=8 --resize_shape=256,256 --crop_shape=224,224 \
        --num_workers=4 --iters_per_visual=256 --iters_per_print=64 \
        --epochs_per_save=5 --num_epochs=50 --learning_rate=1e-2 \
        --weight_decay=5e-3 --optimizer=sgd --sgd_momentum=0.9 \
        --model_depth=26 --model=R2Plus1D
        ```
  - *VNet*
        ```
        python train.py --name=<experiment_name> \
        --dataset=<head_or_spine> --data_dir=<path_to_data_dir> \
        --batch_size=4 --gpu_ids=0,1 \
        --num_slices=32 --resize_shape=256,256 --crop_shape=224,224 \
        --num_workers=8 --iters_per_visual=80 --iters_per_print=4 \
        --epochs_per_save=5 --num_epochs=300 --learning_rate=1e-3 \
        --weight_decay=5e-4 --optimizer=adam --sgd_momentum=0.99 \
        --lr_scheduler=multi_step --lr_milestones=75,150
        --model_depth=5 --model=VNet
        ```
4. **TensorBoard**
  - While training, launch TensorBoard: `tensorboard --logdir=logs --port=5678`
  - Port forward: `ssh -N -f -L localhost:1234:localhost:5678 <SUNET>@bootcamp`
  - View in browser: `http://localhost:1234/`
5. **Test**
  - Code in `test.py` is a work in progress.
