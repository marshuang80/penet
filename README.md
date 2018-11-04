## Chest CT (Moved from pev2 branch of Head & Spine CT)

### Usage
 1. **Conda**
This currently exists on /deep as as `source /deep/group/packages/miniconda3/bin/deactivate /deep/group/packages/miniconda3/envs/ctpe`
  - `conda create -n <ENV_NAME> -f requirements.txt`
  - `source activate <ENV_NAME>`
  
2. **Convert Data**
  - We use `python scripts/convert.py` (run with `-h` for options).
  - Takes a CSV file of annotations and a directory tree containing DICOM files.
  - Writes output to directory containing each individual series as a top-level subdir.
  - Writes a `series_list.{pkl,json}` file to the top of the output directory with metadata.
  The `DataLoader` uses this metadata file.
  
3. **Train**
  - *X-Net*
        python train.py --name=pe_tanay_lr1_wd3_ab30_ft1e-2 --model=XNetClassifier --data_dir=/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/tanay_data_11_4/ --dataset=pe --epochs_per_eval=1 --epochs_per_save=1 --learning_rate=1e-1 --gpu_ids=0,1,2,3 --resize_shape=208,208 --crop_shape=192,192 --weight_decay=1e-3 --batch_size=8 --num_slices=16 --iters_per_print=8 --iters_per_visual=8000 --num_epochs=150 --model_depth=50 --optimizer=sgd --sgd_momentum=0.9 --sgd_dampening=0.9 --num_classes=1 --do_classify=True --do_segment=False --agg_method=max --include_normals=True --use_pretrained=True --fine_tune=True --fine_tuning_boundary=classifier --fine_tuning_lr=1e-2 --ckpt_path=/sailhome/chute/hsct/ckpts/xnet_kin_90.pth.tar --num_classes=1 --cudnn_benchmark=False --use_hem=False --num_workers=16 --abnormal_prob=0.3 --num_visuals=8 --num_epochs=1 --lr_scheduler=cosine_warmup --lr_decay_step=300000 --lr_warmup_steps=10000 --best_ckpt_metric=val_AUROC --save_dir=/sailhome/tanay/ckpts/
        ```
4. **TensorBoard**
  - While training, launch TensorBoard: `tensorboard --logdir=logs --port=5678`
  - Port forward: `ssh -N -f -L localhost:1234:localhost:5678 <SUNET>@sc`
  - View in browser: `http://localhost:1234/`
5. **Test**
  - Code in `test.py`
6. **Jupyter Notebook**
  - Run `jupyter notebook --port=8080` on remote (sc) in the repo dir and run `ssh -N -L 8080:localhost:8080 <suid>@sc` on your local machine. 
