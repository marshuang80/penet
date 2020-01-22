#python ./test_intermountain.py --phase test --pkl_path /home/marshuang80/intermountain_data/series_list.pkl --results_dir ../results_intermountain --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name test --dataset pe --data_dir /home/marshuang80/intermountain_data/ --gpu_ids 0
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name test_shift --dataset pe --data_dir /mnt/bbq/intermountain/TEST1/ --gpu_ids 0 #--pkl_path /mnt/bbq/intermountain/CTPA_RANDOM_NPY_SHIFT/series_list.pkl 
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name rerun --dataset pe --data_dir /mnt/bbq/intermountain/CTPA_SORTED/ --gpu_ids 0 #--pkl_path /mnt/bbq/intermountain/CTPA_RANDOM_NPY_SHIFT/series_list.pkl 
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name test_shift_norm --dataset pe --data_dir /mnt/bbq/intermountain/CTPA_RANDOM_NPY_FLIP/ --gpu_ids 0 #--pkl_path /mnt/bbq/intermountain/CTPA_RANDOM_NPY_SHIFT/series_list.pkl 
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name test_working --dataset pe --data_dir /mnt/bbq/intermountain/CTPA_SORTED --gpu_ids 0 #--pkl_path /mnt/bbq/intermountain/CTPA_RANDOM_NPY_SHIFT/series_list.pkl
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name test_working --dataset pe --data_dir /mnt/bbq/intermountain/CTPA_SORTED_FULL --gpu_ids 0 #--pkl_path /mnt/bbq/intermountain/CTPA_RANDOM_NPY_SHIFT/series_list.pkl
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name joe_friend --dataset pe --data_dir /home/marshuang80/joe --gpu_ids 0 #--pkl_path /mnt/bbq/intermountain/CTPA_RANDOM_NPY_SHIFT/series_list.pkl
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain2 --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name intermountain2 --dataset pe --data_dir /mnt/bbq/intermountain/CTPA_SORTED --gpu_ids 0
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain2 --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name combined --dataset pe --data_dir /mnt/bbq/intermountain2/CTPA_BALANCED_SLICE/ --gpu_ids 0
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain2 --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name inter_0.7_sub --dataset pe --data_dir /mnt/bbq/intermountain2/CTPA_FINAL --gpu_ids 0
#python ./test_intermountain.py --phase test --results_dir ../results_intermountain2 --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name inter_0.7_sub --dataset pe --data_dir /mnt/bbq/intermountain2/CTPA_COMBINED --gpu_ids 1
python ./test_intermountain.py --phase test \
                               --results_dir ../results \
                               --ckpt_path /data4/PE_stanford/ckpts/best.pth.tar \
                               --name stanford \
                               --dataset pe \
                               --data_dir /data4/intermountain2/CTPA_FINAL \
                               --gpu_ids 0

#python ./count.py --phase test --results_dir ../results --ckpt_path /mnt/bbq/PE_stanford/ckpts/best.pth.tar --name stanford --dataset pe --data_dir /mnt/bbq/PE_stanford --gpu_ids 0


