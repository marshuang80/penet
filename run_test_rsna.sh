python ./test_rsna.py --phase all \
                    --results_dir ./results \
                    --ckpt_path /data4/PE_stanford/ckpts/penet_best.pth.tar \
                    --name stanford \
                    --dataset pe \
                    --data_dir /data4/rsna/parsed_data/train/ \
                    --gpu_ids 3