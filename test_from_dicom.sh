python test_from_dicom.py --input_study  /data4/PE_stanford/Stanford_data/ \
                          --series_description CTPA \
                          --ckpt_path /data4/PE_stanford/ckpts/penet_best.pth.tar \
                          --device cuda \
                          --gpu_ids 0