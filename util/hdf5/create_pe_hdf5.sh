#sudo python create_pe_hdf5_intermountain.py --data_dir /mnt/bbq/intermountain/TEST1/images --slice_list /mnt/bbq/intermountain/TEST1/slice_list.txt --use_thicknesses 2.0 --hu_intercept 0 --output_dir /mnt/bbq/intermountain/TEST1/ --csv_path /mnt/bbq/intermountain/intermountain_labeled.csv      
#sudo python create_pe_hdf5_intermountain.py --data_dir /mnt/bbq/intermountain2/CTPA_BALANCED_SLICE/images --slice_list /mnt/bbq/intermountain2/CTPA_BALANCED_SLICE/slice_list.txt --use_thicknesses 2.0 --hu_intercept 0 --output_dir /mnt/bbq/intermountain2/CTPA_BALANCED_SLICE/ --csv_path /mnt/bbq/intermountain2/intermountain_balanced_slice.csv
#sudo python create_pe_hdf5_intermountain.py --data_dir /mnt/bbq/intermountain2/CTPA_FINAL/images --slice_list /mnt/bbq/intermountain2/CTPA_FINAL/slice_list.txt --use_thicknesses 2.0 --hu_intercept 0 --output_dir /mnt/bbq/intermountain2/CTPA_FINAL/ --csv_path /mnt/bbq/intermountain2/intermountain_combined.csv
python create_pe_hdf5_stanford.py --data_dir /data4/intermountain2/CTPA_COMBINED_NEW/images \
				        --output_dir /data4/PE_stanford_test/ \
				        --csv_path /data4/PE_stanford/vision_demo.csv \
				        --slice_list /data4/PE_stanford/slice_list_12_4_dirty.txt \
				        --use_thicknesses 1.25  \
				        --hu_intercept 0 


