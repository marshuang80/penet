#sudo python parse_raw_dicom.py --input_dir /mnt/bbq/intermountain/CTPA_RANDOM_DICOM/ --output_dir /mnt/bbq/intermountain/TEST1/ --csv_path /mnt/bbq/intermountain/intermountain_labeled.csv
#sudo python parse_raw_dicom.py --input_dir /mnt/bbq/intermountain2/CTPA_Images/ --output_dir /mnt/bbq/intermountain2/CTPA_SUBSEG/ --csv_path /mnt/bbq/intermountain2/CTPA_intermountain_2.csv > ./parse_log.txt
python3 parse_raw_dicom.py --input_dir /data4/intermountain/CTPA/test --output_dir ~/  --csv_path /data4/intermountain2/intermountain_combined.csv
