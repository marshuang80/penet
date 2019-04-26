#sudo python parse_raw_dicom.py --input_dir /mnt/bbq/intermountain/CTPA_RANDOM_DICOM/ --output_dir /mnt/bbq/intermountain/TEST1/ --csv_path /mnt/bbq/intermountain/intermountain_labeled.csv
#sudo python parse_raw_dicom.py --input_dir /mnt/bbq/intermountain2/CTPA_Images/ --output_dir /mnt/bbq/intermountain2/CTPA_SUBSEG/ --csv_path /mnt/bbq/intermountain2/CTPA_intermountain_2.csv > ./parse_log.txt
sudo python parse_raw_dicom.py --input_dir /mnt/bbq/intermountain2/CTPA_Images/ --output_dir /mnt/bbq/intermountain2/CTPA_COMBINED_SUBSEG/ --csv_path /mnt/bbq/intermountain2/intermountain_combined.csv
