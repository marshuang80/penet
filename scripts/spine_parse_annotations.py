"""Parse annotated csvs

   Usage: merge_and_clean_annotations('/data/CT-CSPINE/processed-studies/C-sp_Fx_final.csv', '/data3/CT-CSPINE/20150514_labeled_traintestvalsplit.csv', '/data/CT-CSPINE/processed-studies/data_20180523_233650')

"""

import pandas as pd
import numpy as np
import json
import csv
import os
from collections import defaultdict

def load_annotations(annotations_csv_path):
    """ Loads the annotations csv.
        
        Args:
            annotations_csv_path: os path to the csv
            
        Returns:
            Dataframe containing the annotations
    """
    annotations = pd.read_csv(annotations_csv_path)
    annotations.rename(inplace=True,columns={"Start Slice#" : "sagittal_start", 
                                                   "End Slice#" : "sagittal_end", 
                                                   "Start Slice#.1": "axial_start",
                                                   "End Slice#.1": "axial_end",
                                                   "Sag Se" : "sagittal_series",
                                                   "Ax Se ": "axial_series",
                                                   "Unstable ": "Unstable"})
    return annotations

def remove_nonfractures(df):
    """ Interactive reviewer for non-fractures.
        Args: 
            df: Pandas dataframe
        Returns:
            Dataframe with non-fractures removed
    """ 
    df = df[df['Remove']!= 1]
    return df

def save_sliceNumberAnnotations(df, label_type, data_dir):
    """ Save slice number annotations
        Args: 
            df: Pandas dataframe with annotations
            label_type: 'sagittal' or 'axial'
            data_dir: CT-spine processed-studies data directory
        Returns:
            list of series references containing validated fractures
        Outputs:
            CSV with columns Series_Ref, StartSlice, EndSlice, Comments corresponding to the identified fracture
    """
    ref_column = label_type + '_ref'
    start_column = label_type + '_start'
    end_column = label_type + '_end'
    df.reset_index(inplace=True)
    
    fractures = defaultdict(list)
    file = os.path.join(data_dir,label_type + '_sliceNumber_annotations.csv')
    with open(file,'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Series_Ref','StartSlice','EndSlice','Comments'])
        df_subset = df[df[start_column].apply(lambda x: x.isnumeric() or ';' in x )]
        df_subset = df_subset[[ref_column,start_column,end_column,'Comments']]
        for idx, row in df_subset.iterrows():
            if row[ref_column] not in fractures[label_type]:
                fractures[label_type].append(row[ref_column])
            starts = row[start_column].split(';')
            ends = row[end_column].split(';')
            for fIdx in range(len(starts)):
                writer.writerow([row[ref_column],int(starts[fIdx]) - 1, int(ends[fIdx]) - 1, row['Comments']])
    return fractures

def create_labels_csv(series_type, images_dir, label_csv_path, fractures):
    """ Save csvs with train/test/val labels
        Args: 
            series_type: 'sagittal' or 'axial'
            images_dir: CT-spine processed-studies data directory
            label_csv_path: Path to the train/test/val splits by study
            fractures: list of series references validated to contain fractures
    """

    refID_to_trial = json.load(open(os.path.join(data_dir,'ref_to_trial.json'),'rb'))
    patient_splits = pd.read_csv(label_csv_path,index_col='TrialAccNum')
    dataset_dict = defaultdict(list)

    for trial_ref, accession_num in refID_to_trial.items():
        accession_num = int(accession_num)
        if accession_num in patient_splits.index:
            label_bool = patient_splits.loc[accession_num,'Fracture']
            dataset = patient_splits.loc[accession_num,'Dataset']
            label = 'No Fracture'
            if label_bool:
                label = 'Fracture'
            dataset_dict[dataset].append((trial_ref,label))

    for dataset in ['Train','Val','Test']:
        file = os.path.join(images_dir, dataset.lower() + '_' + series_type + '.csv')
        with open(file,'w') as csv_file:
            writer = csv.writer(csv_file)
            for trial_ref, label in dataset_dict[dataset]:
                if series[trial_ref] == series_type:
                    if label == 'No Fracture':
                        writer.writerow([trial_ref,label])
                    if label == 'Fracture':
                        if trial_ref in fractures[series_type] or dataset == 'Test':
                            writer.writerow([trial_ref,label])

def merge_and_clean_annotations(annotations_csv_path, label_csv_path, data_dir):
    annotations = load_annotations(annotations_csv_path)
    all_studies = pd.read_csv(label_csv_path)
    series = json.load(open(os.path.join(data_dir, 'series.json'),'rb'))
    
    df = pd.merge(all_studies, annotations, how="inner", left_on="acc_num", right_on="acc_num")
    df.set_index("TrialAccNum",inplace=True)
    df = remove_nonfractures(df)
    
    df.reset_index(inplace=True)
    df['sagittal_ref'] = df.apply(lambda row: '%d_%03d' % (int(row['TrialAccNum']),int(row['sagittal_series'])) if str(row['sagittal_series']).isnumeric() else '',axis=1)
    df['axial_ref']    = df.apply(lambda row: '%d_%03d' % (int(row['TrialAccNum']),int(row['axial_series']))    if str(row['axial_series']).isnumeric() else '',axis=1)
    df.set_index("TrialAccNum",inplace=True)    
    
    ## Save slice number annotations
    sagittal_fractures = save_sliceNumberAnnotations(df, 'sagittal', data_dir)
    axial_fractures = save_sliceNumberAnnotations(df, 'axial', data_dir)
    
    ## Save train/test/validation splits
    create_labels_csv('sagittal', data_dir, label_csv_path, sagittal_fractures)
    create_labels_csv('axial', data_dir, label_csv_path, axial_fractures)
    
## usage merge_and_clean_annotations('/data/CT-CSPINE/processed-studies/C-sp_Fx_final.csv', '/data3/CT-CSPINE/20180603_labeled_traintestvalsplit.csv', '/data/CT-CSPINE/processed-studies/data_20180523_233650')
