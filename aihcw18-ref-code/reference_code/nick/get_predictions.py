from evaluate.predict import predict
from evaluate.get_best_model import get_best_models
from sklearn import metrics
import matplotlib.pyplot as plt
import time
from collections import defaultdict

view_to_index = {'sagittal':0,'coronal':1,'axial':2}

single_view_models = {
    'abnormal': [
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1521632163_abnormal_sagittal_alexnet_sagittal/val0.10419943183660507_train0.09963367879390717_epoch13',
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1521632177_abnormal_coronal_alexnet_coronal/val0.20479801297187805_train0.09228355437517166_epoch32',
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1521632186_abnormal_axial_alexnet_axial/val0.10126200318336487_train0.10181700438261032_epoch14',
    ],
    'acl': [
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1521633210_acl_sagittal_alexnet_sagittal/val0.21643468737602234_train0.12244126200675964_epoch23',
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1521633273_acl_coronal_alexnet_coronal/val0.189946249127388_train0.1195748895406723_epoch19',
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1521633347_acl_axial_alexnet_axial/val0.13460037112236023_train0.08850131183862686_epoch47',
    ],
    'meniscus': [
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1522784634_meniscus_sagittal_alexnet_sagittal/val0.27973052859306335_train0.2262496054172516_epoch13',
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1521633002_meniscus_coronal_alexnet_coronal/val0.2719162106513977_train0.17812278866767883_epoch26',
        '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/nbien-full-dataset/1521633053_meniscus_axial_alexnet_axial/val0.2577749490737915_train0.18979603052139282_epoch29',
    ],
}

def get_predictions(task, view, split):
    
    predictions = []
    model_path = single_view_models[task][view_to_index[view]]
        
    preds = predict([model_path], split, task)
    
    for i in range(preds.shape[0]):
        predictions.append(preds[i,0])

    outfile = open('../predictions/' + split + '/' + task + '/' + view + '.csv', 'w')
    for p in predictions:
        outfile.write(str(p) + '\n')
    outfile.close()

for task in ['abnormal', 'acl', 'meniscus']:
    for view in ['sagittal', 'coronal', 'axial']:
        for split in ['train', 'valid', 'test', 'valid2']:
            get_predictions(task, view, split)

