import pandas as pd
import numpy as np 
import torch
import torch.utils.data as data
from torch.autograd import Variable
from sklearn import svm, metrics
import utils
import sys

def construct_output_table(models, loader, threshhold = None, filename = None):
    #Models: dict mapping 'ax', 'cor', or 'sag' to the corresponding model
    #Loaders: same thing, but with dataloaders
    #Change filename to not None when we want to save.
    views = ['sagittal', 'coronal', 'axial'] #Allows us to fix order
    output_table = []
    #X = np.array([])
    #np.save('./features.npy', X)
    X = []
    Y = []
    #Assumes single class. 
    counter = 0
    #Right now only works with batch size 1...
    cams = []
    def hook_feature(module, input, output):
        cams.append(input[0].data.cpu().numpy())
    
    for view in views:
        h = models[view]._modules.get('classifier').register_forward_hook(hook_feature)

    for batch, label in loader:
        counter += 1
        outputs = []
        concat_features = np.array([])
        if counter % 100 == 0:
            print("Processed {} out of {} batches".format(counter, len(loader)))
            if counter >= 900:
                break
        for view in views:
            cams = []
            view_output = (models[view](Variable(batch[view].float().cuda()))).data.cpu().numpy().flatten()[0]
            concat_features = np.concatenate((concat_features, np.squeeze(cams[0], 0)))
            
            if threshhold is not None:
                view_output = (view_output > threshhold)
            outputs.append(view_output)
        outputs.append(label.numpy()[0])
        output_table.append(np.array(outputs))
        
        #X = np.load('./features.npy')
        #X = np.append(X, concat_features, axis=0)
        #np.save('./features.npy', X)
        #del X
        X.append(concat_features)
        Y.append(label.numpy()[0])
        #print(len(X))
        #print(X[0].shape)
        #print(len(Y))
        #print(Y[0].shape)
    output_table = np.array(output_table)
    output_dataFrame = pd.DataFrame(data = output_table, columns = ['SAG', 'COR', 'AX', 'LABEL'])
    if filename is not None:
        output_dataFrame.to_csv(filename)
    #Modify if using fewer than 3 views
    #X = output_table[:, :3]
    #Y = output_table[:, 3]
    #X = np.load('./features.npy')
    #X = np.reshape(X, (-1, 768))
    #print(X.shape)
    return np.array(X), np.array(Y)

def construct_svm(X_train, Y_train):
    clf = svm.SVC(probability=True, class_weight='balanced')
    clf.fit(X_train, Y_train)
    return clf


if __name__ == '__main__':
    #ab_cor = ('nbien-full-dataset/1521632177_abnormal_coronal_alexnet_coronal/val0.1570933759212494_train0.12708044052124023_epoch11', 'coronal')
    #ab_sag = ('nbien-full-dataset/1521632163_abnormal_sagittal_alexnet_sagittal/val0.10419943183660507_train0.09963367879390717_epoch13', 'sagittal')
    #ab_ax = ('nbien-full-dataset/1521632186_abnormal_axial_alexnet_axial/val0.10126200318336487_train0.10181700438261032_epoch14', 'axial')
    #sag_model_path = '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/' +\
    #        ab_sag[0]
    #ax_model_path = '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/' +\
    #            ab_ax[0]
    #cor_model_path = '/deep/group/aihc-bootcamp-winter2018/medical-imaging/mr_knee_abnormality/models/' +\
    #            ab_cor[0]
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
    task = sys.argv[1]
    sag_model_path = single_view_models[task][0]
    cor_model_path = single_view_models[task][1]
    ax_model_path = single_view_models[task][2]
    dataset = 'train'
    sag_model, sag_loader = utils.get_model_and_loader(sag_model_path, dataset, view = 'all')
    ax_model, ax_loader = utils.get_model_and_loader(ax_model_path, dataset, view = 'all')
    cor_model, cor_loader = utils.get_model_and_loader(cor_model_path, dataset, view = 'all')

    models = {'sagittal': sag_model, 'coronal': cor_model, 'axial': ax_model}
    print("Constructing training output table...")
    X_train, Y_train = construct_output_table(models, ax_loader)
    print("Saving Table...")
    np.save('X_output_train.npy', X_train)
    np.save('Y_output_train.npy', Y_train)
    print("Training SVM")
    trained_svm = construct_svm(X_train, Y_train)
    dataset = 'valid'
    sag_model, sag_loader = utils.get_model_and_loader(sag_model_path, dataset, view = 'all')
    ax_model, ax_loader = utils.get_model_and_loader(ax_model_path, dataset, view = 'all')
    cor_model, cor_loader = utils.get_model_and_loader(cor_model_path, dataset, view = 'all')
    print("Constructing validation output table")
    X_valid, Y_valid = construct_output_table(models, ax_loader)
    print("Getting predictions")
    predictions = trained_svm.predict(X_valid)
    probabilities = trained_svm.predict_proba(X_valid)
    total_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == Y_valid[i]:
            total_correct += 1
    print("Accuracy: ", total_correct / len(predictions))
    print('AUROC', metrics.roc_auc_score(probabilities, Y_valid))
    with open('probabilities', 'w') as f:
    	for i in range(len(probabilities)):
            f.write(probabilities[i] + '\n')


