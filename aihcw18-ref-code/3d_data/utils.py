import json
from pathlib import Path
import os

import torch
from torch.autograd import Variable

import numpy as np

from data.loader import load_data
from model.models import *
import matplotlib
import matplotlib.pyplot as plt
from joblib import Memory
#memory = Memory(cachedir='./cache', verbose=0)
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, auc, f1_score, cohen_kappa_score, precision_score, recall_score

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def getPredMetrics(labels, preds, scores):
    '''
    Gets accuracy, f1, cohen kappa, precision, and recall for passed in labels and predictions (expects numpy arrays)
    '''
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    roc_auc = roc_auc_score(labels, scores)
    return roc_auc, accuracy, f1, kappa, precision, recall

# convert dictionary to object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def transform_inputs(inputs, use_gpu):
    inputs = inputs.cuda()
    inputs = Variable(inputs, requires_grad=False, volatile=True)
    return inputs


def transform_data(data, use_gpu, train=False, grad=False):
    inputs, labels = data
    labels = labels.type(torch.FloatTensor)
    if use_gpu is True:
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs = Variable(inputs, requires_grad=grad, volatile=not train)
    labels = Variable(labels, requires_grad=grad, volatile=not train)
    return inputs, labels

def get_loader(datadir, split):

    # Define the args to groundtruth loader.
    args_dict = {
        'datadir': datadir,
        'batch_size': 1, # change this depending on memory usage
        'workers': 8, # 8 is good default for workers
        'toy': False, # never subsample when evaluating
        'scale': 512, # doesn't matter, only using labels
        'horizontal_flip': False, # doesn't matter, only using labels
        'rotate': False, # doesn't matter, only using labels
        'weighted_loss': False, # doesn't matter, only using labels
        'verbose': False,
        'max_len': 40,
        'crop': 336,
        'invalid_files_lower_threshold': 16, 
        'mode': 'copy',
    }
    args = Struct(**args_dict)

    train_loader, val_loader, test_loader = load_data(args)
    loaders = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
    loader = loaders[split]

    return loader

import pickle
def get_data_distribution(csv_file, data_path):
    '''
    csv_file: csv containing name of ct scan file, and prediction in this format: <filename>,<prediction>
    data_path: directory storing specific data we're analyzing
    '''
    store_dir = '/deep/group/aihc-bootcamp-winter2018/nborus/chest_CT/HU_data_big/distribution/'
    print("Data directory: ",data_path)
    print("CSV file: ", csv_file)
    csv_file_path = data_path + '/' + csv_file
    with open(csv_file_path) as f:
        content = f.readlines()

    print("Num lines in {}: {}".format(csv_file_path, len(content)))
    label_info = {}

    #remove whitespace characters like `\n` at the end of each line
    labels = [int((x.strip()).split(',')[1]) for x in content]
    filenames = [(x.strip()).split(',')[0] for x in content]
    for filename, label in list(zip(filenames, labels)):
        label_info[filename] = label

    true_fraction = sum(labels) / len(labels)
    false_fraction = 1.0 - true_fraction
    label_info['true_fraction'] = true_fraction
    label_info['false_fraction'] = false_fraction
    #print("Content: ", content)
    print("True fraction: ", true_fraction)
    print("False fraction: ",false_fraction)
    with open(store_dir + csv_file, 'wb') as fp:
        pickle.dump(label_info, fp)



def check_diff(orig_dir, modified_dir):
    '''
    Return diff in content btwn 2 directories (to figure out which train examples were discarded during preprocessing-localization)

    '''
    orig_dir_files = set(os.listdir(orig_dir))
    modified_dir_files = set(os.listdir(modified_dir))
    print ("Set diff: ", orig_dir_files - modified_dir_files)



def create_pngs(ct_scan_filepath, store_parent_dir, start=None,end=None):

    '''
    Input:
        ct_scan_filepath: (absolute path) of file containing nx512x512 array, where n represents # of image slices for one chest ct_scan
        store_parent_dir: parent directory for where pngs should be stored
        start: index of first image slice
        end: index of last image slice

    Goal: Create pngs of each image slice in the ct_scan numpy array and store the results in the relevant dir
    '''
    ct_scan_filename = os.path.basename(ct_scan_filepath)
    store_dir = store_parent_dir + "/" + ct_scan_filename + "/"
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    files_stored = os.listdir(store_dir)
    if len(files_stored) > 0:
        print("store_dir {} already populated".format(ct_scan_filename))
        return None

    print("PNGS will be stored in: {}".format(store_dir))
    ct_scan = np.load(ct_scan_filepath)
    if start == None:
        start = 0
    if end == None:
        end = len(ct_scan)

    print("Start: {}, End: {}".format(start, end))
    for index,matrix in enumerate(ct_scan):
        plt.imsave(store_dir + str(index) +".png", matrix, cmap='gray')

def test_lung_heuristic(ct_scan_filepath,upper_threshold, lower_threshold, lung_percentage_bound, num_slices):
    ct_scan = np.load(ct_scan_filepath)
    filename = os.path.basename(ct_scan_filepath)
    print("File: {}, Ct scan shape: {}".format(filename, ct_scan.shape))
    #min_max_vals, relevant_slices, start,end= lung_region_heuristic(ct_scan)
    relevant_slices_cw, relevant_slices_ap, relevant_slices_max_range, lung_percentage_details, relevance_check = lung_region_heuristic(ct_scan, upper_threshold, lower_threshold, lung_percentage_bound, num_slices)
    return relevant_slices_cw, relevant_slices_ap, relevant_slices_max_range, lung_percentage_details, relevance_check
   # print ("File: {} --> Start : {}, End: {}".format(filename, start,end))
  #  print("Min Max info: ", min_max_vals)
  #  print("Relevant slices: ", relevant_slices)
   # return filename, min_max_vals, relevant_slices, start,end

def test_data_big(ct_scan_filepath,upper_threshold, lower_threshold, lung_percentage_bound, num_slices):

    try:
        ct_scan = np.load(ct_scan_filepath)
    except UnicodeDecodeError as e:
        print("*"*60)
        print("Unicode decode error")
        print("File name: ", ct_scan_filepath)
        pass
    except UnicodeError as e:
        print("*"*60)
        print("Unicode error")
        print("File name: ", ct_scan_filepath)
        pass

        #print(e.strerror)

def contiguous_window(relevance_check, num_slices):
    num_slices = 60
    relevant_slices = []
    opt_sum = None
    start, end  = 0,num_slices-1
    opt_start = None
    opt_end = None
    while end < len(relevance_check):
        curr_sum = sum(relevance_check[start:end+1])
        if opt_sum == None or curr_sum > opt_sum:
            opt_sum = curr_sum
            opt_start = start
            opt_end = end
        start += 1
        end += 1

    for i,elem in enumerate(relevance_check):
        if opt_start == i:
            print("[[", end=' ')
        print(elem, end=' ')
        if opt_end == i:
            print("]]", end=' ')

    print("")

    relevant_slices = list(range(opt_start, opt_end+1))
    return relevant_slices

def max_relevant_range(relevance_check):
    start = 0
    end = len(relevance_check) - 1
    for i, is_relevant in enumerate(relevance_check):
        if is_relevant:
            start = i
            break

    for i in range(len(relevance_check)-1, -1, -1):
        is_relevant = relevance_check[i]
        if is_relevant:
            end = i
            break

    relevant_slices = list(range(start,end+1))
    #print("start: ", start, ": end: ", end,"relevant: ", relevant_slices)
    return relevant_slices

def all_possible_slices(relevance_check):
    relevant_slices = []
    #Get all relevant slices
    for i, is_relevant in enumerate(relevance_check):
        if is_relevant:
            relevant_slices.append(i)

    return relevant_slices

def lung_region_heuristic(ct_scan, upper_threshold, lower_threshold, lung_percentage_bound, num_slices):
    '''
    Input: ct_scan - n x 512 x 512 array, where n represents # of image slices for one chest ct_scan
    Output: lung_region - m x 512 x 512 array, where m represents region containing lungs

    Approach:
    -Using the Hunfield scale as it applies to medical-grade CT scans (https://en.wikipedia.org/wiki/Hounsfield_scale)
    -HU for lung region are in -700 to −600 range
    -For each image array in ct_scan, we check whether any of it's entries have values in lung HU range (with some allowance). If so, we include it as a relevant slice
    '''
    n,_,_ = ct_scan.shape
    offset = 1000
    lung_hu_lower_threshold = lower_threshold + offset
    lung_hu_upper_threshold = upper_threshold + offset
    print("Lower threshold + offset: ", lung_hu_lower_threshold)
    print("Upper threshold + offset: ", lung_hu_upper_threshold)
    #offset_array = np.full(ct_scan[0].shape, offset, dtype='uint16')
    min_result_slices = 10
    within_lung_hu_range = lambda x: x >= lung_hu_lower_threshold and x <= lung_hu_upper_threshold
    relevant_slices = []
    relevance_check = []
    min_max_vals = []
    lung_percentage_details = []
    lung_area_percentage_threshold = lung_percentage_bound
    image_area = 512*512
    for index,image in enumerate(ct_scan):
        vfunc = np.vectorize(within_lung_hu_range)
        output = vfunc(image)
        lung_percentage = (100.0 * np.sum(output)) / image_area
       # print(np.sum(output), lung_percentage, lung_percentage_bound)
       # print("# of entries with HU val in lung tissue range: ", np.sum(output), "Lung percentage: ", lung_percentage)
      # print("Output: {}".format(output))
        lung_percentage_details.append(lung_percentage)
        if lung_percentage >= lung_area_percentage_threshold:
            #print("Lung percentage true", lung_percentage)
            relevance_check.append(1)
        else:
            relevance_check.append(0)

    print("Lung percentage details per slice: ", lung_percentage_details)
    print("Relevance check:" , relevance_check)

    relevant_slices_cw = contiguous_window(relevance_check, num_slices)
    print("Contiguous window relevant slices: {}, {}".format(relevant_slices_cw, len(relevant_slices_cw)))

    relevant_slices_ap = all_possible_slices(relevance_check)
    print("All possible relevant slices check: {}, {}".format(relevant_slices_ap, len(relevant_slices_ap)))

    relevant_slices_max_range = max_relevant_range(relevance_check)
    print("Max relevant range (from first 1 to last 1): {}, {}".format(relevant_slices_max_range , len(relevant_slices_max_range)))

   # assert start-end >= min_result_slices, "Region found too small, maybe change threshold"
   # percentage_relevant_slices = 100.0 * len(relevant_slices_cw) / n
   # print("Out of {} slices, {} were relevant. % of relevant slices extracted by heuristic: {}".format(n, len(relevant_slices), percentage_relevant_slices))
    return relevant_slices_cw, relevant_slices_ap, relevant_slices_max_range, lung_percentage_details, relevance_check


def create_directories(np_array_dir):
    for np_file in os.listdir(np_array_dir):
        print("Create directories: file: ", np_file)
        create_directory(np_file)

def check_HU_vals(ct_scan_filepath, index):
    filename = os.path.basename(ct_scan_filepath)
    print("Checking HU vals for file: {}, slice #{}".format(filename, index))
    ct_scan = np.load(ct_scan_filepath)
    x_start = 175
    x_end = 200
    y = 256
    image_slice = ct_scan[index]
    lung_crop = image_slice[200:300,150:200]
    data =lung_crop.flatten()
    bins = np.arange(100, 1400, 100) # fixed bin size
    plt.xlim([min(data)-5, max(data)+5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('HU distribution for image {}, slice {}, on crop: [{}:{},{}:{}'.format(filename, index, 200,300,150,200))
    plt.xlabel('Hunsfield units (bin size = 100)')
    plt.ylabel('count')
    plt.savefig('histogram.png',bbox_inches='tight')
   # for row in lung_crop:
   #     print (row)

   # print (lung_crop.shape)
   # for y in range(len(lung_crop)):
   #     for x in range(len(lung_crop[0])):
   #         print("({},{}) --> {}".format(x+112,y+175,lung_crop[y,x]))

def check_lung_crop_png(ct_scan_filepath, index):
    filename = os.path.basename(ct_scan_filepath)
    print("Checking HU vals for file: {}, slice #{}".format(filename, index))
    ct_scan = np.load(ct_scan_filepath)
    x_start = 175
    x_end = 200
    y = 256
    image_slice = ct_scan[index]
   #lung_crop = image_slice[175:350,112:252]
    lung_crop = image_slice[200:300,150:200]
    plt.imsave(str(index) +".png", lung_crop, cmap='gray')




def create_directory(ct_scan_filename):
    '''
    Input: filename storing ct_scan - n x 512 x 512 array, where n represents # of image slices for one chest ct_scan
    Create folder in save_dir with same name as ct_scan_filename
    '''
    save_dir = '/deep/group/aihc-bootcamp-winter2018/nborus/chest_CT/images'
    ct_scan_dir = Path(save_dir + '/' + ct_scan_filename)
    if not ct_scan_dir.exists():
        ct_scan_dir.mkdir()


def get_model_and_loader(model_path, dataset):

    # Load model args
    with open(Path(model_path).parent / 'args.json') as args_f:
        model_args_dict = json.load(args_f)
    model_args = Struct(**model_args_dict)
    # Get loader from args
    train_loader, val_loader, test_loader = load_data(model_args)
    loaders = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
    loader = loaders[dataset]
    # Load model
    model = model_dict[model_args.model](model_args, loader.dataset.num_classes).cuda()
 

    model.load_state_dict(torch.load(model_path))
    '''    
    state = model.state_dict()
    loaded_state = torch.load(model_path)
    for k in state.keys():
        if k not in loaded_state and 'features' in k:
            new_k = k.replace('features.', '')
            state[k] = loaded_state[new_k]
        elif k not in loaded_state and ('features.' + k) in loaded_state:
            new_k = 'features.' + k
            state[k] = loaded_state[new_k]
        else:
            state[k] = loaded_state[k]
    model.load_state_dict(state) '''
    model.eval()
    csv_file_path = model_args.datadir +'/'+ dataset +".csv"

    return model, loader, csv_file_path



def compute_probs_from_objects(model, loader):
    all_losses = []
    all_labels = []
    all_probs = []

    num_batches = len(loader)

    for batch in loader: #TODO tqdm
        inputs, labels = transform_data(batch, True, train=False)
        predictions = model.forward(inputs.float())
        # BCEWithLogitsLoss averages over the batch by default.
        batch_loss = nn.BCEWithLogitsLoss()(predictions, labels)

        all_probs.append(predictions.data.cpu().numpy().flatten())
        all_labels.append(labels.data.cpu().numpy().flatten())
        all_losses.append(batch_loss.data.cpu().numpy().flatten())

    all_probs = sigmoid(np.concatenate(all_probs))

    all_predictions = (all_probs > 0.5).astype(int)
    all_labels = np.concatenate(all_labels).astype(int)
    all_losses = np.concatenate(all_losses)

    total = len(all_predictions)
    tot_pred_pos = np.sum(all_predictions == 1)
    tot_pred_neg = np.sum(all_predictions == 0)

    neg_fraction = tot_pred_neg / total
    pos_fraction = tot_pred_pos / total
    avg_loss = np.sum(all_losses) / num_batches

    roc_auc, accuracy, f1, kappa, precision, recall = getPredMetrics(all_labels, all_predictions, all_probs)
    print ("ROC", roc_auc)
    print ("Acc", accuracy)
    print ("loss", avg_loss)
    print ("precision", precision)
    print ("recall", recall)

    return all_probs, all_labels



def compute_probs_from_paths(model_path, dataset):

    model, loader, _ = get_model_and_loader(model_path, dataset)
    return compute_probs_from_objects(model, loader)

