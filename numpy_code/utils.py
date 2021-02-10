from dependencies import *

import hashlib 
import pickle
import json
import os
import itertools
import numpy as np
import tqdm
import shutil
from shutil import copyfile

# =======================================
# haven
from haven import haven_jupyter as hj
from haven import haven_results as hr
# from haven import haven_dropbox as hd
from haven import haven_utils as hu

import pylab as plt 
import pandas as pd 
import numpy as np
import copy 
import glob 
from itertools import groupby 

def save_pkl(fname, data):
    """Save data in pkl format."""
    # Save file
    fname_tmp = fname + "_tmp.pkl"
    with open(fname_tmp, "wb") as f:
        pickle.dump(data, f)
    os.rename(fname_tmp, fname)


def load_pkl(fname):
    """Load the content of a pkl file."""
    with open(fname, "rb") as f:
        return pickle.load(f)

def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

def save_json(fname, data):
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

def read_text(fname):
    # READS LINES
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # lines = [line.decode('utf-8').strip() for line in f.readlines()]
    return lines


def make_minibatches(n, m, minibatch_size):
    ''' Create m minibatches from the training set by sampling without replacement.
        This function may sample the training set multiple times.
    Parameters:
        n: the number of examples in the dataset
        m: number of minibatches to generate
        batch_size: size of the desired minibatches'''

    k = math.ceil(m * minibatch_size / n)
    batches = []
    for i in range(k):
        batches += minibatch_data(n, minibatch_size)

    return batches


def minibatch_data(n, batch_size):
    '''Splits training set into minibatches by sampling **without** replacement.
    This isn't performant for large datasets (e.g. we should switch to PyTorch's streaming data loader eventually).
    Parameters:
        n: the number of examples in the dataset
        batch_size: size of the desired minibatches'''
    # shuffle training set indices before forming minibatches
    indices = np.arange(n)
    np.random.shuffle(indices)

    batches = []
    num_batches = math.floor(n / batch_size)
    # split the training set into minibatches
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        stop_index = (batch_num + 1) * batch_size

        # create a minibatch
        batches.append(indices[start_index:stop_index])

    # generate a final, smaller batch if the batch_size doesn't divide evenly into n
    if num_batches != math.ceil(n / batch_size):
        batches.append(indices[stop_index:])

    return batches

def reset(model):
    # reset the model
    for param in model.parameters():
        param.data = torch.zeros_like(param)
    loss_results = []
    gradnorm_results = []

    return model

def compute_pflug_statistic(term1, term2, t, x, g, eta):

    term1 = 1.0/t * ((t - 1) * term1 + eta * np.dot(x,g))
    term2 = 1.0/t * ((t - 1) * term2 + (eta**2/2) * np.dot(g, g))

    return term1, term2

def compute_pflug_statistic2(term1, term2, t, x, g, eta):

    term1 = 1.0/t * ((t - 1) * term1 + np.dot(x,g))
    term2 = 1.0/t * ((t - 1) * term2 + (eta/2) * np.dot(g, g))

    return term1, term2

def compute_batch_mean_variance_estimator(delta_list):
    N = len(delta_list)
    p = int(np.sqrt(N))
    q = int(np.sqrt(N))
    mu_N = np.mean(delta_list)
    sigma_square = 0
    for j in range(p):
        mu_j = np.mean(delta_list[j*q:(j+1)*q])
        sigma_square += (mu_j - mu_N)**2
    return q / (p-1) * sigma_square

def pad_score_list(data, max_epoch, padding = "last"):
    data_copy = copy.deepcopy(data)
    list_len = len(data_copy)
    if list_len == max_epoch:
        return data
    else:
        if data[list_len - 1]['grad_norm'] < 1e-13:
            data_copy[list_len - 1]['grad_norm'] = 1e-13
        if data[list_len - 1]['grad_norm'] > 1e-7 or np.isnan(data[list_len - 1]['grad_norm']): #this means you diverged
            padding = "last"
        if padding == "last":
            padding = data_copy[list_len - 1]['grad_norm']
        #get average n_grad_evals to pad the dict with
        #don't count epoch 0 because counting is different
        if list_len > 1:
            avg_increment_n_grad_evals = (data[list_len-1]['n_grad_evals'] - data[0]['n_grad_evals'])/ (list_len - 1)
        else:
            #if you are in this case, then what is going on is that you diverged right away
            avg_increment_n_grad_evals = 2*data[0]['n_grad_evals']
        n_grad_evals_end = data[list_len - 1]['n_grad_evals']
        if "n_grad_evals_normalized" in data[list_len - 1].keys():
            n_grad_evals_normalized_end = data[list_len - 1]['n_grad_evals_normalized']
            number_data = n_grad_evals_end / n_grad_evals_normalized_end
        nb_increments_to_add = 1
        for i in range(list_len+1, max_epoch):
            if "n_grad_evals_normalized" in data[list_len - 1].keys():
                dict_to_append = {'epoch': i,
                              'n_grad_evals': n_grad_evals_end + nb_increments_to_add * avg_increment_n_grad_evals,
                              'n_grad_evals_normalized': (n_grad_evals_end + nb_increments_to_add * avg_increment_n_grad_evals)/number_data,
                              'grad_norm':padding}
                nb_increments_to_add += 1
                data_copy.append(dict_to_append)
            else:
                dict_to_append = {'epoch': i,
                              'n_grad_evals': n_grad_evals_end + nb_increments_to_add * avg_increment_n_grad_evals,
                              'grad_norm':padding}
                nb_increments_to_add += 1
                data_copy.append(dict_to_append)
        return data_copy

def pad_all(save_dir, target_dir, max_epoch=50, padding="last"):
    #os.makedirs('target_dir')
    for subdir, dirs, files in os.walk(save_dir):
        if not os.path.exists(subdir.replace(save_dir, target_dir)):
            os.makedirs(subdir.replace(save_dir, target_dir))
        for file in files:
            if file == "score_list.pkl":
                with open(os.path.join(subdir, file), 'rb') as f:
                    data = pickle.load(f)
                    new_data = pad_score_list(data, max_epoch=max_epoch, padding=padding)
                    target_file = os.path.join(subdir.replace(save_dir, target_dir), file)
                    with open(target_file, "wb") as fout:
                        pickle.dump(new_data, fout)
            else:
                copyfile(os.path.join(subdir, file), os.path.join(subdir.replace(save_dir, target_dir), file))
                
    