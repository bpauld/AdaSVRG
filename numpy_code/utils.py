from dependencies import *

import hashlib 
import pickle
import json
import os
import itertools
import numpy as np
import tqdm

# =======================================
# haven
from haven import haven_jupyter as hj
from haven import haven_results as hr
# from haven import haven_dropbox as hd
from haven import haven_utils as hu

import os
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