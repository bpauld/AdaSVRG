import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import math
import itertools
import os, sys
import pylab as plt
import exp_configs
import time
import numpy as np
import torch.nn as nn


from src import models
from src import datasets
from src import utils as ut
from src import metrics
from src.optimizers import svrg

import argparse

from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import default_collate

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
import shutil

import pprint

def get_svrg_step_size(exp_dict):

    # learning rates selected by cross-validation.
    lr_dict = {
        "logistic_loss": {          
                    "rcv1"      : 500,
                    "mushrooms" : 500,
                    "ijcnn"     : 500,
                    "w8a"       : 0.0025,
                    'syn-0.01'  : 1.5,
                    'syn-0.05'  : 0.1,
                    'syn-0.1'   : 0.025,
                    'syn-0.5'   : 0.0025,
                    'syn-1.0'   : 0.001,},
    "squared_hinge_loss" : {            
                    'mushrooms' : 150., 
                    'rcv1'      : 3.25,
                    "ijcnn"     : 2.75,
                    "w8a"       : 0.00001,
                    'syn-0.01'  : 1.25,
                    'syn-0.05'  : 0.025,
                    'syn-0.1'   : 0.0025,
                    'syn-0.5'   : 0.001,
                    'syn-1.0'   : 0.001,}
    }

    if exp_dict["loss_func"] in lr_dict:
        ds_name = exp_dict["dataset"]
        if ds_name == "synthetic":
            ds_name = "syn-%s" % str(exp_dict["margin"])
        lr = lr_dict[exp_dict["loss_func"]][ds_name]
    else:
        lr = 0.1

    return lr            

def get_svrg_optimizer(model, loss_function, train_loader, lr):
    n = len(train_loader.dataset)
    full_grad_closure = svrg.full_loss_closure_factory(train_loader,
                                                       loss_function,
                                                       grad=True)
    opt = svrg.SVRG(model,
                    train_loader.batch_size,
                    lr,
                    n,
                    full_grad_closure,
                    m=len(train_loader))

    return opt


def trainval_svrg(exp_dict, savedir_base, reset, metrics_flag=True, datadir=None, cuda=False):

    '''
        SVRG-specific training and validation loop.
    '''
    pprint.pprint(exp_dict)
     # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    print(pprint.pprint(exp_dict))
    print('Experiment saved in %s' % savedir)


    # set seed
    # ==================
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
    else:
        device = 'cpu'

    print('Running on device: %s' % device)

    # Load Train Dataset
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    train_loader = DataLoader(train_set,
                              drop_last=False,
                              shuffle=True,
                              batch_size=exp_dict["batch_size"])

    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict)

    # Load model
    model = models.get_model(exp_dict["model"],
                             train_set=train_set).to(device=device)

    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # lookup the learning rate
    lr = get_svrg_step_size(exp_dict)

    # Load Optimizer
    opt = get_svrg_optimizer(model, loss_function, train_loader=train_loader, lr=lr)

    # Resume from last saved state_dict
    if (not os.path.exists(savedir + "/run_dict.pkl") or
        not os.path.exists(savedir + "/score_list.pkl")):
        ut.save_pkl(savedir + "/run_dict.pkl", {"running":1})
        score_list = []
        s_epoch = 0
    else:
        score_list = ut.load_pkl(savedir + "/score_list.pkl")
        model.load_state_dict(torch.load(savedir + "/model_state_dict.pth"))
        opt.load_state_dict(torch.load(savedir + "/opt_state_dict.pth"))
        s_epoch = score_list[-1]["epoch"] + 1

    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}

        if metrics_flag:
            # 1. Compute train loss over train set
            score_dict["train_loss"] = metrics.compute_metric_on_dataset(model, train_set,
                                                metric_name=exp_dict["loss_func"])

            # 2. Compute val acc over val set
            score_dict["val_acc"] = metrics.compute_metric_on_dataset(model, val_set,
                                                        metric_name=exp_dict["acc_func"])

        # 3. Train over train loader
        model.train()
        print("%d - Training model with %s..." % (epoch, exp_dict["loss_func"]))

        s_time = time.time()
        for batch in tqdm.tqdm(train_loader):
            images, labels = batch['images'].to(device=device), batch['labels'].to(device=device)

            opt.zero_grad()
            closure = lambda svrg_model : loss_function(svrg_model, images, labels,
                                                                    backwards=True)
            opt.step(closure)

        e_time = time.time()

        # Record step size and batch size
        score_dict["step_size"] = opt.state["step_size"]
        score_dict["batch_size"] =  train_loader.batch_size
        score_dict["train_epoch_time"] = e_time - s_time

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        ut.save_pkl(savedir + "/score_list.pkl", score_list)
        ut.torch_save(savedir + "/model_state_dict.pth", model.state_dict())
        ut.torch_save(savedir + "/opt_state_dict.pth", opt.state_dict())
        print("Saved: %s" % savedir)

    return score_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)
    parser.add_argument('-v', '--view_results', default=None)
    parser.add_argument('-c', '--cuda', type=int, default=False)

    args = parser.parse_args()


    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    ####
    # Run experiments or View them
    # ----------------------------
    if args.view_results:
        # view results
        table = hr.get_score_df(exp_list, args.savedir_base, verbose=False, flatten_columns=False)
        print(table[['dataset', 'model', 'opt', 'train_loss', 'val_acc']])
        print('Results are in variable table')
        import ipdb; ipdb.set_trace()
        

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval_svrg(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    reset=args.reset,
                    datadir=args.datadir,
                    cuda=args.cuda)