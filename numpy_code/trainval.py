from dependencies import *

from objectives import *
from datasets import *
from utils import *

from optimizers.svrg import *
from optimizers.svrg_bb import *
from optimizers.svrg_ada import *
from optimizers.svrg_cb import *
from optimizers.sarah import *
from optimizers.svrg_ada_at import *
from optimizers.svrg_ada_diag import *
from optimizers.sls import *
from optimizers.sls_svrg_ada import *

import argparse
import exp_configs

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
import shutil

from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_jobs as hj

def trainval(exp_dict, savedir_base, reset=False):

	# dataset options
	data_dir = './'

	# get experiment directory
	exp_id = hu.hash_dict(exp_dict)
	savedir = os.path.join(savedir_base, exp_id)

	if reset:
	    # delete and backup experiment
	    hc.delete_experiment(savedir, backup_flag=True)

	# create folder and save the experiment dictionary
	os.makedirs(savedir, exist_ok=True)
	hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
	pprint.pprint(exp_dict)
	print('Experiment saved in %s' % savedir)

	score_list_path = os.path.join(savedir, 'score_list.pkl')

	seed = 42 + exp_dict['runs']
	np.random.seed(seed)

	# default values	
	if "is_subsample" not in exp_dict.keys():
		is_subsample = 0
	else:
		is_subsample = exp_dict["is_subsample"]

	if "is_kernelize" not in exp_dict.keys():
		is_kernelize = 0
	else:
		is_kernelize = exp_dict["is_kernelize"]

	if "false_ratio" not in exp_dict.keys():		
		false_ratio = 0		
	else:
		false_ratio = exp_dict["false_ratio"]
	
    # load the dataset
	if exp_dict["dataset"] == "synthetic":				
	    n, d = exp_dict["n_samples"], exp_dict["d"]
	    false_ratio = exp_dict["false_ratio"]	   
	    margin = exp_dict["margin"]	    
	    X, y, X_test, y_test = data_load(data_dir, exp_dict["dataset"],n, d, margin, false_ratio)
	else:
		if is_subsample == 1:
			n = subsampled_n
		else:	
			n = 0

		if is_kernelize == 1:
			d = n
		else:
			d = 0

		X, y, X_test, y_test = data_load(data_dir, exp_dict["dataset"] , n, d, false_ratio, is_subsample=is_subsample, is_kernelize=is_kernelize)
		n = X.shape[0]
	#define the regularized losses we will use
	regularization_factor = exp_dict["regularization_factor"]
	if exp_dict["loss_func"] == "logistic_loss":
		closure = make_closure(logistic_loss, regularization_factor)
	elif exp_dict["loss_func"] == "squared_hinge_loss":
		closure = make_closure(squared_hinge_loss, regularization_factor)
	elif exp_dict["loss_func"] == "squared_loss":
		closure = make_closure(squared_loss, regularization_factor)
	else:
		print("Not a valid loss")

	# check if score list exists 
	if os.path.exists(score_list_path):
		# resume experiment
		score_list = hu.load_pkl(score_list_path)
		s_epoch = score_list[-1]['epoch'] + 1
	else:
		# restart experiment
		score_list = []
		s_epoch = 0

	print('Starting experiment at epoch %d' % (s_epoch))


	opt_dict = exp_dict["opt"]

	if opt_dict["name"] in ['svrg']:
	    init_step_size = opt_dict["init_step_size"]
	    r = opt_dict["r"]
	    adaptive_termination = opt_dict["adaptive_termination"]
	    if "interval_size" in opt_dict.keys():
		    interval_size = opt_dict["interval_size"]
	    else:
		    interval_size=10
	    if "threshold_at" in opt_dict.keys():
		    threshold_at = opt_dit["threshold_at"]
	    else:
		    threshold_at = 1

	    score_list = svrg(score_list, closure=closure,batch_size=exp_dict["batch_size"], 
						   max_epoch=exp_dict["max_epoch"],                                               
                           D=X, labels=y,
                           init_step_size=init_step_size, r=r, adaptive_termination= adaptive_termination, D_test=X_test, labels_test=y_test, interval_size=interval_size, threshold_at=threshold_at)


	elif opt_dict["name"] in ['svrg_bb']:		

		init_step_size = opt_dict["init_step_size"]
		r = opt_dict["r"]
		adaptive_termination = opt_dict["adaptive_termination"]
		score_list = svrg_bb(score_list, closure=closure,batch_size=exp_dict["batch_size"], 
						   max_epoch=exp_dict["max_epoch"],                                               
                           D=X, labels=y,
                           init_step_size=init_step_size, r=r, adaptive_termination= adaptive_termination, D_test=X_test, labels_test=y_test)

	elif opt_dict["name"] == 'svrg_ada':
		init_step_size = opt_dict["init_step_size"]
		r = opt_dict["r"]		
		linesearch_option = opt_dict["linesearch_option"]
		reset = opt_dict["reset"]
		adaptive_termination = opt_dict["adaptive_termination"]
		if "interval_size" in opt_dict.keys():
			interval_size = opt_dict["interval_size"]
		else:
			interval_size=10
		if "threshold_at" in opt_dict.keys():
			threshold_at = opt_dit["threshold_at"]
		else:
			threshold_at = 1
		if "average_iterates" in opt_dict.keys():
			average_iterates = opt_dict["average_iterates"]
		else:
			average_iterates = False

		score_list = svrg_ada(score_list, closure=closure,batch_size=exp_dict["batch_size"], 
						   max_epoch=exp_dict["max_epoch"],                                               
                           D=X, labels=y, 
                           r = r, init_step_size=init_step_size, 
                           linesearch_option = linesearch_option,
                           adaptive_termination = adaptive_termination,
                           reset = reset, D_test=X_test, labels_test=y_test, threshold_at=threshold_at, interval_size=interval_size, average_iterates=average_iterates)
        
        
	elif opt_dict["name"] == 'svrg_ada_diag':
		init_step_size = opt_dict["init_step_size"]
		r = opt_dict["r"]		
		linesearch_option = opt_dict["linesearch_option"]
		reset = opt_dict["reset"]
		adaptive_termination = opt_dict["adaptive_termination"]
		if "epsilon" in opt_dict.keys():
			epsilon = opt_dict["epsilon"]
		else:
			epsilon = 1e-8      

		score_list = svrg_ada_diag(score_list, closure=closure,batch_size=exp_dict["batch_size"], 
						   max_epoch=exp_dict["max_epoch"],                                               
                           D=X, labels=y, 
                           r = r, init_step_size=init_step_size, 
                           linesearch_option = linesearch_option,
                           adaptive_termination = adaptive_termination,
                           reset = reset, D_test=X_test, labels_test=y_test, epsilon=epsilon)

	elif opt_dict["name"] == 'svrg_cb':
		r = opt_dict["r"]
		adaptive_termination = opt_dict["adaptive_termination"]
		reset = opt_dict["reset"]

		score_list = svrg_cb(score_list, closure=closure,batch_size=exp_dict["batch_size"], 
						   max_epoch=exp_dict["max_epoch"],                                               
                           D=X, labels=y, 
                           r = r,                            
                           adaptive_termination= adaptive_termination,
                           reset = reset, D_test = X_test, labels_test=y_test)		


	elif opt_dict["name"] == 'sarah':
		r = opt_dict["r"]
		init_step_size = opt_dict["init_step_size"]
		score_list = sarah(score_list, closure=closure,batch_size=exp_dict["batch_size"], 
						   max_epoch=exp_dict["max_epoch"],                                               
                           D=X, labels=y, 
                           r = r,   init_step_size = init_step_size,                          
                           D_test = X_test, labels_test=y_test)	

	elif opt_dict["name"] == 'svrg_ada_at':
		init_step_size = opt_dict["init_step_size"]
		r = opt_dict["r"]		
		linesearch_option = opt_dict["linesearch_option"]
		adaptive_termination = opt_dict["adaptive_termination"]
		reset = opt_dict["reset"]

		score_list = svrg_ada_at(score_list, closure=closure,batch_size=exp_dict["batch_size"], 
						   max_epoch=exp_dict["max_epoch"],                                               
                           D=X, labels=y, 
                           r = r, init_step_size=init_step_size, 
                           linesearch_option = linesearch_option, 
						   adaptive_termination = adaptive_termination,                          
                           reset = reset, D_test=X_test, labels_test=y_test)	

	elif opt_dict["name"] == 'sls':		
		adaptive_termination = opt_dict["adaptive_termination"]		
		init_step_size = opt_dict["init_step_size"]	
		score_list = sls(score_list, closure = closure, batch_size=exp_dict["batch_size"], 
						max_epoch=exp_dict["max_epoch"], 
						init_step_size=init_step_size, 
						D = X, labels = y, 						
         				adaptive_termination = adaptive_termination,  
						D_test=X_test, labels_test=y_test)[0]

	elif opt_dict["name"] == 'sls_svrg_ada':		
		adaptive_termination = opt_dict["adaptive_termination"]			
		r = opt_dict["r"]		
		score_list = sls_svrg_ada(score_list, closure = closure, 
					batch_size=exp_dict["batch_size"], 
					max_epoch=exp_dict["max_epoch"], 
					r = r, adaptive_termination = adaptive_termination, 
					D = X, labels = y, 											
					D_test=X_test, labels_test=y_test)

	else:
		print('Method does not exist')
    
	save_pkl(score_list_path, score_list)  

	return score_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)
    parser.add_argument('-v', '--view_jupyter', default=None)
    parser.add_argument('-j', '--run_jobs', default=None)

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


    # Run experiments or View them
    # ----------------------------
    if args.run_jobs:
        # launch jobs
        from haven import haven_jobs as hj
        hj.run_exp_list_jobs(exp_list, 
                       savedir_base=args.savedir_base, 
                       workdir=os.path.dirname(os.path.realpath(__file__)))

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    reset=args.reset)