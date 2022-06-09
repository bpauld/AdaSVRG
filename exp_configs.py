from haven import haven_utils as hu
import itertools 
import numpy as np

def get_benchmark(benchmark,
                  opt_list,
                  batch_size = 1,
                  runs = [0,1,2,3,4],
                  max_epoch=[50],
                  losses=["logistic_loss", "squared_loss", "squared_hinge_loss"]
                 ):
    
    if benchmark in ["mushrooms", "mushrooms_diagonal"]:
        return {"dataset":["mushrooms"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./8000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "ijcnn":
        return {"dataset":["ijcnn"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./35000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark in ["a1a", "a1a_diagonal"]:
        return {"dataset":["a1a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./1600,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "a2a":
        return {"dataset":["a2a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./2300,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark in ["w8a", "w8a_diagonal"]:
        return {"dataset":["w8a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./50000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "covtype":
        return {"dataset":["covtype"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./500000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "phishing":
        return {"dataset":["phishing"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1e-4,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "rcv1":
        return {"dataset":["rcv1"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./20000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "synthetic_interpolation":
        return {"dataset":["synthetic"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":0.,
                "margin":[0.1],
                "false_ratio" : [0, 0.1, 0.2],
                "n_samples": [10000],
                "d": [200],
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    else:
        print("Benchmark unknown")
        return
    

EXP_GROUPS = {}
MAX_EPOCH = 50
RUNS = [0, 1, 2, 3, 4]
benchmarks_list = ["mushrooms", "ijcnn", "a1a", "a2a", "w8a", "rcv1", "covtype", "phishing"]
benchmarks_diagonal_list = ["a1a_diagonal", "mushrooms_diagonal", "w8a_diagonal"]
benchmarks_interpolation_list = ["synthetic_interpolation"]

for benchmark in benchmarks_list + benchmarks_diagonal_list + benchmarks_interpolation_list:
    EXP_GROUPS["exp_%s" % benchmark] = []
    
    
#=== Setting up scalar vs diagonal experiments ===

for batch_size in [64]:
    opt_list = []
    for variant in ["diagonal", "scalar"]:
        opt_list += [{'name':'AdaSVRG_General',
                 'r':1/batch_size,
                 'init_step_size':None,
                 'linesearch_option':1,
                 'adaptive_termination':0,
                 'variant':variant}]
        
    for benchmark in benchmarks_diagonal_list:
        EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, opt_list, batch_size=batch_size, max_epoch=[50], runs=[0, 1, 2, 3, 4], losses=['logistic_loss']))

#=== Setting up main experiments ===

for batch_size in [1, 8, 64, 128]:
    opt_list = []
    
    # Baseline optimizers
    for eta in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
        
        opt_list += [{'name':'SVRG_BB',
                  'r':1/batch_size,
                  'init_step_size':eta}]
        
        opt_list += [{'name':'SVRG',
                  'r':1/batch_size,
                  'adaptive_termination':0,
                  'init_step_size':eta}]
        
        opt_list += [{'name':'SVRG_Loopless',
                  'r':1/batch_size,
                  'init_step_size':eta}]

        opt_list += [{'name':'SARAH',
                  'r':1/batch_size,
                  'init_step_size':eta}]
            
    #AdaSVRG without adaptive termination
    opt_list += [{'name':'AdaSVRG',
                 'r':1/batch_size,
                 'init_step_size':None,
                 'linesearch_option':1,
                 'adaptive_termination':0}]

    #AdaSVRG with adaptive termination
    opt_list += [{'name':'AdaSVRG',
                 'r':10/batch_size,
                 'init_step_size':None,
                 'linesearch_option':1,
                 'adaptive_termination':1,
                 'threshold_at':0.5}]
                    

                    
    for benchmark in benchmarks_list:
        EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, opt_list, batch_size=batch_size, max_epoch=[MAX_EPOCH], runs=RUNS, losses=['logistic_loss', 'squared_loss', 'huber_loss']))



#=== Setting up interpolation experiment ===
        
for batch_size in [1, 8, 64, 128]:
    opt_list = []
    
    # Hybrid method
    opt_list +=[{'name':'AdaGrad_AdaSVRG',      
                 "init_step_size": 1000,
                 "r":1/batch_size,
                 "max_epoch_sgd":MAX_EPOCH, 
                 "adaptive_termination":1,
                 "threshold_at": 0.5,
                 "linesearch_option_sgd_ada":1}]
    
    # Find optimal manual switching
    for max_epoch_sgd in range(1, MAX_EPOCH):
        opt_list +=[{'name':'AdaGrad_AdaSVRG',      
                     "init_step_size": 1000,
                     "r":1/batch_size,
                     "max_epoch_sgd":max_epoch_sgd,
                     "adaptive_termination":0,
                     "linesearch_option_sgd_ada":1}]
    
    # Adagrad with linesearch
    opt_list +=[{'name':'AdaGrad',      
                 "init_step_size": 1000,
                 "r":1/batch_size,
                 "linesearch_option":1,
                 "beta":0.7,
                 "c":0.5}]
    
    # AdaSVRG without adaptive termination
    opt_list += [{'name':'AdaSVRG',
                 'r':1/batch_size,
                 'init_step_size':None,
                 'linesearch_option':1,
                 'adaptive_termination':0}]
    
    # SVRG
    for eta in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]:
        opt_list += [{'name':'SVRG',
                  'r':1/batch_size,
                  'init_step_size':eta}]

    
    for benchmark in benchmarks_interpolation_list:
        EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, opt_list=opt_list, batch_size=batch_size, runs=RUNS, losses=["squared_hinge_loss", "logistic_loss"], max_epoch=[MAX_EPOCH])) 

        
