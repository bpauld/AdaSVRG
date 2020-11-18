from haven import haven_utils as hu
import itertools 

RUNS = [0]
LOSSES = ["logistic_loss"] #, "squared_loss", "squared_hinge_loss"]
def get_benchmark(benchmark,
                  opt_list,
                  batch_size = 1,
                  runs = [0,1,2,3,4],
                  max_epoch=[50],
                  losses=["logistic_loss", "squared_loss", "huber_loss"]):
    if benchmark == "synthetic":
        return {"dataset":["synthetic"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1e-4,
                'margin':[1e-6],
                "false_ratio" : [0.25],
                "n_samples": [10000],
                "d": [20],
                "batch_size":[batch_size],
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "mushrooms":
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
    
    elif benchmark == "a1a":
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
    
    elif benchmark == "w8a":
        return {"dataset":["w8a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./50000,
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
    
    elif benchmark == "syn_interpolation1":
        return {"dataset":["synthetic"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":0.,
                "margin":[0.1],
                "false_ratio" : [0.],
                "n_samples": [10000],
                "d": [20],
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "syn_interpolation2":
        return {"dataset":["synthetic"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":0.,
                "margin":[0.01],
                "false_ratio" : [0.],
                "n_samples": [10000],
                "d": [20],
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "syn_interpolation3":
        return {"dataset":["synthetic"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":0.,
                "margin":[0.001],
                "false_ratio" : [0.],
                "n_samples": [10000],
                "d": [20],
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    
    elif benchmark == "kernel_mushrooms":
        return {"dataset": ["mushrooms"],
               "is_kernelize":1,
               "loss_func":losses,
               "opt":opt_list,
               "regularization_factor":0.,
               "batch_size":batch_size,
               "max_epoch":max_epoch,
               "runs":runs}


EXP_GROUPS = {}
benchmarks_list = ["synthetic", "mushrooms", "ijcnn", "a1a", "a2a", "w8a", "rcv1"]
benchmarks_interpolation_list = ["syn_interpolation1", "syn_interpolation2", "syn_interpolation3", "kernel_mushrooms"]
for benchmark in benchmarks_list + benchmarks_interpolation_list:
    EXP_GROUPS["exp_%s" % benchmark] = []

#======================= Setting up basic optimizers ===========================
stepsizes = [1e-3, 1e-2, 1e-1]

#svrg optimizers
svrg_list = []
for eta in stepsizes:
    svrg_list += [{'name':'svrg',
                  'r':0.,
                  'adaptive_termination':0,
                  'init_step_size':eta}]

#svrg-bb optimizers
svrg_bb_list = []
for eta in stepsizes:
    svrg_bb_list += [{'name':'svrg_bb',
                    "r":0.,
                    "init_step_size" : eta,
                    "adaptive_termination": 0}]
#svrg-ada optimizers   
svrg_ada_list = [{'name':'svrg_ada',
                 'r':0.,
                 'init_step_size':1,
                 'linesearch_option':1,
                 'reset':True,
                 'adaptive_termination':False}]
svrg_ada_list += [{'name':'svrg_ada',
                 'r':0.,
                 'init_step_size':1,
                 'linesearch_option':1,
                 'reset':True,
                 'adaptive_termination':False,
                 'average_iterates':True}]

#hybrid sls svrg-ada optimizers
sls_svrg_ada_list = [{'name':'sls_svrg_ada',      
                "r" : 0., 
                "adaptive_termination": 2}] 

basic_opt_list = svrg_list + svrg_bb_list + svrg_ada_list + sls_svrg_ada_list
#basic_opt_list = svrg_list

#for benchmark in benchmarks_list:
 #   EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, basic_opt_list, batch_size=1)) 

    
    
#============================ Setting up interpolation experiments =======================================
sls_opt_list = []
sls_opt_list += [{'name':'sls',      
                "init_step_size":  None,          
                "adaptive_termination": 0}]
for eta in stepsizes:
    sls_opt_list +=[{'name':'sls',      
                "init_step_size":  eta,          
                "adaptive_termination": 0}]

interpolation_opt_list = sls_opt_list + svrg_list + sls_svrg_ada_list
# for benchmark in benchmarks_interpolation_list:
#     EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, interpolation_opt_list, batch_size=1))
    

    
    
#============================ Setting up optimizers with different batch sizes ========================
batch_sizes = [8, 16, 32, 64, 128, 256, 512]
for bs in batch_sizes:
    bs_opt_list = []
    for eta in stepsizes:
        bs_opt_list += [{'name':'svrg',
                "r":1/bs,
                "adaptive_termination": 0,
                "init_step_size" : eta}]      
        bs_opt_list += [{'name':'svrg_bb',
                "r":1/bs,
                "adaptive_termination": 0,
                "init_step_size" : eta}]
    bs_opt_list += [{'name':'svrg_ada',
                 "r":1/bs,
                 "init_step_size" : 1,                            
                 "linesearch_option": 1,
                 "reset":  True,
                 "adaptive_termination":False}]
    bs_opt_list += [{'name':'svrg_ada',
                 'r':1/bs,
                 'init_step_size':1,
                 'linesearch_option':1,
                 'reset':True,
                 'adaptive_termination':False,
                 'average_iterates':True}]
    # for benchmark in benchmarks_list:
    #     EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, bs_opt_list, batch_size=bs)) 
        

        
        
#=========================== Setting up svrg-ada with adaptive termination ==========================
intervals_list = [5, 10, 50, 100, 200]
thresholds_list = [0.95, 0.99, 1, 1.01, 1.1, 1.5]
svrg_ada_at_list = []
for interval in intervals_list:
    for threshold in thresholds_list:
        svrg_ada_at_list += [{'name':'svrg_ada',
                             'r':10,
                             'adaptive_termination':True,
                             'linesearch_option':1,
                             'reset':True,
                             'init_step_size':1,
                             'threshold_at': threshold,
                             'interval_size':interval}]

#for benchmark in benchmarks_list:
 #   EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, svrg_ada_at_list, batch_size=1))        


#=========================== Debugging svrg-ada with adaptive termination ==========================
intervals_list = [100]
thresholds_list = [1e-3]
svrg_ada_at_list = []
batch_size  = 1
for interval in intervals_list:
    for threshold in thresholds_list:
        svrg_ada_at_list += [{'name':'svrg_ada',
                             'r':1,
                             'adaptive_termination':1,
                             'linesearch_option':1,
                             'reset':True,
                             'init_step_size':1}]
        svrg_ada_at_list += [{'name':'svrg_ada',
                             'r':1,
                             'adaptive_termination':2,
                             'linesearch_option':1,
                             'reset':True,
                             'init_step_size':1}]
        svrg_ada_at_list += [{'name':'svrg_ada',
                             'r':10,
                             'adaptive_termination':True,
                             'linesearch_option':1,
                             'reset':True,
                             'init_step_size':1,
                             'threshold_at': threshold,
                             'interval_size':interval}]
                            
for benchmark in benchmarks_list:
    EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, svrg_ada_at_list, batch_size=batch_size))

    
#======================== testing kernels =====================================
batch_sizes = [32]
for bs in batch_sizes:
    bs_opt_list = []
    bs_opt_list += [{'name':'sls',      
                "init_step_size":  None,          
                "adaptive_termination": 0,
                "batch_size":bs}]
    bs_opt_list += [{'name':'svrg_ada',
                 'r':1/bs,
                 'init_step_size':1,
                 'linesearch_option':1,
                 'reset':True,
                 'adaptive_termination':False,
                 'average_iterates':True}]
    for eta in [1e-3, 1e-2, 1e-1, 1]:
        bs_opt_list += [{'name':'sls',      
                "init_step_size":  eta,          
                "adaptive_termination": 0,
                "batch_size":bs}]
    for benchmark in ["kernel_mushrooms"]:
        EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, bs_opt_list, batch_size=bs, runs=[0,1], max_epoch=20))
        
        
        
#================================== diagonal experiments ==================================================
svrg_ada_diag_list = []
for benchmark in benchmarks_list:
        EXP_GROUPS['exp_%s_diag' % benchmark] = []

for bs in [1]:
    svrg_ada_diag_list = []
    if True:    
        svrg_ada_diag_list += [{'name':'svrg_ada',
                 'r':1/bs,
                 'init_step_size':1,
                 'linesearch_option':14,
                 'reset':True,
                 'adaptive_termination':False}]
    for ls in [2, 14]:
        for eps in [1e-10]:
            for scale in [0, 1 , 2]:
                svrg_ada_diag_list += [{'name':'svrg_ada_diag',
                       'r':1/bs,
                 'init_step_size':1,
                 'linesearch_option':ls,
                 'reset':True,
                 'adaptive_termination':0,
                 'average_iterates':False,
                 'scaling_by_d':scale,
                  'epsilon':eps}]
        
    for benchmark in benchmarks_list:
        EXP_GROUPS['exp_%s_diag' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, svrg_ada_diag_list, batch_size=bs, max_epoch=[40], runs=[0, 1], losses = ['logistic_loss', 'squared_loss', 'huber_loss']))

