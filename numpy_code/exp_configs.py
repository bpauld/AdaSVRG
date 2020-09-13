from haven import haven_utils as hu
import itertools 

EXP_GROUPS = {}


step_sizes = [1e-5]
opt_list = [{'name':'svrg',
            "r":0.,
            "adaptive_termination": 0,
            "linesearch_option":0,
            "max_sgd_warmup_epochs":0},

            # {'name':'svrg_bb',  
            # 'init_step_size':1e-3,
            # "m":0.,
            # "adaptive_termination": False},

            # {'name':'svrg_ada',             
            # 'init_step_size':1e-3,
            # "m":0.,
            # "adaptive_termination": False,
            # "linesearch_option" : 0,
            # "max_sgd_warmup_epochs" : 0
            # }
           ]
     
EXP_GROUPS['exp1'] = hu.cartesian_exp_group({"dataset":["synthetic"],
                                            "loss_func": ["logistic_loss"],
                                            "opt": opt_list,
                                            "regularization_factor":1e-4,
                                            'margin':[1e-6],
                                            "false_ratio" : [0.25],
                                            "n_samples": [1000],
                                            "d": 3,
                                            "batch_size":[1],
                                            "max_epoch":[10],
                                            "runs":[0]})


#Setup for experiment 2, comparing svrg, svrg_bb, and svrg_ada with constant stepsize, line search, and bb stepsize.
losses = ["logistic_loss", "squared_loss", "squared_hinge_loss"]
stepsizes = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
svrg_list = []
for eta in stepsizes:
    svrg_list += {'name':'svrg',
                "r":0.,
                "adaptive_termination": 0,
                "init_step_size" : eta},

svrg_bb_list = []
for eta in stepsizes:
    svrg_bb_list += {'name':'svrg_bb',
                    "r":0.,
                    "init_step_size" : eta,
                    "adaptive_termination": 0},

svrg_ada_list = []
for ls in [0,1,2]:
    for reset in [True, False]:
        for eta in stepsizes:
            svrg_ada_list += {'name':'svrg_ada',
                            "r":0.,
                            "init_step_size" : eta,                
                            "adaptive_termination": 0,
                            "linesearch_option": ls,
                            "reset":  reset},  


svrg_cb_list = []
for reset in [False, True]:    
    svrg_cb_list += {'name':'svrg_cb',
                    "r":0.,              
                    "adaptive_termination": 0,                        
                    "reset":  reset},                                              

opt_list2 = svrg_list + svrg_bb_lis + svrg_ada_list + svrg_list

EXP_GROUPS['exp2'] = hu.cartesian_exp_group({"dataset":["synthetic"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1e-4,
                                            'margin':[1e-6],
                                            "false_ratio" : [0.25],
                                            "n_samples": [10000],
                                            "d": [20],
                                            "batch_size":[1],
                                            "max_epoch":[100],
                                            "runs":[0,1,2,3,4]})



#Setup for experiment 3, with dataset "mushrooms"
EXP_GROUPS['exp3'] = hu.cartesian_exp_group({"dataset":["mushrooms"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1/8000,
                                            "batch_size":[1],
                                            "max_epoch":[100],
                                            "runs":[0,1,2,3,4]})

#Setup for experiment 4, with dataset "ijcnn"
EXP_GROUPS['exp4'] = hu.cartesian_exp_group({"dataset":["ijcnn"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1/35000,
                                            "batch_size":[1],
                                            "max_epoch":[100],
                                            "runs":[0,1,2,3,4]})


#Setup for experiment 5, with dataset "a1a"
EXP_GROUPS['exp5'] = hu.cartesian_exp_group({"dataset":["a1a"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1/1600,
                                            "batch_size":[1],
                                            "max_epoch":[100],
                                            "runs":[0,1,2,3,4]})


# ---------  Edited by Sharan ---------------- #
runs = [0]
losses = ["logistic_loss"] #, "squared_loss", "squared_hinge_loss"]
stepsizes = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

svrg_list = []
for eta in stepsizes:
    svrg_list += {'name':'svrg',
                "r":0.,
                "adaptive_termination": 0,
                "init_step_size" : eta},

svrg_bb_list = []
for eta in stepsizes:
    svrg_bb_list += {'name':'svrg_bb',
                    "r":0.,
                    "init_step_size" : eta,
                    "adaptive_termination": 0},

svrg_ada_list = []
for reset in [True, False]:
    for eta in [100]:
        svrg_ada_list += {'name':'svrg_ada',
                        "r":0.,
                        "init_step_size" : eta,                
                        "adaptive_termination": 0,
                        "linesearch_option": 2,
                        "reset":  reset},  


svrg_cb_list = []
for reset in [False, True]:    
    svrg_cb_list += {'name':'svrg_cb',
                    "r":0.,              
                    "adaptive_termination": 0,                        
                    "reset":  reset},                                              

opt_list2 =  svrg_cb_list  +svrg_list +  svrg_bb_list  + svrg_ada_list 
            

EXP_GROUPS['sharan_test'] = hu.cartesian_exp_group({"dataset":["synthetic"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1e-4,                                            
                                            'margin':[1e-6],
                                            "false_ratio" : [0.25],
                                            "n_samples": [10000],
                                            "d": [20],
                                            "batch_size":[1],
                                            "max_epoch":[100],
                                            "runs":runs})