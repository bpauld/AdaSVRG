from haven import haven_utils as hu
import itertools 

EXP_GROUPS = {}

# ------------ for debugging ----------------------- #
# losses = ["squared_loss", "logistic_loss", "squared_hinge_loss"]
# opt_list = []
# opt_list += [{'name':'svrg_ada_at',
#                 "r":5,
#                 "init_step_size" : 1,                     
#                 "linesearch_option": 4,
#                 "adaptive_termination": True, 
#                 "reset":  True}]     
# EXP_GROUPS['exp1'] = hu.cartesian_exp_group({"dataset":["synthetic"],
#                                             "loss_func": losses,
#                                             "opt": opt_list,
#                                             "regularization_factor":1e-4,
#                                             'margin':[1e-6],
#                                             "false_ratio" : [0.25],
#                                             "n_samples": [10000],
#                                             "d": [20],
#                                             "batch_size":[1],
#                                             "max_epoch":[100],
                                            # "runs":[0]})
# ------------ for debugging ----------------------- #                                            

#Setup for comparing svrg, svrg_bb, and svrg_ada 
losses = ["logistic_loss", "squared_hinge_loss", "squared_loss"]
stepsizes = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
runs = [0,1,2,3,4]

# optimizers
svrg_list = []
for eta in stepsizes:
    svrg_list += [{'name':'svrg',
                "r":0.,
                "adaptive_termination": 0,
                "init_step_size" : eta}]

svrg_bb_list = []
for eta in stepsizes:
    svrg_bb_list += [{'name':'svrg_bb',
                    "r":0.,
                    "init_step_size" : eta,
                    "adaptive_termination": 0}]

svrg_ada_list = []
for ls in [1, 3]:
    for reset in [True]:
        for eta in [1]:
            svrg_ada_list += [{'name':'svrg_ada',
                            "r":0,
                            "init_step_size" : eta,                            
                            "linesearch_option": ls,
                            "reset":  reset}]


svrg_cb_list = []
for reset in [True]:    
    svrg_cb_list += [{'name':'svrg_cb',
                    "r":0.,              
                    "adaptive_termination": 0,                        
                    "reset":  reset}]

sarah_list = []
for eta in stepsizes:
    sarah_list += [{'name':'sarah',
                "r":0.,                
                "init_step_size" : eta}]

# final list of optimizers
opt_list2 = svrg_ada_list + svrg_list + sarah_list + svrg_bb_list
# print(opt_list2)

# for debugging
# opt_list2 = svrg_bb_list + svrg_ada_list
# opt_list2 = svrg_ada_list 

#Setup for experiment 2, with synthetic dataset 
EXP_GROUPS['exp2'] = hu.cartesian_exp_group({"dataset":["synthetic"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1e-4,
                                            'margin':[1e-6],
                                            "false_ratio" : [0.25],
                                            "n_samples": [10000],
                                            "d": [20],
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})

#Setup for experiment 3, with dataset "mushrooms"
EXP_GROUPS['exp3'] = hu.cartesian_exp_group({"dataset":["mushrooms"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1./8000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})                                           

#Setup for experiment 4, with dataset "ijcnn"
EXP_GROUPS['exp4'] = hu.cartesian_exp_group({"dataset":["ijcnn"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1./35000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})

#Setup for experiment 5, with dataset "a1a"
EXP_GROUPS['exp5'] = hu.cartesian_exp_group({"dataset":["a1a"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1./1600,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})

#Setup for experiment 6, with dataset "a2a"
EXP_GROUPS['exp6'] = hu.cartesian_exp_group({"dataset":["a2a"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1./2300,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})   

#Setup for experiment 7, with dataset "w8a"
EXP_GROUPS['exp7'] = hu.cartesian_exp_group({"dataset":["w8a"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1./50000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})      

#Setup for experiment 8, with dataset "rcv1"
EXP_GROUPS['exp8'] = hu.cartesian_exp_group({"dataset":["rcv1"],
                                            "loss_func": losses,
                                            "opt": opt_list2,
                                            "regularization_factor":1./20000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})

#================================ Setup for experiments on adaptive termination for svrg ========================================
opt_list9 = []
for at in [1,2]:
    opt_list9 += [{'name':'svrg',
                "r":10,
                "adaptive_termination": at,
                "init_step_size" : 1e-3}
EXP_GROUPS['exp9'] = hu.cartesian_exp_group({"dataset":["synthetic"],
                                            "loss_func": losses,
                                            "opt": opt_list9,
                                            "regularization_factor":1e-4,
                                            'margin':[1e-6],
                                            "false_ratio" : [0.25],
                                            "n_samples": [10000],
                                            "d": [20],
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})
                                  
opt_list10 = []
for at in [1,2]:
    opt_list10 += [{'name':'svrg',
                "r":10,
                "adaptive_termination": at,
                "init_step_size" : 1e-2}
EXP_GROUPS['exp10'] = hu.cartesian_exp_group({"dataset":["mushrooms"],
                                            "loss_func": losses,
                                            "opt": opt_list10,
                                            "regularization_factor":1./8000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})
                   
opt_list11 = []
for at in [1,2]:
    opt_list11 += [{'name':'svrg',
                "r":10,
                "adaptive_termination": at,
                "init_step_size" : 1e-1}
EXP_GROUPS['exp11'] = hu.cartesian_exp_group({"dataset":["ijcnn"],
                                            "loss_func": losses,
                                            "opt": opt_list11,
                                            "regularization_factor":1./35000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})
                   
opt_list12 = []
for at in [1,2]:
    opt_list12 += [{'name':'svrg',
                "r":10,
                "adaptive_termination": at,
                "init_step_size" : 1e-2}
EXP_GROUPS['exp12'] = hu.cartesian_exp_group({"dataset":["a1a"],
                                            "loss_func": losses,
                                            "opt": opt_list12,
                                            "regularization_factor":1./1600,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})
                   
#========================== Setup for experiments with svrg-ada with linesearch option 5 ====================================
svrg_ada_ls5_list = [{'name':'svrg_ada',
                "r":0,
                "adaptive_termination": False,
                "linesearch_option":5,
                "reset":True,
                "init_step_size" : 1}]
                   
EXP_GROUPS['exp13'] = hu.cartesian_exp_group({"dataset":["synthetic"],
                                            "loss_func": losses,
                                            "opt": svrg_ada_ls5_list,
                                            "regularization_factor":1e-4,
                                            'margin':[1e-6],
                                            "false_ratio" : [0.25],
                                            "n_samples": [10000],
                                            "d": [20],
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})

EXP_GROUPS['exp14'] = hu.cartesian_exp_group({"dataset":["mushrooms"],
                                            "loss_func": losses,
                                            "opt": svrg_ada_ls5_list,
                                            "regularization_factor":1./8000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})                                           

EXP_GROUPS['exp15'] = hu.cartesian_exp_group({"dataset":["ijcnn"],
                                            "loss_func": losses,
                                            "opt": svrg_ada_ls5_list,
                                            "regularization_factor":1./35000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})

EXP_GROUPS['exp16'] = hu.cartesian_exp_group({"dataset":["a1a"],
                                            "loss_func": losses,
                                            "opt": svrg_ada_ls5_list,
                                            "regularization_factor":1./1600,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})
                   
#=================================== Setup for experiments with svrg-ada with adaptive termination ===============================
svrg_ada_at_list = [{'name':'svrg_ada',
                "r":10,
                "adaptive_termination": True,
                "linesearch_option":5,
                "reset":True,
                "init_step_size" : 1}]
                   
EXP_GROUPS['exp17'] = hu.cartesian_exp_group({"dataset":["synthetic"],
                                            "loss_func": losses,
                                            "opt": svrg_ada_at_list,
                                            "regularization_factor":1e-4,
                                            'margin':[1e-6],
                                            "false_ratio" : [0.25],
                                            "n_samples": [10000],
                                            "d": [20],
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})

EXP_GROUPS['exp18'] = hu.cartesian_exp_group({"dataset":["mushrooms"],
                                            "loss_func": losses,
                                            "opt": svrg_ada_at_list,
                                            "regularization_factor":1./8000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})                                           

EXP_GROUPS['exp19'] = hu.cartesian_exp_group({"dataset":["ijcnn"],
                                            "loss_func": losses,
                                            "opt": svrg_ada_at_list,
                                            "regularization_factor":1./35000,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})

EXP_GROUPS['exp20'] = hu.cartesian_exp_group({"dataset":["a1a"],
                                            "loss_func": losses,
                                            "opt": svrg_ada_at_list,
                                            "regularization_factor":1./1600,
                                            "batch_size":[1],
                                            "max_epoch":[50],
                                            "runs":runs})