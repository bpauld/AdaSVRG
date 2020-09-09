from haven import haven_utils as hu
import itertools 

EXP_GROUPS = {}

opt_list = [

            {'name':'svrg',             
            'init_step_size':1e-1,
            "r":0.,
            "adaptive_termination": 0},

            {'name':'svrg',             
            'init_step_size':1e-1,
            "r":2.,
            "adaptive_termination": 2},

			{'name':'svrg',             
            'init_step_size':1e-1,
            "r":2.,
            "adaptive_termination": 1},

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
                                            'margin':[1e-6],
                                            "false_ratio" : [0.25],
                                            "n_samples": [1000],
                                            "d": 20,
                                            "batch_size":[1],
                                            "max_epoch":[100],
                                            "runs":[0]})