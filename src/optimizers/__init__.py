import numpy as np
import torch
from src.optimizers import sls, sps, svrg

def get_optimizer(opt, params, n_batches_per_epoch=None, n_train=None, lr=None,
                  train_loader=None, model=None, loss_function=None, exp_dict=None, batch_size=None):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    if isinstance(opt, dict):
        opt_name = opt["name"]
        opt_dict = opt
    else:
        opt_name = opt
        opt_dict = {}

    # ===============================================
    # our optimizers   
    n_batches_per_epoch = opt_dict.get("n_batches_per_epoch") or n_batches_per_epoch    
    

    if opt_name == "sgd_armijo":
        # if opt_dict.get("infer_c"):
        #     c = (1e-3) * np.sqrt(n_batches_per_epoch)
        if opt_dict['c'] == 'theory':
            c = (n_train - batch_size) / (2 * batch_size * (n_train - 1))
        else:
            c = opt_dict.get("c") or 0.1
        
        opt = sls.Sls(params,
                    c = c,
                    n_batches_per_epoch=n_batches_per_epoch,
                    init_step_size=opt_dict.get("init_step_size", 1),
                    line_search_fn=opt_dict.get("line_search_fn", "armijo"), 
                    gamma=opt_dict.get("gamma", 2.0),
                    reset_option=opt_dict.get("reset_option", 1),
                    eta_max=opt_dict.get("eta_max"))
    
    elif opt_name == 'sps':
        opt = sps.Sps(params, c=opt_dict["c"], 
                        n_batches_per_epoch=n_batches_per_epoch, 
                        adapt_flag=opt_dict.get('adapt_flag', 'basic'),
                        fstar_flag=opt_dict.get('fstar_flag'),
                        eta_max=opt_dict.get('eta_max'),
                        eps=opt_dict.get('eps', 0))

    elif opt_name == 'svrg':

        lr = 0.1
        n = len(train_loader.dataset)
        full_grad_closure = svrg.full_loss_closure_factory(train_loader,
                                                        loss_function,
                                                        grad=True)
        opt = svrg.SVRG(model,
                        train_loader.batch_size,
                        lr,
                        n,
                        full_grad_closure,
                        m=len(train_loader),
                        splr_flag=exp_dict['opt'].get('splr_flag'),
                        c=exp_dict['opt'].get('c'),
                        
                        )        

    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt
