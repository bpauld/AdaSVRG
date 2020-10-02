from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time
from optimizers.sls import *
from optimizers.svrg_ada import *

def sls_svrg_ada(score_list, closure, batch_size, D, labels, 
            max_epoch=100, r=0, x0=None, verbose=True,            
            adaptive_termination = 0,
            D_test=None, labels_test=None):                    

    score_list_sls, x_sls, sls_grad_evals, sls_epochs = sls(score_list, closure = closure, batch_size=batch_size,
                max_epoch=max_epoch,
                init_step_size=None, 
                D = D, labels = labels, 						
                adaptive_termination = adaptive_termination,  
                D_test=D_test, labels_test=labels_test)

    print('SLS grad evals = ', sls_grad_evals )
    print('SLS epochs = ', sls_epochs )
    len_first_list = len(score_list_sls)

    score_list = svrg_ada(score_list, x0 = x_sls, closure=closure,batch_size=batch_size,
                    max_epoch=max_epoch - sls_epochs,                
                    D=D, labels=labels, 
                    r = r, init_step_size =  1., 
                    linesearch_option = 5,
                    adaptive_termination = False,
                    reset = True, D_test=D_test, labels_test=labels_test)

    for i in range(len_first_list, len(score_list)):
        score_list[i]["n_grad_evals"]  += sls_grad_evals
        score_list[i]["epoch"] += len_first_list
    return score_list