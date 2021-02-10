from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time
from optimizers.adagrad import *
from optimizers.adasvrg import *

def AdaGrad_AdaSVRG(score_list, closure, batch_size, D, labels, 
            max_epoch=100, r=0, init_step_size=1, threshold_at=0.5, linesearch_option_sgd_ada=0,
                     max_epoch_sgd=50,
                     adaptive_termination=0, x0=None, verbose=True,
            D_test=None, labels_test=None):
    
    (n,d) = D.shape
    score_list_sls, x_sls, sls_grad_evals, sls_epochs = AdaGrad(score_list, closure = closure, batch_size=batch_size,
                max_epoch=max_epoch_sgd,
                init_step_size=init_step_size, 
                linesearch_option = linesearch_option_sgd_ada,
                D = D, labels = labels, 						
                adaptive_termination = adaptive_termination, threshold_at=threshold_at,
                D_test=D_test, labels_test=labels_test)

    print('SGD-ADA grad evals = ', sls_grad_evals )
    print('SGD-ADA epochs = ', sls_epochs + 1 )
    end_sls_list = len(score_list_sls)
    
    score_list = AdaSVRG(score_list, x0 = x_sls, closure=closure,batch_size=batch_size,
                    max_epoch=max_epoch - sls_epochs,                
                    D=D, labels=labels, 
                    r = r, init_step_size =  1., 
                    linesearch_option = 1,
                    adaptive_termination = 0,
                    D_test=D_test, labels_test=labels_test)
    
    for i in range(end_sls_list, len(score_list)):
        score_list[i]["n_grad_evals"]  += sls_grad_evals
        score_list[i]["n_grad_evals_normalized"] += sls_grad_evals/n
        score_list[i]["epoch"] += sls_epochs + 1
    return score_list