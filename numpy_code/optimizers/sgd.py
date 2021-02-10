from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time




def SGD(score_list, closure, batch_size, D, labels, max_epoch=100, init_step_size=None,
         x0=None, verbose=True, D_test=None, labels_test=None):
    """
        SGD with fixed step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]
 
    m = int(n/batch_size)

    if x0 is None:
        x = np.zeros(d)
        x0 = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0

    step_size = init_step_size     

    for k in range(max_epoch):        
        t_start = time.time()

        loss, full_grad = closure(x, D, labels)
    

        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % step_size
            output += ', Num gradient evaluations/n: %f' % (num_grad_evals / n) 
            print(output) 

        

        score_dict = {"epoch": k}
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["n_grad_evals_normalized"] = num_grad_evals / n
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x, D, labels)
        if D_test is not None:
            test_loss = closure(x, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
        score_list += [score_dict]
        
        if np.linalg.norm(full_grad) <= 1e-12:
            break
        if np.linalg.norm(full_grad) > 1e10:
            break
        if np.isnan(full_grad).any():
            break
                   
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]

            # compute the loss, gradients
            loss, x_grad = closure(x, Di, labels_i)        
            gk  = x_grad
            num_grad_evals = num_grad_evals + batch_size    

            x -= step_size * gk

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch    

    return score_list, x, num_grad_evals, k