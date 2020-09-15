from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time

def sarah(score_list, closure, batch_size, D, labels, init_step_size, max_epoch=100, 
         r=0, x0=None, verbose=True, D_test=None, labels_test=None):
    """
        SARAH with fixed step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]

    step_size = init_step_size
    if r <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')
    else:
        m = int(r * n)
        print('Info: set m = ', r, ' n')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0
                    
    for k in range(max_epoch):

        if num_grad_evals >= 2 * n * max_epoch:
            # exceeds the number of standard SVRG gradient evaluations (only for batch-size = 1)
            print('End of budget for gradient evaluations')
            break
        t_start = time.time()

        loss, full_grad = closure(x, D, labels)

        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % step_size
            output += ', Num gradient evaluations: %d' % num_grad_evals
            print(output) 

        if np.linalg.norm(full_grad) <= 1e-12:
            break

        num_grad_evals = num_grad_evals + n

        score_dict = {"epoch": k}
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x, D, labels)
        score_dict["train_loss_log"] = np.log(loss)
        score_dict["grad_norm_log"] = np.log(score_dict["grad_norm"])
        if D_test is not None:
            test_loss = closure(x, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
            score_dict["test_loss_log"] = np.log(test_loss)

        score_list += [score_dict]

        v  = full_grad
        x_prev = np.copy(x) # store the previoous iterate

        x -= step_size * v # do the update using the full gradient 
                            # before getting into the loop        
        
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]

            # compute the gradients at the current/previous points
            x_grad = closure(x, Di, labels_i)[1]
            x_grad_prev = closure(x_prev, Di, labels_i)[1]
            num_grad_evals = num_grad_evals + batch_size

            v = x_grad - x_grad_prev + v            
            x_prev = np.copy(x)
            x -= step_size * v

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch    
    return score_list