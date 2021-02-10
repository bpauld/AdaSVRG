from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time


def SVRG(score_list, closure, batch_size, D, labels, init_step_size, max_epoch=100, 
         r=0, x0=None, verbose=True, D_test=None, labels_test=None):
    """
        SVRG with fixed step size for solving finite-sum problems
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
        t_start = time.time()


        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()
        num_grad_evals = num_grad_evals + n

        last_full_grad = full_grad
        last_x_tilde = x_tilde
        
        score_dict = {"epoch": k}
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["n_grad_evals_normalized"] = num_grad_evals / n
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x, D, labels)
        score_dict["train_loss_log"] = np.log(loss)
        score_dict["grad_norm_log"] = np.log(score_dict["grad_norm"])
        # score_dict["train_accuracy_log"] = np.log(score_dict["train_accuracy"])
        if D_test is not None:
            test_loss = closure(x, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
            score_dict["test_loss_log"] = np.log(test_loss)
            # score_dict["test_accuracy_log"] = np.log(score_dict["test_accuracy"])
        score_list += [score_dict]
        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % step_size
            output += ', Num gradient evaluations/n: %f' % (num_grad_evals / n) 
            print(output) 

        full_grad_norm = np.linalg.norm(full_grad)
        if full_grad_norm <= 1e-12:
            return score_list
        elif full_grad_norm >= 1e10:
            return score_list
        elif np.isnan(full_grad_norm):
            return score_list
           
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]

            # compute the gradients:
            x_grad = closure(x, Di, labels_i)[1]
            x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
            gk  = x_grad - x_tilde_grad + full_grad
            num_grad_evals = num_grad_evals + 2 * batch_size
                                   

            x -= step_size * gk
        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch    
    return score_list