from dependencies import *

from utils import *
from datasets import *
from objectives import *


def svrg(score_list, closure, batch_size, D, labels, init_step_size, max_epoch=100, 
         r=0, x0=None, verbose=True, adaptive_termination = False):
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

        if num_grad_evals >= 2 * n * max_epoch:
            # exceeds the number of standard SVRG gradient evaluations (only for batch-size = 1)
            print('End of budget for gradient evaluations')
            break

        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()

        last_full_grad = full_grad
        last_x_tilde = x_tilde

        term1 = 0
        term2 = 0

        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Num gradient evaluations: %d' % num_grad_evals
            print(output)


        if np.linalg.norm(full_grad) <= 1e-10:
            break

        num_grad_evals = num_grad_evals + n

        score_dict = {"epoch": k}
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)

        score_list += [score_dict]
            
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
            num_grad_evals = num_grad_evals + batch_size

            if adaptive_termination == True:
                if (i+1) >= int(n/2.): # evaluate the statistic halfway through the inner loop
                    term1, term2 = compute_pflug_statistic(term1, term2, (i+1), x, gk, step_size)
                    if (np.abs(term1 - term2)) < 1e-10:
                        print('Test passed. Breaking out of inner loop at iteration ', (i+1))
                        x -= step_size * gk
                        break
            x -= step_size * gk
    
    return score_list