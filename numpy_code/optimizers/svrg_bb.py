from dependencies import *

from utils import *
from datasets import *
from objectives import *


def svrg_bb(score_list, closure, batch_size, D, labels, 
            init_step_size, max_epoch=100, r=0, x0=None, verbose=True, adaptive_termination = False):
    """
        SVRG with Barzilai-Borwein step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: initial step size
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]

    if r <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')
    else:
        m = int(r * n)
        print('Info: set m = ', r, ' n')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    step_size = init_step_size
    num_grad_evals = 0

    for k in range(max_epoch):

        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()
        # estimate step size by BB method
        if k > 0:
            s = x_tilde - last_x_tilde
            y = full_grad - last_full_grad
            step_size = np.linalg.norm(s)**2 / np.dot(s, y) / m

        last_full_grad = full_grad
        last_x_tilde = x_tilde

        term1 = 0
        term2 = 0

        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad) )
            output += ', Func. value: %e' % loss
            output += ', Num gradient evaluations: %d' % num_grad_evals
            output += ', Step size: %.2e' % step_size
            print(output)

        # Add termination condition based on the norm of full gradient
        # Without this, "np.dot(s, y)" can underflow and produce divide-by-zero errors.
        if np.linalg.norm(full_grad) <= 1e-10:
            return score_list

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
            num_grad_evals = num_grad_evals + batch_size

            gk = x_grad - x_tilde_grad + full_grad

            if adaptive_termination == True:
                if (i+1) == int(n/2.):
                    term1, term2 = compute_pflug_statistic(term1, term2, (i+1), x, gk, step_size)
                    if (np.abs(term1 - term2)) < 1e-8:
                        print('Test passed. Breaking out of inner loop')
                        break

            x -= step_size * gk

    return score_list