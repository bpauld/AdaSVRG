from dependencies import *

from utils import *
from datasets import *
from plotting import *
from objectives import *


def compute_statistic(term1, term2, t, x, g, eta, eps = 1e-4):

    term1 = 1.0/t * ((t - 1) * term1 + eta * np.dot(x,g))
    term2 = 1.0/t * ((t - 1) * term2 + (eta**2/2) * np.dot(g, g))

    return term1, term2


def svrg_bb(closure, batch_size, D, labels, init_step_size, n, d, max_epoch=100, m=0, x0=None, verbose=True, adaptive_termination = False):
    """
        SVRG with Barzilai-Borwein step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: initial step size
        n, d: size of the problem
    """
    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    step_size = init_step_size

    LOSS = []
    GRAD_NORM = []
    GRAD_EVALS = []

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
            GRAD_EVALS[k+1:max_epoch] = [0] * (max_epoch - k)
            LOSS[k + 1:max_epoch] = [0] * (max_epoch - k)
            GRAD_NORM[k + 1:max_epoch] = [0] * (max_epoch - k)
            return x, GRAD_EVALS, LOSS, GRAD_NORM

        num_grad_evals = num_grad_evals + n

        GRAD_EVALS.append(num_grad_evals)
        LOSS.append(loss)
        GRAD_NORM.append(np.linalg.norm(full_grad))

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
                if (i+1) == int(m/2.):
                    term1, term2 = compute_statistic(term1, term2, (i+1), x, gk, step_size)
                    if (np.abs(term1 - term2)) < 1e-8:
                        print('Test passed. Breaking out of inner loop')
                        break

            x -= step_size * gk

    GRAD_EVALS = np.asarray(GRAD_EVALS)
    LOSS = np.asarray(LOSS)
    GRAD_NORM = np.asarray(GRAD_NORM)
    return x, GRAD_EVALS, LOSS, GRAD_NORM
