from dependencies import *


from utils import *
from datasets import *
from plotting import *
from objectives import *

def svrg_armijo(closure,
                batch_size,
                D,
                labels,
                n,
                d,
                c,
                beta,
                max_iter_armijo,
                max_step_size,
                reset_step_size=False,
                max_epoch=100,
                m=0,
                x0=None,
                verbose=True):
    """
        SVRG with Armijo step size in the inner loop for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: initial step size
        n, d: size of the problem
        max_iter_armijo: maximum number of Armijo iterations before falling back to small determined stepsize
        reset_step_size: whether or not the stepsize should be reset to max_step_size at the next iteration
        beta: multiplicative factor in Armijo line search (0 < beta < 1).
    """
    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    start_step_size = max_step_size
    LOSS = np.zeros((max_epoch))
    GRAD_NORM = np.zeros((max_epoch))
    AVERAGE_ARMIJO_STEPS = np.zeros((max_epoch))
    MAX_ARMIJO_STEPS_REACHED = np.zeros((max_epoch))

    for k in range(max_epoch):
        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()

        last_full_grad = full_grad
        last_x_tilde = x_tilde

        if verbose:
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %.2e' % \
                     (k, max_step_size, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            print(output)

        # Add termination condition based on the norm of full gradient
        # Without this, "np.dot(s, y)" can underflow and produce divide-by-zero errors.
        if np.linalg.norm(full_grad) <= 1e-14:
            return x, LOSS, GRAD_NORM

        LOSS[k] = loss
        GRAD_NORM[k] = np.linalg.norm(full_grad)

        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]

            # compute the gradients:
            f_x, x_grad = closure(x, Di, labels_i)
            x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
            g = x_grad - x_tilde_grad + full_grad
            g_norm = np.linalg.norm(g)

            step_size = start_step_size
            found = False
            for j in range(max_iter_armijo):
                if closure(x - step_size * g, Di, labels_i)[0] > f_x - c * step_size * g_norm:
                    step_size *= beta
                else:
                    found = True
                    break

            AVERAGE_ARMIJO_STEPS[k] += j
            if found:
                x -= step_size * g
                if not reset_step_size:
                    start_step_size = step_size
            else:
                MAX_ARMIJO_STEPS_REACHED[k] += 1
                x -= 1e-6 * g
        AVERAGE_ARMIJO_STEPS[k] /= m
        output = 'Epoch.: %d, Avg armijo steps: %.2e, Nb max iteration reached: %.2e' % \
                 (k, AVERAGE_ARMIJO_STEPS[k], MAX_ARMIJO_STEPS_REACHED[k])
        print(output)

    return x, LOSS, GRAD_NORM