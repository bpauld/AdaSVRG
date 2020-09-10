from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time

def armijo_ls(closure, D, labels, x, loss, grad, init_step_size, c, beta):

    temp_step_size = init_step_size
    armijo_iter = 1
    while closure(x - temp_step_size * grad, D, labels,
                  backwards=False) > loss - c * temp_step_size * np.linalg.norm(grad) ** 2:

        temp_step_size *= beta
        armijo_iter += 1
        if armijo_iter == 50:
            temp_step_size = 1e-6
            break

    step_size = temp_step_size

    return step_size, armijo_iter


def svrg_ada(score_list, closure, batch_size, D, labels, 
            init_step_size, max_epoch=100, r=0, x0=None, verbose=True,
            linesearch_option = 0, max_sgd_warmup_epochs= 0,  adaptive_termination = False,
            D_test=None, labels_test=None):
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

    term1 = 0
    term2 = 0
    t = 0

    step_size = init_step_size

    # SGD (Adagrad) passes
    for k in range(2 * max_sgd_warmup_epochs):
        t_start = time.time()

        if k % 2 == 0:
            loss, full_grad = closure(x, D, labels)
        
            score_dict = {"epoch": k}
            score_dict["n_grad_evals"] = num_grad_evals
            score_dict["train_loss"] = loss
            score_dict["grad_norm"] = np.linalg.norm(full_grad)
            score_dict["train_accuracy"] = accuracy(x, D, labels)
            score_dict["train_loss_log"] = np.log(loss)
            score_dict["grad_norm_log"] = np.log(score_dict["grad_norm"])
            score_dict["train_accuracy_log"] = np.log(score_dict["train_accuracy"])
            if D_test is not None:
                test_loss = closure(x, D_test, labels_test, backwards=False)
                score_dict["test_loss"] = test_loss
                score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
                score_dict["test_loss_log"] = np.log(test_loss)
                score_dict["test_accuracy_log"] = np.log(score_dict["test_accuracy"])

            if verbose:
                output = 'Epoch.: %d, Grad. norm: %.2e' % \
                         (k, np.linalg.norm(full_grad))
                output += ', Func. value: %e' % loss
                output += ', Num gradient evaluations: %d' % num_grad_evals
                print(output)

        Gk2 = 0
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            # compute the gradients:
            loss_temp, x_grad = closure(x, Di, labels_i)
            num_grad_evals = num_grad_evals + batch_size
            Gk2 = Gk2 + (np.linalg.norm(x_grad) ** 2)

            if linesearch_option == 2:
                reset_stepsize = step_size * ((2)**(batch_size/n))
                c = 0.5
                beta = 0.9
                step_size, armijo_iter = armijo_ls(closure, Di, labels_i, x, loss_temp, x_grad, reset_stepsize, c, beta)
                num_grad_evals = num_grad_evals + batch_size * armijo_iter

            else:
                step_size = init_step_size

            # computing statistic to decide termination
            # t = t + 1
            # term1, term2 = compute_statistic(term1, term2, t, x, x_grad, step_size)
            # print(np.abs(term1 - term2))

            x -= (step_size / np.sqrt(Gk2)) * x_grad
        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    for k in range(max_epoch - max_sgd_warmup_epochs):
        t_start = time.time()

        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()

        # initialize running sum of gradient norms
        Gk2 = 0

        term1 = 0
        term2 = 0

        if linesearch_option == 1 and k > 0:
            # c = 1e-4
            # beta = 0.9
            # reset_step_size = init_step_size
            # step_size, armijo_iter = armijo_ls(closure, D, labels, x, loss, full_grad, reset_step_size, c, beta)
            # num_grad_evals = num_grad_evals + n * armijo_iter

            s = x_tilde - last_x_tilde
            y = full_grad - last_full_grad
            step_size = np.linalg.norm(s) ** 2 / np.dot(s, y) / m

        else:
            step_size = init_step_size

        last_full_grad = full_grad
        last_x_tilde = x_tilde

        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Num gradient evaluations: %d' % num_grad_evals
            print(output)

        if np.linalg.norm(full_grad) <= 1e-12:
            return score_list

        num_grad_evals = num_grad_evals + n
        
        score_dict = {"epoch": k}
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x, D, labels)
        score_dict["train_loss_log"] = np.log(loss)
        score_dict["grad_norm_log"] = np.log(score_dict["grad_norm"])
        score_dict["train_accuracy_log"] = np.log(score_dict["train_accuracy"])
        if D_test is not None:
            test_loss = closure(x, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
            score_dict["test_loss_log"] = np.log(test_loss)
            score_dict["test_accuracy_log"] = np.log(score_dict["test_accuracy"])

        score_list += [score_dict]

        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            num_grad_evals = num_grad_evals + batch_size

            # compute the gradients:
            loss_temp, x_grad = closure(x, Di, labels_i)
            x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
            gk = x_grad - x_tilde_grad + full_grad
            Gk2 = Gk2 + (np.linalg.norm(gk) ** 2)

            if linesearch_option == 2:
                reset_step_size = step_size
                c = 0.5
                beta = 0.9
                step_size, armijo_iter = armijo_ls(closure, Di, labels_i, x, loss_temp, x_grad, reset_step_size, c, beta)
                num_grad_evals = num_grad_evals + batch_size * armijo_iter

            if adaptive_termination == True:
                if (i+1) == int(n/2.):
                    term1, term2 = compute_pflug_statistic(term1, term2, (i+1), x, gk, step_size)
                    if (np.abs(term1 - term2)) < 1e-8:
                        print('Test passed. Breaking out of inner loop')
                        break

            x -= (step_size / np.sqrt(Gk2)) * gk
        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list
