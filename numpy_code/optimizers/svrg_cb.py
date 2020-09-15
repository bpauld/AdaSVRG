from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time

def svrg_cb(score_list, closure, batch_size, D, labels, 
            max_epoch=100, r=0, x0=None, verbose=True,             
            D_test=None, labels_test=None, alpha = 100, reset = True, adaptive_termination = 0):
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
        m = int(n)
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

    if adaptive_termination == 2:
        # hardcoding these parameters for now. 
        q = 1.5
        k0 = 5                
        num_checkpoints = 20 # number of times to check
        threshold  = 0.6

        start = int(q**(k0))
            
        check_iter_indices = list(map(int,  list(np.linspace(start, m, num_checkpoints))))
        save_iter_indices = list(map(int, np.linspace(start, m, num_checkpoints) / q))
        print(check_iter_indices, save_iter_indices)

    for k in range(max_epoch):

        if num_grad_evals >= (n + n / batch_size) * max_epoch:
            # exceeds the number of standard SVRG gradient evaluations (only for batch-size = 1)
            print('End of budget for gradient evaluations')
            break

        t_start = time.time()

        if adaptive_termination == 2:
            save_dist = np.zeros((num_checkpoints)) 
            checkpoint_num = 0

        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()
    
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
        # score_dict["train_accuracy_log"] = np.log(score_dict["train_accuracy"])
        if D_test is not None:
            test_loss = closure(x, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
            score_dict["test_loss_log"] = np.log(test_loss)
            # score_dict["test_accuracy_log"] = np.log(score_dict["test_accuracy"])

        score_list += [score_dict]

        if (k == 0)  or (reset):
            L_max = np.zeros(d)
            gk_norm_max = np.zeros(d)
            reward = np.zeros(d)
            theta = np.zeros(d)
            x0 = np.copy(x)     
            # alpha  = np.absolute(full_grad        )
            # alpha = 1. / np.linalg.norm(full_grad)       
            # print(alpha)

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
            gk = -1. * (x_grad - x_tilde_grad + full_grad)
        
            L_max = np.maximum(L_max,np.absolute(gk))
            gk_norm_max = gk_norm_max + np.absolute(gk)        

            reward = np.maximum(np.zeros(d), reward + np.multiply((x - x0),gk) )
            theta = theta + gk
            
            denom = np.multiply(L_max, np.maximum(np.multiply(alpha, L_max), gk_norm_max + L_max))            
            grad_cb = np.divide(theta, denom)
            grad_cb = np.multiply(grad_cb, L_max + reward )

            x = x0 + grad_cb 
                        
            if adaptive_termination == 2:

                if (i+1) in (save_iter_indices):                    
                    save_dist[checkpoint_num] = np.linalg.norm(x - x_tilde)
                    checkpoint_num += 1
            
                elif ((i+1) in check_iter_indices):
               
                    t1 = i+1
                                
                    ind = check_iter_indices.index(i+1)                    
                    x_prev_dist = save_dist[ind]
                    x_dist = np.linalg.norm(x - x_tilde)
                    t2 = save_iter_indices[ind]
                    S = (np.log(x_dist**2) - np.log(x_prev_dist**2)) / (np.log(t1) - np.log(t2))
                    print('S = ', S)
                    if S < threshold:                        
                        print('Test passed. Breaking out of inner loop at iteration ', (i+1))
                        break

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list
