from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time

def armijo_ls(closure, D, labels, x, loss, grad, p, init_step_size, c, beta):

    temp_step_size = init_step_size
    armijo_iter = 1
    while closure(x - temp_step_size * grad, D, labels,
                  backwards=False) > loss - c * temp_step_size * np.dot(grad, p) :

        temp_step_size *= beta
        if armijo_iter == 50:
            temp_step_size = 1e-6
            break
        armijo_iter += 1

    step_size = temp_step_size

    return step_size, armijo_iter


def svrg_ada(score_list, closure, batch_size, D, labels, 
            init_step_size = 1, max_epoch=100, r=0, x0=None, verbose=True,
            linesearch_option = 1,
            adaptive_termination = 0,
            D_test=None, labels_test=None, reset=True, threshold_at=1, interval_size=10,
            average_iterates=False):
    """
        SVRG-Ada for solving finite-sum problems
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
    step_size = init_step_size
    for k in range(max_epoch):
        t_start = time.time()

        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()
        num_grad_evals = num_grad_evals + n

        # initialize running sum of gradient norms
        if (k == 0) or (reset):
            Gk2 = 0
            
        if average_iterates:
            running_average = 0
            counter = 0
           
        if linesearch_option == 0:
            step_size = init_step_size                       
        
        elif linesearch_option == 1:
            if k == 0:
                x_rand = np.random.normal(0, 1, d)
                loss_rand , full_grad_rand = closure(x_rand, D, labels)
                num_grad_evals = num_grad_evals + n
                L_hat = np.linalg.norm(full_grad - full_grad_rand) / np.linalg.norm(x_tilde - x_rand)

                # to prevent the step-size from blowing up. 
                if (L_hat < 1e-8):
                    step_size = 1e-4
                else:
                    step_size = np.linalg.norm(full_grad) / L_hat
                       
            else:
                L_hat = np.linalg.norm(full_grad - last_full_grad) / np.linalg.norm(x_tilde - last_x_tilde)

                # to prevent the step-size from blowing up. 
                if (L_hat < 1e-8):
                    step_size =  1e-4 # some small default step-size
                else:
                    step_size = np.linalg.norm(full_grad) / L_hat
               
        elif linesearch_option == 2 and (k ==  0):
            step_size = init_step_size
                                      
        last_full_grad = full_grad
        last_x_tilde = x_tilde

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

        # for adaptive termination
        if adaptive_termination == 1:
            at_mem = []
            check_pt  = 0  
            S_list = []
        if adaptive_termination == 2:
            sum_vector = np.zeros(2)
            sum_matrix = np.zeros((2,2))
            S_list = []
        
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            num_grad_evals = num_grad_evals + 2 * batch_size

            # compute the gradients:
            loss_temp, x_grad = closure(x, Di, labels_i)
            x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
            gk = x_grad - x_tilde_grad + full_grad
            Gk2 = Gk2 + (np.linalg.norm(gk) ** 2)

            if linesearch_option == 2:
                reset_step_size = step_size 
                c = 0.5
                beta = 0.9
                step_size, armijo_iter = armijo_ls(closure, Di, labels_i, x, loss_temp, x_grad, gk, reset_step_size, c, beta)
                num_grad_evals = num_grad_evals + batch_size * armijo_iter
                
            if adaptive_termination == 1:
                if (i % interval_size == 0): # and i >= int(min(n,m)/2):
                    check_pt = check_pt  + 1                                                            
                    if check_pt > 2:                                                                      
                        a0 = at_mem[0][0]
                        a1 = at_mem[1][0]

                        b0 = at_mem[0][1]
                        b1 = at_mem[1][1]

                        slope = (b1 - b0) / (a1 - a0)
                        bias =  b1 - slope * a1

                        S = np.abs(slope * i + bias - Gk2) / Gk2
                        S_list.append(S)                                                                  
                        if S < threshold_at:
                            print('Breaking out of inner loop at iteration', i)
                            break

                    if len(at_mem) < 2:
                        at_mem.append([i, Gk2])
                    else:
                        at_mem[0] = at_mem[1]
                        at_mem[1] = [i, Gk2]   
                        
            if adaptive_termination == 2:
                if (i % interval_size == 0):
                    check_pt = check_pt  + 1
                    if check_pt > 2:
                        weight = np.linalg.lstsq(sum_matrix, sum_vector, rcond=None)
                        slope = weight[0][0]
                        bias = weight[0][1]
                        S = np.abs(slope * i + bias - Gk2) / Gk2
                        S_list.append(S) 
                        if S < threshold_at:
                            print('Breaking out of inner loop at iteration', i)  
                            break                        
                    a = np.zeros(2)
                    a[0] = i
                    a[1] = 1 #add bias term
                    sum_vector += Gk2*a
                    sum_matrix += np.outer(a, np.transpose(a))
            
            x -= (step_size / np.sqrt(Gk2)) * gk
            
            if average_iterates and i > m/2: #allow for a burn-in phase
                running_average += x
                counter += 1
                
        if average_iterates and counter > 0: #just in case the batch size is so big that we don't average at all
            x = 1/counter * running_average    
        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list
