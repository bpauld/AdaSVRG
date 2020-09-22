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

def initialize_at(num_checkpoints,m):
    # hardcoding these parameters for now. 
    q =  1.5
    k0 = 5

    start = int(q**(k0))
    check_iter_indices = list(map(int,  list(np.linspace(start, m, num_checkpoints))))
    save_iter_indices = list(map(int, np.linspace(start, m, num_checkpoints) / q))    
    return check_iter_indices, save_iter_indices

def reset_at(num_checkpoints):
    save_dist = np.zeros((num_checkpoints)) 
    checkpoint_num = 0  
    return checkpoint_num, save_dist

def check_update_at(i, x,  x_tilde, checkpoint_num, save_dist, 
                    save_iter_indices, check_iter_indices):  

    threshold  = 0.6
    terminate  = 0
    
    if (i+1) in (save_iter_indices):                    
        save_dist[checkpoint_num] = np.linalg.norm(x - x_tilde)
        checkpoint_num += 1

    elif ((i+1) in check_iter_indices):
        t1 = i+1                    
        ind = check_iter_indices.index(i+1)                    
        x_prev_dist = save_dist[ind]
        x_dist = np.linalg.norm(x - x_tilde)
        t2 = save_iter_indices[ind]

        # print(x_dist, x_prev_dist)

        S = (np.log10(x_dist**2) - np.log10(x_prev_dist**2)) / (np.log10(t1) - np.log10(t2))
        print('S = ', S)        
        if S  < threshold:
            # print('Terminaating')
            terminate = 1 

    return checkpoint_num,save_dist, terminate
    
def svrg_ada_at(score_list, closure, batch_size, D, labels, 
            init_step_size = 1, max_epoch=100, r=0, x0=None, verbose=True,
            linesearch_option = 1, adaptive_termination = False,  
            D_test=None, labels_test=None, reset = True):
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
    # num_checkpoints = 50 # number of times to check the test in each outer loop
    check_pt  = 0

    step_size = init_step_size    

    # if adaptive_termination == True:        
    #     check_iter_indices, save_indices = initialize_at(num_checkpoints, m)
        
    for k in range(max_epoch):
        t_start = time.time()

        if num_grad_evals >= 2 * n * max_epoch:
            # exceeds the number of standard SVRG gradient evaluations (only for batch-size = 1)
            print('End of budget for gradient evaluations')
            break        

        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()
        num_grad_evals = num_grad_evals + n

        # initialize running sum of gradient norms
        if (k == 0) or (reset):
            Gk2 = 0
           
        if linesearch_option == 0:
            step_size = init_step_size

        elif linesearch_option == 1:

            if k == 0:      
                # do a line-search in the first epoch
                reset_step_size = 1
            else:
                reset_step_size = reciprocal_L_hat * 2

            c = 1e-4
            beta = 0.9            
            reciprocal_L_hat, armijo_iter = armijo_ls(closure, D, labels, x, loss, full_grad, full_grad, reset_step_size, c, beta)
            num_grad_evals = num_grad_evals + (n * armijo_iter)/2
      
            # incorporate the correction for Adagrad
            step_size = np.linalg.norm(full_grad) * reciprocal_L_hat                        
            
        elif linesearch_option == 2 and (k ==  0):
            step_size = init_step_size
            
        elif linesearch_option == 3:
            if k == 0:
                reset_step_size = 1
                c = 1e-4
                beta = 0.9            
                reciprocal_L_hat, armijo_iter = armijo_ls(closure, D, labels, x, loss, full_grad, full_grad, reset_step_size, c, beta)
                num_grad_evals = num_grad_evals + (n * armijo_iter)/2
                step_size = np.linalg.norm(full_grad) * reciprocal_L_hat
            else:
                L_hat = np.linalg.norm(full_grad - last_full_grad) / np.linalg.norm(x_tilde - last_x_tilde)
                step_size = np.linalg.norm(full_grad) / L_hat
                
        elif linesearch_option == 4:
            if k == 0:
                reset_step_size = 1
                c = 1e-4
                beta = 0.9            
                reciprocal_L_hat, armijo_iter = armijo_ls(closure, D, labels, x, loss, full_grad, full_grad, reset_step_size, c, beta)
                num_grad_evals = num_grad_evals + (n * armijo_iter)/2
                step_size = np.linalg.norm(full_grad) * reciprocal_L_hat
            else:
                L_hat = np.linalg.norm(full_grad - last_full_grad) / np.linalg.norm(x_tilde - last_x_tilde)
                step_size = min(np.linalg.norm(full_grad) / (L_hat), 2*step_size)
                
        # if adaptive_termination == True:
            # checkpoint_num, save_dist = reset_at(num_checkpoints)
               
        last_full_grad = full_grad
        last_x_tilde = x_tilde

        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % step_size
            output += ', Num gradient evaluations: %d' % num_grad_evals
            print(output)        

        if np.linalg.norm(full_grad) <= 1e-12:
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
        
        check_pt  = 0 

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
                step_size, armijo_iter = armijo_ls(closure, Di, labels_i, x, loss_temp, x_grad, gk, reset_step_size, c, beta)
                num_grad_evals = num_grad_evals + batch_size * armijo_iter

            if adaptive_termination == True:
                if  (i % n / (2. * batch_size)) == 0 and i  > 0:

                    check_pt = check_pt  + 1
                                                            
                    if check_pt > 1:
                        # print(Gk2)
                        # S = (Gk2 / (i+1) - prev_Gk2)  /  (i - prev_i) 
                        # S = (Gk2 / prev_Gk2) /  (i / prev_i)
                        S = (Gk2 / i) / (prev_Gk2 / prev_i) 
                        print('S =  ', S)

                        if S > 1:
                            break

                    prev_Gk2 = Gk2
                    prev_i = i
                    # print(prev_Gk2, prev_i)
                
                # checkpoint_num, save_dist, terminate = \
                #     check_update_at(i, x,  x_tilde, checkpoint_num, 
                #                     save_dist, save_iter_indices, check_iter_indices)  

                # if terminate == 1 and (i > n / (2 * batch_size)):                                        
                    # print('Test passed. Breaking out of inner loop at iteration ', (i+1))
                    # break            
                    
            x -= (step_size / np.sqrt(Gk2)) * gk
            # print(np.sqrt(Gk2) / math.sqrt(i+1))
            
        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list
