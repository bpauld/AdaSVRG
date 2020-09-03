from dependencies import *


def make_closure(loss_fn, prior_prec=1e-2):
    '''Computes loss and gradient of the loss w.r.t. w
        Parameters:
            loss_fn: the loss function to use (logistic loss, hinge loss, squared error, etc)
            prior_prec: precision of the Gaussian prior (pass 0 to avoid regularization)
        Returns: a closure fn for computing the loss and gradient. '''

    def closure(w, X, y, backwards=True):
        '''Computes loss and gradient of the loss w.r.t. w
        Parameters:
            w: weight vector
            X: minibatch of input vectors
            y: labels for the input vectors
            prior_prec: precision of the Gaussian prior (pass 0 to avoid regularization)
        Returns: (loss, gradient)'''
        # change the Numpy Arrays into PyTorch Tensors
        X = torch.tensor(X)
        # Type of X is double, so y must be double.
        y = torch.tensor(y, dtype=torch.double)
        w = torch.tensor(w, requires_grad=True)

        # Compute the loss.
        loss = loss_fn(w, X, y) + (prior_prec / 2) * torch.sum(w ** 2)

        if backwards:
            # compute the gradient of loss w.r.t. w.
            loss.backward()
            # Put the gradient and loss back into Numpy.
            grad = w.grad.detach().numpy()
            loss = loss.item()

            return loss, grad
        else:
            loss = loss.item()

            return loss

    return closure

# PyTorch Loss Functions

def logistic_loss(w, X, y):
    ''' Logistic Loss'''
    n,d = X.shape
    return torch.mean(torch.log(1 + torch.exp(-torch.mul(y, torch.matmul(X, w)))))

def squared_hinge_loss(w, X, y):
    n,d = X.shape
    '''Squared Hinge Loss '''
    return torch.mean((torch.max( torch.zeros(n,dtype=torch.double) , torch.ones(n,dtype=torch.double) - torch.mul(y, torch.matmul(X, w))))**2 )

def squared_loss(w, X, y):
    n,d = X.shape
    '''Squared Loss'''
    return torch.mean(( y - torch.matmul(X, w) )**2)