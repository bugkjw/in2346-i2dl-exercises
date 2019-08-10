"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # Detect matrix size
    N = y.shape[0]
    D, C = W.shape
    # Iterate over the minibatch dataset and compute loss for each data, along with gradient dLi/dW.
    loss_minibatch = np.zeros(N,)
    dW_minibatch = np.zeros((D, C, N))
    for ii in range(0, N):

        # TODO: Process current data
        # Data column
        temp_X = X[ii, :]
        # Training label
        temp_y = y[ii]
        # Evaluate scores with given weight matrix
        temp_f = np.dot(temp_X, W)
        # Use normalization trick for numerical stability
        temp_f -= np.max(temp_f)
        temp_exps = np.exp(temp_f)
        temp_normalizer = np.sum(temp_exps)

        # TODO: Compute the cross-entropy loss
        # Score of correct training label
        temp_f_y = temp_f[temp_y]
        loss_minibatch[ii] = -np.log(np.exp(temp_f_y)/temp_normalizer)

        # TODO: Compute gradient
        # dLi/dW
        temp_dW = np.zeros((D, C))
        for mm in range(0, D):
            for nn in range(0, C):
                temp_f_n = temp_f[nn]
                temp_p = np.exp(temp_f_n)/temp_normalizer
                temp_dW[mm, nn] = (temp_p-(nn == temp_y))*temp_X[mm]
        dW_minibatch[:, :, ii] = temp_dW

    # Final loss
    loss = np.mean(loss_minibatch)
    # Add loss from regularization
    loss += reg*np.sum(np.multiply(W, W))
    # Final dW
    dW = np.mean(dW_minibatch, 2)+2*reg*W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # TODO: Process input data
    # Detect matrix size
    N = y.shape[0]
    D, C = W.shape
    f = np.matmul(X, W)
    # Numerical stability
    f -= np.amax(f, 1, keepdims=True)
    exps = np.exp(f)
    exp_sum = np.sum(exps, 1, keepdims=True)
    p = np.divide(exps, exp_sum)
    p_correct = p[np.arange(N), y]

    # TODO: Compute the cross-entropy loss
    loss_minibatch = -np.log(p_correct)
    loss = np.mean(loss_minibatch)
    # Add loss from regularization
    loss += reg*np.sum(np.multiply(W, W))

    # TODO: Vectorized gradient computation
    p_del = p.copy()
    p_del[np.arange(N), y] -= 1
    dW = np.matmul(X.T, p_del)/N
    # Add loss from regularization
    dW += 2*reg*W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    learning_rates = np.logspace(np.log10(4e-7), np.log10(16e-7), 5)
    # [2e-7, 8e-7, 32e-7]
    regularization_strengths = np.logspace(np.log10(3e3), np.log10(48e3), 5)
    # [3000, 12000, 48000]

    for lr in learning_rates:
        for reg in regularization_strengths:
            print('lr %e reg %e' % (lr, reg))
            hyperparameters = (lr, reg)
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=1500, verbose=False)
            y_train_pred = softmax.predict(X_train)
            train_accuracy = np.mean(y_train == y_train_pred)
            y_val_pred = softmax.predict(X_val)
            val_accuracy = np.mean(y_val == y_val_pred)
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_softmax = softmax
            results[hyperparameters] = (train_accuracy, val_accuracy)
            all_classifiers.append((softmax, val_accuracy))
            print('train accuracy: %f val accuracy: %f' % (train_accuracy, val_accuracy))

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
