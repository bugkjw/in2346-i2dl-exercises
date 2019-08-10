"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################

        # Setting up bias trick
        W1_ = np.append(W1, [b1], 0)
        X_ = np.append(X, np.ones((N, 1)), 1)
        W2_ = np.append(W2, [b2], 0)
        # Layer 1 activation by RELU
        h = np.matmul(X_, W1_)
        h[h < 0] = 0
        h_ = np.append(h, np.ones((h.shape[0], 1)), 1)
        # Output layer activation
        scores = np.matmul(h_, W2_)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################

        # Apply softmax to scores
        scores = scores-np.max(scores, 1, keepdims=True)
        exps = np.exp(scores)
        exps_sum = np.sum(exps, 1, keepdims=True)
        # Compute loss
        p = np.divide(exps, exps_sum)
        p_correct = p[np.arange(N), y]
        loss = np.mean(-np.log(p_correct))
        # Add regularization loss
        loss += reg*np.sum(np.multiply(W1_, W1_))/2
        loss += reg*np.sum(np.multiply(W2_, W2_))/2

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################

        # TODO: Output layer
        p_del = p.copy()
        p_del[np.arange(y.shape[0]), y] -= 1
        dW2_ = np.matmul(h_.T, p_del)/N
        # Add loss from regularization
        dW2_ += 2*reg*W2_/2

        # TODO: Hidden layer
        # X_: N X D+1
        # p_del: N X C
        # W2: H X C
        # h: N X H
        dW1_ = np.matmul(X_.T, np.multiply(np.matmul(p_del, W2.T), h > 0))/N
        dW1_ += 2*reg*W1_/2

        grads['W1'] = dW1_[:-1, :]
        grads['b1'] = dW1_[-1, :]
        grads['W2'] = dW2_[:-1, :]
        grads['b2'] = dW2_[-1, :]

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################

            batch_inds = np.random.choice(num_train, batch_size)
            X_batch = X[batch_inds, :]
            y_batch = y[batch_inds]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################

        scores = self.loss(X)
        y_pred = np.argmax(scores, 1)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 

    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################
    '''
    results = {}
    best_val = -1
    # Hyperparameter grid
    hidden_layer_sizes = [50, 200, 500]
    learning_rates = [5e-4, 25e-4, 1e-3]
    regularization_strengths = [5e-5, 5e-3, 5e-1]
    for hs in hidden_layer_sizes:
        for lr in learning_rates:
            for reg in regularization_strengths:
                net = TwoLayerNet(input_size=32*32*3, hidden_size=hs, output_size=10)
                net.train(X_train, y_train, X_val, y_val, num_iters=2000, learning_rate=lr, reg=reg, verbose=False)
                y_train_pred = net.predict(X_train)
                train_accuracy = np.mean(y_train == y_train_pred)
                y_val_pred = net.predict(X_val)
                val_accuracy = np.mean(y_val == y_val_pred)
                if val_accuracy > best_val:
                    best_val = val_accuracy
                    best_net = net
                hyperparameters = (hs, lr, reg)
                accuracy = (train_accuracy, val_accuracy)
                results[hyperparameters] = accuracy
                print('hs %d lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    hs, lr, reg, train_accuracy, val_accuracy))

    # Print out results.
    for (hs, lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(hs, lr, reg)]
        print('hs %d lr %e reg %e train accuracy: %f val accuracy: %f' % (
            hs, lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during validation: %f' % best_val)
    '''
    net = TwoLayerNet(input_size=32 * 32 * 3, hidden_size=500, output_size=10)
    net.train(X_train, y_train, X_val, y_val, num_iters=2000, learning_rate=1e-3, reg=5e-3, verbose=True)
    y_train_pred = net.predict(X_train)
    train_accuracy = np.mean(y_train == y_train_pred)
    y_val_pred = net.predict(X_val)
    val_accuracy = np.mean(y_val == y_val_pred)
    best_net = net
    print('hs %d lr %e reg %e train accuracy: %f val accuracy: %f' % (
        500, 25e-4, 5e-3, train_accuracy, val_accuracy))

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
