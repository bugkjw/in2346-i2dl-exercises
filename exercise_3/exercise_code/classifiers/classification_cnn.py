"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=9,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
        # [conv - relu - conv - relu - 2x2 max pool] * 2 - [fc - dropout - relu - fc]
        same = int((kernel_size - 1) / 2)
        # Initialize NN layers with trainable parameters.
        self.conv11 = nn.Conv2d(channels, num_filters, kernel_size, stride=stride_conv, padding=same)
        self.conv12 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride_conv, padding=same)
        h_pool1 = int(1 + (height - pool) / stride_pool)
        w_pool1 = int(1 + (width - pool) / stride_pool)
        self.conv21 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride_conv, padding=same)
        self.conv22 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride_conv, padding=same)
        h_pool2 = int(1 + (h_pool1 - pool) / stride_pool)
        w_pool2 = int(1 + (w_pool1 - pool) / stride_pool)
        self.fc1 = nn.Linear(num_filters * h_pool2 * w_pool2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        with torch.no_grad():
            self.conv11.weight.mul_(weight_scale)
            self.conv12.weight.mul_(weight_scale)
            self.conv21.weight.mul_(weight_scale)
            self.conv22.weight.mul_(weight_scale)
            self.fc1.weight.mul_(weight_scale)
            self.fc2.weight.mul_(weight_scale)
        self.other_params = (pool, stride_pool, dropout)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        # [conv - relu - conv - relu - 2x2 max pool] * 2 - [fc - dropout - relu - fc]
        pool, stride_pool, dropout = self.other_params
        x = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(x)))), kernel_size=pool, stride=stride_pool)
        x = F.max_pool2d(F.relu(self.conv22(F.relu(self.conv21(x)))), kernel_size=pool, stride=stride_pool)
        # From spatial convolution layer to 1D FC layer
        size = x.size()[1:]
        x = x.view(-1, reduce(lambda i, j: i * j, size))
        # FC
        x = self.fc1(x)
        x = self.fc2(F.relu(F.dropout(x, p=dropout)))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x


    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
