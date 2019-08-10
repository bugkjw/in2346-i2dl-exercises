import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        # [conv - relu - 2x2 max pool] - [conv - relu - 2x2 max pool] - [fc - dropout - relu - fc]
        # Initialize NN layers with trainable parameters.
        self.conv1 = nn.Conv2d(1, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.fc1 = nn.Linear(32 * 24 * 24, 500)
        self.fc2 = nn.Linear(500, 30)
        with torch.no_grad():
            weight_scale = 1
            self.conv1.weight.mul_(weight_scale)
            self.conv2.weight.mul_(weight_scale)
            self.fc1.weight.mul_(weight_scale)
            self.fc2.weight.mul_(weight_scale)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        # [conv - relu - 2x2 max pool] - [conv - relu - 2x2 max pool] - [fc - dropout - relu - fc]
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        # From spatial convolution layer to 1D FC layer
        x = x.view(-1, 32 * 24 * 24)
        # FC
        x = self.fc1(x)
        x = self.fc2(F.relu(F.dropout(x, p=0.5)))
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
