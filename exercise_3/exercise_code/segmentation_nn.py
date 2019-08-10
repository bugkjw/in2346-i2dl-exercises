"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        # Load pre-trained CNN classifier
        # [conv - relu - conv - relu - 2x2 max pool] * 2 - [fc - dropout - relu - fc]
        cnn = torch.load(f="models/classification_cnn.model")
        print(cnn)
        self.conv11 = copy.deepcopy(cnn.conv11)
        self.conv12 = copy.deepcopy(cnn.conv12)
        self.conv21 = copy.deepcopy(cnn.conv21)
        self.conv22 = copy.deepcopy(cnn.conv22)
        kernel_size = self.conv11.kernel_size[0]
        num_filters = self.conv11.out_channels
        hidden_dim = cnn.fc1.out_features
        same = int((kernel_size - 1) / 2)
        self.other_params = cnn.other_params
        # Modify the architecture
        # [conv - relu - conv - relu - 2x2 max pool] * 3 - [1-conv - dropout - relu - 1-conv] - upsample
        self.conv31 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=1, padding=same)
        self.conv32 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=1, padding=same)
        self.convf1= nn.Conv2d(num_filters, hidden_dim, 1)
        self.convf2 = nn.Conv2d(hidden_dim, num_classes, 1)
        del cnn
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        # [conv - relu - conv - relu - 2x2 max pool] * 3  - [1-conv - dropout - relu - 1-conv] - upsample
        pool, stride_pool, dropout = self.other_params
        h0, w0 = (x.size()[2], x.size()[3])
        x120 = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(x)))), kernel_size=pool, stride=stride_pool)
        x60 = F.max_pool2d(F.relu(self.conv22(F.relu(self.conv21(x120)))), kernel_size=pool, stride=stride_pool)
        x30 = F.max_pool2d(F.relu(self.conv32(F.relu(self.conv31(x60)))), kernel_size=pool, stride=stride_pool)
        #x15 = F.max_pool2d(F.relu(self.conv42(F.relu(self.conv41(x30)))), kernel_size=pool, stride=stride_pool)
        x = F.interpolate(self.convf2(F.relu(F.dropout(self.convf1(x30), p=dropout))), size=(h0, w0), mode='bilinear')
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
