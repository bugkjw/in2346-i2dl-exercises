import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialize h as 0 if these values are not given.                          #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        T = x.size()[0]
        h = torch.zeros((1, x.size()[1], self.hidden_size))
        h = h[0, :, :]
        for t in range(T):
            x_t = x[t, :, :]
            h = self.tanh(self.W(x_t) + self.V(h))
            h_seq.append(h)
        h_seq = torch.stack(tuple(h_seq))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h
    
    
class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
    ############################################################################
    # TODO: Build a one layer LSTM with an activation with the attributes      #
    # defined above and a forward function below. Use the nn.Linear() function #
    # as your linear layers.                                                   #
    # Initialse h and c as 0 if these values are not given.                    #
    ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_forget = nn.Linear(input_size, hidden_size)
        self.U_forget = nn.Linear(hidden_size, hidden_size)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.U_in = nn.Linear(hidden_size, hidden_size)
        self.W_out = nn.Linear(input_size, hidden_size)
        self.U_out = nn.Linear(hidden_size, hidden_size)
        self.W_cell = nn.Linear(input_size, hidden_size)
        self.U_cell = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        seq_len, batch_size, input_size = x.size()
        if h is None:
            h = torch.zeros((1, batch_size, self.hidden_size))
        if c is None:
            c = torch.zeros((1, batch_size, self.hidden_size))
        h_seq = []
        for t in range(seq_len):
            x_t = x[t, :, :]
            forget_t = F.sigmoid(self.W_forget(x_t)+self.U_forget(h))
            in_t = torch.sigmoid(self.W_in(x_t)+self.U_in(h))
            out_t = torch.sigmoid(self.W_out(x_t)+self.U_out(h))
            c = torch.mul(forget_t, c)
            c += torch.mul(in_t, F.tanh(self.W_cell(x_t)+self.U_cell(h)))
            h = torch.mul(out_t, F.tanh(c))
            h_seq.append(h)
        h_seq = torch.stack(tuple(h_seq)).reshape((seq_len, batch_size, self.hidden_size))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################
        self.hidden_size = hidden_size
        self.RNN = nn.RNN(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, classes)
       
    def forward(self, x):
        batch_size = x.size()[1]
        rec, x = self.RNN(x)
        x = F.dropout(F.relu(self.fc1(x.reshape(batch_size, self.hidden_size))))
        x = F.relu(self.fc2(x))
        return x

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28 , hidden_size=128):
        super(LSTM_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a LSTM classifier                                           #
    ############################################################################
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, classes)
    
    def forward(self, x):
        batch_size = x.size()[1]
        rec, (x, _) = self.LSTM(x)
        x = F.dropout(F.relu(self.fc1(x.reshape(batch_size, self.hidden_size))))
        x = F.relu(self.fc2(x))
        return x
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)