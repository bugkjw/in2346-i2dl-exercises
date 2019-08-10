from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=100):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        # Train
        ii = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_acc = 0
            running_loss = 0.0
            running_acc = 0.0
            for i, data in enumerate(train_loader):
                ii += 1
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels.long())
                # Forward pass
                outputs = model(inputs)
                # Loss
                loss = self.loss_func(outputs, labels)
                # Backward pass and weight update
                optim.zero_grad()
                loss.backward()
                optim.step()
                # Accuracy and loss
                _, pred = torch.max(outputs.data, 1)
                #print(labels)
                #print(pred)
                running_acc += int(sum(pred == labels)) / labels.size(0)
                running_loss += loss.item()
                if ii % log_nth == 0:
                    print('[Iteration %d/%d] TRAIN loss: %.3f'
                          % (ii, iter_per_epoch*num_epochs, running_loss / log_nth))
                    train_loss = running_loss / log_nth
                    train_acc = running_acc / log_nth
                    running_loss = 0.0
                    running_acc = 0.0
            # Validation
            model.eval()
            running_loss = 0.0
            cnt = 0
            correct = 0
            total = 0
            for i, data in enumerate(val_loader):
                # Predict
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels.long())
                outputs = model(inputs)
                loss = self.loss_func(outputs, labels)
                _, pred = torch.max(outputs.data, 1)
                # Record
                running_loss += loss.item()
                cnt += 1
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            val_loss = running_loss / cnt
            val_acc = correct / total
            # Save statistics
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            # Print
            print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f'
                  % (epoch+1, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1]))
            print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f'
                  % (epoch+1, num_epochs, self.val_acc_history[-1], self.val_loss_history[-1]))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
