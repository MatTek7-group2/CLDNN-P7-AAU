# -*- coding: utf-8 -*-
"""

Created on Tuesday Dec. 15 13:37:00 2020

Authors:  Martin Voigt Vejling, Daniel Bernard van Diepen,
          Andreas Anton Andersen, Morten Stig Kaaber
E-Mails: {mvejli17, dvandi17, aand17, mkaabe17}@student.aau.dk

This module contains functionality to import the Aurora-2 data and Apollo-11
data to use for voice activity detection with a convolutional long
short-term memory fully connected deep neural network (CLDNN) as described
in the paper

    On the Generalisation Ability of Unsupervised and Supervised Voice
                  Activity Detection Methods (2020)

Functionality for training and testing the CLDNN using PyTorch is included
herein the use of early stopping, computing and plotting the receiver
operation characteristic (ROC) curves and approximating the area under
the ROC curve, abreviated AUC.

The functionality provided in this module is used in the script "CLDNN_MAIN"
which is located in the GitHub repository.

To use this code the Aurora-2 and Apollo-11 data should be available and
organised as described in the readme file.

The scripts are developed using Python 3.6 and PyTorch 1.7.0.

"""

from __future__ import print_function
import numpy as np
import argparse
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import P7_Module as P7
from time import time
from sklearn.model_selection import train_test_split
import copy
from sklearn.utils import shuffle


class CLDNN(nn.Module):

    def __init__(self, w_len, step, context_size, **kwargs):
        len_context = context_size*step
        len_data = 2*len_context + w_len

        self.stride1 = kwargs['stride1']

        self.dropout_rate_input = kwargs['dropout_rate_input']
        self.dropout_rate_hidden = kwargs['dropout_rate_hidden']

        self.out_channels1 = kwargs['out_channels1']
        self.kernel_len1 = kwargs['kernel_len1']
        self.pool_size1 = kwargs['pool_size1']

        #print((len_data - self.kernel_len1 + 1)/self.pool_size1)
        self.len2 = self.out_channels1
        self.in_channels2 = floor((len_data - self.kernel_len1 + 1)/(self.stride1*self.pool_size1))
        self.out_channels2 = kwargs['out_channels2']
        self.kernel_len2 = kwargs['kernel_len2']
        self.pool_size2 = kwargs['pool_size2']

        #print((self.len2 - self.kernel_len2 + 1)/self.pool_size2)
        self.len_lstm1 = self.out_channels2*floor((self.len2 - self.kernel_len2 + 1)/self.pool_size2)
        #print(self.len_lstm1)
        self.lstm_hidden_units = kwargs['lstm_hidden_units']

        self.dense_hidden_units = kwargs['dense_hidden_units']


        super(CLDNN, self).__init__()
        self.dropout_input = nn.Dropout(self.dropout_rate_input)
        self.dropout_hidden = nn.Dropout(self.dropout_rate_hidden)

        self.conv1 = nn.Conv1d(1, self.out_channels1, self.kernel_len1, self.stride1)
        self.conv2 = nn.Conv1d(self.in_channels2, self.out_channels2, self.kernel_len2)

        self.lstm1 = nn.LSTM(self.len_lstm1, self.lstm_hidden_units)
        self.lstm2 = nn.LSTM(self.lstm_hidden_units, self.lstm_hidden_units)

        self.fc1 = nn.Linear(self.lstm_hidden_units, self.dense_hidden_units)
        self.fc2 = nn.Linear(self.dense_hidden_units, 2)

    def forward(self, x):
        x = self.dropout_input(x)

        # Time convolution
        x = self.conv1(x)
        x = self.dropout_hidden(x)
        x = F.max_pool1d(x, self.pool_size1)
        x = torch.log(torch.add(F.relu(x), 0.01)) #log(ReLU(x) + 0.01)

        # Frequency convolution
        x = torch.transpose(x, 1, 2)
        x = self.conv2(x)
        x = F.max_pool1d(x, self.pool_size2)
        x = F.relu(x)
        x = torch.flatten(x, 1).unsqueeze(0)
        x = self.dropout_hidden(x)

        # LSTM
        x, _ = self.lstm1(x) #tanh activation by default
        x = self.dropout_hidden(x)

        x, _ = self.lstm2(x)
        x = self.dropout_hidden(x)

        # Dense
        x = x.permute(1, 0 , 2).squeeze() #Ødelægger batch size 1?
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_hidden(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    
class PyTorchDatasetList(Dataset):

    def __init__(self, data, target, frame_idx, w_len, step, context_size):
        self.data = data
        self.target = target
        self.frame_idx = frame_idx
        self.w_len = w_len
        self.step = step
        self.context_size = context_size

    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, idx):
        length_context = self.context_size*step

        list_idx = self.frame_idx[idx]
        datapoint = self.data[list_idx[0]][list_idx[1]-length_context:list_idx[1]+self.w_len+length_context]
        datapoint = torch.from_numpy(datapoint.astype(np.float32)).unsqueeze(0)

        nr_frame_in_file = int((list_idx[1]-length_context)/self.step)
        target = self.target[list_idx[0]][nr_frame_in_file]
        target = torch.from_numpy(np.array(target)).type(torch.LongTensor)
        sample = (datapoint, target)

        return sample


if __name__ == '__main__':
    train_set = 'Aurora-2'

    test_set = 'B'
    noise_types = [1, 2, 3, 4]

    save_model = True
    save_results = True
    context_size = 16
    model_name = 'Third_draft\\CLDNN_context{}'.format(context_size)
    noisy_context = True
    SNR_list = [5]
    test_on_clean = False

    model_kwargs = {'dropout_rate_input': 0.2,
                    'dropout_rate_hidden': 0.5,
                    'out_channels1': 40,
                    'kernel_len1': 201,
                    'pool_size1': 40,
                    'out_channels2': 16,
                    'kernel_len2': 8,
                    'pool_size2': 3,
                    'lstm_hidden_units': 16,
                    'dense_hidden_units': 16,
                    'stride1': 2}

    parser = argparse.ArgumentParser(description='CLDNN')
    parser.add_argument('--path', type=str,
                        default='C:\\Users\\marti',
                        help='Path to the folder where the aurora2 folder is')
    parser.add_argument('--fs', type=int, default=8000,
                        help='Sample rate (Default: 8000)')
    parser.add_argument('--segment_time', type=float, default=0.035,
                        help='length of segment in time given in seconds (Default: 0.035)')
    parser.add_argument('--window_overlap', type=float, default=0.01,
                        help='window overlap size in seconds (Default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='size of minibatches (Default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (Default: 1024)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='max number of epochs to train (Default: 30')
    parser.add_argument('--lr', type=float, default=0.0016, metavar='LR',
                        help='learning rate (Default: 0.0016)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (Default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (Default: False)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (Default: 42)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='Validation loss computation interval (Default: 500)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience in early stopping (Default: 10)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    es_figname = model_name if save_model is True else False

    w_len = int(args.segment_time*args.fs)
    step = int(args.window_overlap*args.fs)

    # ========================================================================
    # IMPORT DATA
    # ========================================================================
    #### OBS HUSK AT RETTE TRAIN PATH TIL README
    train_path = args.path + "\\aurora2\\SPEECHDATA\\TRAIN_NOISY/"
    train_target_path = args.path + "\\aurora2\\Aurora2TrainSet-ReferenceVAD/"
    test_path = args.path + "\\aurora2\\SPEECHDATA\\"
    test_target_path = args.path + "\\aurora2\\Aurora2TestSet-ReferenceVAD/"

    train_data, train_target, test_data, test_target = P7.import_data(train_path, train_target_path,
                                                                      test_path, test_target_path,
                                                                      SNR_list, test_on_clean)    
    train_data, train_target = shuffle(train_data, train_target) #Shuffling train data
    if len(train_data) == 0:
        print('Directory error')

    #Add context to data
    train_data = P7.add_context(train_data, context_size, w_len, step, noisy=noisy_context)
    test_data = P7.add_context(test_data, context_size, w_len, step, noisy=noisy_context)

    #Determine indices of frames in data
    train_frame_idx = P7.frame_index_list(w_len, step, train_data, context_size)
    test_frame_idx = P7.frame_index_list(w_len, step, test_data, context_size)

    #Create Dataloaders
    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda is True:
        train_kwargs['pin_memory'] = True
    dset1 = PyTorchDatasetList(train_data, train_target, train_frame_idx, w_len, step, context_size)
    train_loader = torch.utils.data.DataLoader(dset1, **train_kwargs)

    test_kwargs = {'batch_size': args.test_batch_size}
    dset2 = PyTorchDatasetList(test_data, test_target, test_frame_idx, w_len, step, context_size)
    test_loader = torch.utils.data.DataLoader(dset2, **test_kwargs)


    #Subtrain and validation data sets
    subtrain_data, valid_data, subtrain_target, valid_target = train_test_split(train_data, train_target, test_size=0.2, random_state = args.seed)

    subtrain_frame_idx = P7.frame_index_list(w_len, step, subtrain_data, context_size)
    valid_frame_idx = P7.frame_index_list(w_len, step, valid_data, context_size)

    dset_subtrain = PyTorchDatasetList(subtrain_data, subtrain_target, subtrain_frame_idx, w_len, step, context_size)
    subtrain_loader = torch.utils.data.DataLoader(dset_subtrain, **train_kwargs)

    valid_kwargs = {'batch_size': 128}
    dset_valid = PyTorchDatasetList(valid_data, valid_target, valid_frame_idx, w_len, step, context_size)
    valid_loader = torch.utils.data.DataLoader(dset_valid, **valid_kwargs)


    #Targets used for ROC
    actuals = np.concatenate([tt[:-1] for tt in test_target]) #One less target pr. file since 35ms-25ms = 10ms
    actuals_valid = np.concatenate([tv[:-1] for tv in valid_target])


    # ========================================================================
    # Neural Network Training and Testing
    # ========================================================================
    ### Train and Test Model ###
    print('\nPre-training:\n')
    model = CLDNN(w_len, step, context_size, **model_kwargs).to(device)
    init_params = copy.deepcopy(model.state_dict()) #Kopierer initial parameters
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) #Exponential learning rate decay

    ### Early stopping ###
    opt_upd, upd_epoch = P7.early_stopping(args, model, device, optimizer, scheduler,
                                           subtrain_loader, valid_loader, test_loader,
                                           actuals_valid, actuals, figname=es_figname)

    ### Re-training ###
    print('\nRe-training:\n')
    model = CLDNN(w_len, step, context_size, **model_kwargs).to(device) #Reset model parameters
    model.load_state_dict(init_params) #Same parameter initialisation as for early stopping
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    updates_counter = 0
    epoch = 1
    while updates_counter < opt_upd:
        updates_counter = P7.early_stopping_retrain(args, model, device,
                                                    train_loader, optimizer, epoch,
                                                    opt_upd, updates_counter,
                                                    scheduler, upd_epoch)
        epoch += 1
        if updates_counter < opt_upd:
            _, y_hat = P7.test(model, device, test_loader)
            print('AUC: {}\n'.format(P7.roc(actuals, y_hat, nr_points=51)[2]))
    print('LR: {}\n'.format(scheduler.get_last_lr()))

    ### Test time ###
    if use_cuda is True:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ACC, y_hat = P7.test(model, device, test_loader)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        test_time = start.elapsed_time(end) / 1000 #GPU time in seconds
        print('{:.5g} seconds'.format(test_time))
    else:
        time1 = time()
        ACC, y_hat = P7.test(model, device, test_loader)
        time2 = time()
        test_time = time2-time1 #CPU time in seconds
        print('Test time: {:.5g} seconds'.format(test_time))

    fp, tp, AUC, ACC_list, threshold = P7.roc(actuals, y_hat)

    if save_model is True:
        torch.save(model.state_dict(), "NeuralNetworks\\" + model_name + ".pt")
        P7.save_obj(model_kwargs, model_name, 'Model_Dictionaries/')

    PARAMS = sum(p.numel() for p in model.parameters())
    print('Area Under the Curve: {:.5g}'.format(AUC))
    print('Number of parameters: {}'.format(PARAMS))

    if save_results is True:
        P7.plot_roc(fp, tp, figname = model_name + '_ROC')
        results = {'SNR': SNR_list,
                   'context_size': context_size,
                   'PARAMS': PARAMS,
                   'AUC': AUC,
                   'ACC': ACC,
                   'ACC_list': ACC_list,
                   'Threshold': threshold,
                   'Test_Time': test_time,
                   'Learning_rate': args.lr,
                   'fp_tp': (fp, tp)}
        P7.save_obj(results, model_name, 'Results/')
    else:
        P7.plot_roc(fp, tp)