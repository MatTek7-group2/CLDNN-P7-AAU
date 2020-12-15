# -*- coding: utf-8 -*-
"""

Created on Tuesday Dec. 15 14:03:53 2020

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
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.io import wavfile
from sklearn.utils import shuffle
import pickle
import torch
import torch.nn.functional as F


# =============================================================================
# IMPORT AURORA-2 DATA 
# =============================================================================

def data_dir(path):
    """

    Parameters
    ----------
    path : string
        Path to the aurora2 folder.

    Returns
    -------
    files08 : list
        List of strings containing the path
        to the files in the dataset.
        This is used as input to load_data().

    """
    files08 = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".08"):
                files08.append(os.path.join(root, file))
    return files08


def target_dir(path):
    """

    Parameters
    ----------
    path : string
        Path to the folder where the aurora2 folder.

    Returns
    -------
    files08 : list
        List of strings containing the path
        to the target files in the dataset.
        This is used as input to load_target().

    """
    files_target = []
    for root, dirs, files in os.walk(path):
        for file in files:
            files_target.append(os.path.join(root, file))

    return files_target


def file_name(data_dir, target_dir):
    """

    Parameters
    ----------
    data_dir : list
        List of strings containing the path
        to the files in the dataset.
    target_dir : list
        List of strings containing the path
        to the files in the target dataset.

    Returns
    -------
    data_name : list
        List of strings containing name of the file.
    target_name : list
        List of strings containing name of the file.

    Takes a list of file paths and returns the file names.
    It is used to sort the files when using the Aurora-2 data.

    """
    data_name = []
    target_name = []

    if len(data_dir) != len(target_dir):
        print('Data and Target length mismatch: {} and {}'.format(len(data_dir), len(target_dir)))

    for a, b in zip(data_dir, target_dir):
        a = (a.split('/')[-1]).split('.')
        a = a[0].split('\\')[-1]
        b = b.split('/')[-1]
        data_name.append(a)
        target_name.append(b)
    return data_name, target_name


def file_name_data(data_dir):
    """

    Parameters
    ----------
    data_dir : list
        List of strings containing the path
        to the files in the dataset.

    Returns
    -------
    data_name : list
        List of strings containing name of the file.

    Takes a list of file paths and returns the file names.
    It is used to sort the files when using the Aurora-2 data.

    """
    data_name = []

    for a in data_dir:
        a = (a.split('/')[-1]).split('.')
        a = a[0].split('\\')[-1]
        data_name.append(a)
    return data_name


def file_name_target(target_dir):
    """

    Parameters
    ----------
    target_dir : list
        List of strings containing the path
        to the files in the target dataset.

    Returns
    -------
    target_name : list
        List of strings containing name of the file.

    Takes a list of file paths and returns the file names.
    It is used to sort the files when using the Aurora-2 data.

    """
    target_name = []

    for b in target_dir:
        b = b.split('/')[-1]
        target_name.append(b)
    return target_name


def sort_list(list1, list2):
    """

    Parameters
    ----------
    list1 : list
        List to be sorted.
    list2 : list
        List to sort according to.

    Returns
    -------
    z : list
        Sorted list.

    """

    if len(list1) != len(list2):
        print('List length mismatch: {} and {}'.format(len(list1), len(list2)))

    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


def same_file(data_dir, target_dir, noisy = True):
    """

    Parameters
    ----------
    data_dir : list
        List of strings containing the path
        to the files in the dataset.
    target_dir : list
        List of strings containing the path
        to the files in the target dataset.
    noisy : bool, optional
        Input option to say if the clean dataset
        is used or the noisy dataset.
        The default is True.

    Check if all the data and target files match.

    """
    try:
        assert len(data_dir) == len(target_dir)
        is_not_same = 0
        for a, b in zip(data_dir, target_dir):
            if not noisy:
                a = (a.split('/')[-1]).split('.')
                b = b.split('/')
            elif noisy:
                a = (a.split('/')[-1]).split('.')
                a = a[0].split('\\')[-1]
                b = b.split('/')[-1]
            if a != b:
                is_not_same += 1
        if is_not_same != 0:
            print('Error1:\n{} file names are different'.format(is_not_same))
    except AssertionError:
        print('Error2:\nLength of data {}\nLength of target {}'.format(len(data_dir), len(target_dir)))


def load_data(files):
    """

    Parameters
    ----------
    files : list or string
        List of strings containing the path to the files
        in the dataset or just a single string.

    Returns
    -------
    data : list
        List of ndarrays. Each array corresponding to a sound file.

    """
    if type(files) == str:
        with open(files, 'rb') as file:
            a = file.read()
            data = np.frombuffer(a, '>i2')
    elif type(files) == list:
        data = []
        for filename in files:
            with open(filename, 'rb') as file:
                a = file.read()
                temp = np.frombuffer(a, '>i2')
                data.append(temp)
    else:
        print('Type Error')
    return data


def load_target(files):
    """

    Parameters
    ----------
    files : list or string
        List of strings containing the path to the files
        in the target dataset or just a single string.

    Returns
    -------
    target : list
        List containing the ground thruths (labels/targets).

    """
    if type(files) == str:
        with open(files, 'rb') as file:
            a = file.readlines()
            target = [int(a[i]) for i in range(len(a))]
    elif type(files) == list:
        target = []
        for filename in files:
            with open(filename, 'r') as file:
                a = file.readlines()
                a = [int(a[i]) for i in range(len(a))]
                target.append(a)

    return target


def SNR_data(test_path, SNR_list, include_clean=False, test_set='B',
             noise_types=[1, 2, 3, 4]):
    """

    Parameters
    ----------
    test_path : str
        Path to the test data.
    test_target_path : str
        Path to the targets in the test data.
    SNR : list
        List of ints in the set {-5, 0, 5, 10, 15, 20}.
        The SNR levels of the data to load.

    Returns
    -------
    test_dir : list
        List of strings giving the paths to the files in the test set.
    test_target_dir : list
        List of strings giving the paths to the targets in the test set.

    Search through the test data and find the data which has SNR as
    stated in SNR_list and also find the clean data if includ_clean is true.

    """
    test_dir_full = []
    for SNR in SNR_list:
        test_list = ['TEST' + test_set +'\\N{}_SNR{}/'.format(i, SNR) for i in noise_types]
        test_dir = []
        for p in test_list:
            test_dir.append(data_dir(test_path + p))
        test_dir_full.append([item for sublist in test_dir for item in sublist])

    if include_clean is True:
        clean_dir = []
        test_list = ['TEST' + test_set +'\\Clean{}/'.format(i) for i in noise_types]
        for p in test_list:
            clean_dir.append(data_dir(test_path + p))
        clean_dir_full = [item for sublist in clean_dir for item in sublist]
        test_dir_full.append(clean_dir_full)

    test_dir = [item for sublist in test_dir_full for item in sublist]

    return test_dir


def import_test_data(test_path, test_target_path,
                     SNR_list=[-5, 0, 5, 10, 15, 20], include_clean=False,
                     test_set='B', noise_types=[1, 2, 3, 4]):
    """

    Parameters
    ----------
    test_path : str
        Path to the test data.
    test_target_path : str
        Path to the targets in the test data.
    SNR : list, optional (Default = [-5, 0, 5, 10, 15, 20])
        List of ints in the set {-5, 0, 5, 10, 15, 20}.
        The SNR levels of the data to load.
    include_clean : bool, optional (Default: False)
        If True include the clean test data.
        If False do not include the clean test data.

    Returns
    -------
    test_data : list
        Test data.
    test_target : list
        Test targets.

    The function finds the names of each of the files in the given directory,
    i.e. data of specified SNR levels belonging to the specified test set.
    Then it sorts the files such that the data and the targets match and
    checks that this is done correctly. Then the data is imported
    into a list of arrays where each array contains the data for a
    given file. Finally, the targets are import into a list of lists containing the 
    targets for each file.

    This function is used for the Aurora-2 Test Sets.

    """
    #Only include certain SNR levels in the test data
    test_dir = SNR_data(test_path, SNR_list, include_clean, test_set, noise_types)

    target_dir_full = target_dir(test_target_path)
    test_dir_name = file_name_data(test_dir)
    test_target_dir_name = file_name_target(target_dir_full)

    test_target_dir_temp = []
    for i in test_dir_name:
        if i in test_target_dir_name:
            test_target_dir_temp.append(test_target_path + '/' + i)

    if len(SNR_list) == 1 and include_clean is False:
        test_target_dir = test_target_dir_temp
    elif len(SNR_list) > 1 and include_clean is False:
        test_target_dir = []
        for i in range(len(SNR_list)):
            test_target_dir.append(test_target_dir_temp)
        test_target_dir = [item for sublist in test_target_dir for item in sublist]
    elif len(SNR_list) == 0 and include_clean is True:
        test_target_dir = test_target_dir_temp
    elif len(SNR_list) > 0 and include_clean is True:
        test_target_dir = []
        for i in range(len(SNR_list)+1):
            test_target_dir.append(test_target_dir_temp)
        test_target_dir = [item for sublist in test_target_dir for item in sublist]
    else:
        print('Load Test Data Error')

    #Sort to make the data match the targets
    test_dir_name2, test_target_dir_name2 = file_name(test_dir, test_target_dir)
    sorted_test = sort_list(test_dir, test_dir_name2)
    sorted_test_target = sort_list(test_target_dir, test_target_dir_name2)

    #Check that the files in the data and the target match
    same_file(sorted_test, sorted_test_target)

    #Load data
    test_data = load_data(sorted_test)
    test_target = load_target(sorted_test_target)

    if len(test_data) == 0 or len(test_target) == 0:
        print('Test Data Directory error')

    return test_data, test_target


def import_train_data(train_path, train_target_path):
    """

    Parameters
    ----------
    train_path : str
        Path to the training data.
    train_target_path : str
        Path to the targets in the training data.

    Returns
    -------
    train_data : list
        Training data.
    train_target : list
        Training targets.

    The function finds the names of each of the files in the given directory.
    Then it sorts the files such that the data and the targets match and
    checks that this is done correctly. Then the data is imported into a
    list of arrays where each array contains the data for a given file.
    Finally, the targets are import into a list of lists containing the 
    targets for each file.

    This function is used for the Aurora-2 Training Set.

    """
    #Directory of data files
    train_dir = data_dir(train_path)
    train_target_dir = target_dir(train_target_path)

    #Sort the files so the data matches the target
    train_dir_name, train_target_dir_name = file_name(train_dir, train_target_dir)
    sorted_train = sort_list(train_dir, train_dir_name)
    sorted_train_target = sort_list(train_target_dir, train_target_dir_name)
    same_file(sorted_train, sorted_train_target)

    train_data = load_data(sorted_train)
    train_target = load_target(sorted_train_target)

    if len(train_data) == 0 or len(train_target) == 0:
        print('Train Data Directory error')

    #Shuffling train data
    train_data, train_target = shuffle(train_data, train_target)

    return train_data, train_target


def add_context(data, context_size, w_len, step, noisy=True):
    """

    Parameters
    ----------
    data : list
        List of arrays. Each array containing a waveform.
    context_size : int
        How many adjacent frames to include in the context.
    w_len : int
        Number of samples in frame without context.
    step : int
        Number of samples between frames.
    noisy : bool, optional (Default: True)
        Option to repeat the beginning and end of the file when adding context
        or zero-pad. If noisy is True then the waveform is repeated.

    Returns
    -------
    context_data : list
        List of arrays. Each array containing a waveform,
        but with silence appended in each end to allow for
        the use of context in the neural network.

    """
    if context_size == 0:
        return data
    else:
        context_data = []
        for datapoint in data:
            if noisy is False:
                length_zeros = step*context_size
                context_datapoint = np.hstack((np.zeros(length_zeros), datapoint, np.zeros(length_zeros)))
            elif noisy is True:
                begin = np.repeat(datapoint[:step], context_size)
                end = np.repeat(datapoint[-step:], context_size)
                context_datapoint = np.hstack((begin, datapoint, end))
            else:
                print('Context function input error')
            context_data.append(context_datapoint)
        return context_data


def frame_index_list(w_len, step, data, context_size = 0):
    """

    Parameters
    ----------
    w_len : int
        Window length.
    step : int
        Step length (overlap).
    data : list
        List of ndarrays. Each array corresponding to a sound file.
        Output of load_data().
    context_size : int, optional (Default: 0)
        How many adjacent frames to include in the context.

    Returns
    -------
    flat_list : list
        List where each element is a list of length two
        where the first entry is the index of which file
        the frame is in and the second entry is the index
        in the waveform where the frame starts.

    This function is used to generate a list of indices which
    are used to load datapoints in from data during training
    and testing of the neural network (see the class Dataset).

    """
    frame_idx = []
    for j, dp in enumerate(data): #loop over files
        if context_size != 0:
            len_in_ends = context_size*step
            dp_true = dp[len_in_ends:-len_in_ends]
            nr_of_segments = int(np.floor((len(dp_true) - w_len + step)/step))
            temp_idx = [[j, len_in_ends + i*step] for i in range(nr_of_segments)]
        else:
            nr_of_segments = int(np.floor((len(dp) - w_len + step)/step))
            temp_idx = [[j, i*step] for i in range(nr_of_segments)]
        frame_idx.append(temp_idx)
    flat_list = [item for sublist in frame_idx for item in sublist]
    return flat_list


# =============================================================================
# IMPORT APOLLO-11 DATA 
# =============================================================================

def data_dir_APOLLO(path, type_):
    """

    Parameters
    ----------
    path : string
        Path to the Fearless_2019 folder.
    type_ : string
        Options: 'train', 'test'. What type of data to import.

    Returns
    -------
    data_files : list
        List of strings containing the path
        to the files in the dataset.
        This is used as input to load_data_APOLLO().

    """
    number_file = path + '\\Fearless_2019\\Fearless_Steps\\Data\\' + type_ + '_path.txt'
    with open(number_file, 'r') as file:
        a_init = file.readlines()
        a = [int(a_i.split('.')[0]) for a_i in a_init]

    folder = path + "\\Fearless_2019\\Fearless_Steps\\Data\\Audio\\Tracks\\Dev"
    data_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            number = int(file.split('_')[-1].split('.')[0])
            if number in a:
                data_files.append(os.path.join(root, file))
    return data_files


def target_dir_APOLLO(path, type_):
    """

    Parameters
    ----------
    path : string
        Path to the Fearless_2019 folder.
    type_ : string
        Options: 'train', 'test'. What type of data to import.

    Returns
    -------
    files_target : list
        List of strings containing the path
        to the target files in the dataset.
        This is used as input to load_target().

    """
    number_file = path + '\\Fearless_2019\\Fearless_Steps\\Data\\' + type_ + '_target_path.txt'
    with open(number_file, 'r') as file:
        a_init = file.readlines()
        a = [int(a_i.split('.')[0]) for a_i in a_init]

    folder = path + "\\Fearless_2019\\Fearless_Steps\\Data\\Transcripts\\SAD\\Dev"
    files_target = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith("labels"):
                number = int(file.split('_')[-1].split('.')[0])
                if number in a:
                    files_target.append(os.path.join(root, file))
    return files_target


def load_data_APOLLO(files):
    """

    Parameters
    ----------
    files : list or string
        List of strings containing the path to the files
        in the dataset or just a single string.

    Returns
    -------
    data : list
        List of ndarrays. Each array corresponding to a sound file.

    """
    if type(files) == str:
        data = wavfile.read(files)[1]
    elif type(files) == list:
        data = []
        for filename in files:
            data.append(wavfile.read(filename)[1])
    else:
        print('Type Error')
    return data


# =============================================================================
# TRAIN AND TEST
# =============================================================================


def train(args, model, device, train_loader, optimizer, epoch):
    """

    Parameters
    ----------
    args : Namespace
    model : PyTorch model class
    device : device
    train_loader : Dataloader
    optimizer : PyTorch optmiser
    epoch : int
        Number of epochs to run during training.

    Trains the neural network without early stopping.

    """
    model.train()
    interval_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        target = target.squeeze()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        interval_loss += loss.item()
        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), interval_loss/args.log_interval))
            interval_loss = 0


def test(model, device, test_loader):
    """

    Parameters
    ----------
    model : PyTorch model class
    device : device
    test_loader : Dataloader

    Returns
    -------
    ACC : float
        Accuracy of testing.
    y_hat : ndarray
        Model scores.

    Evaluates the neural network.

    """
    model.eval()
    test_loss = 0
    correct = 0
    y_hat = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            target = target.squeeze()

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            y_hat.extend(np.exp(output.cpu().numpy()))

    test_loss /= len(test_loader.dataset)
    ACC = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * ACC))

    y_hat = np.array(y_hat)
    return ACC, y_hat


# =============================================================================
# EARLY STOPPING
# =============================================================================


def validation(model, device, valid_loader):
    """

    Parameters
    ----------
    model : PyTorch model class
    device : device
    valid_loader : Dataloader
        Dataloader for the validation set.

    Returns
    -------
    valid_loss : float
        The validation loss.
    y_hat : ndarray
        Output of the neural network for the validation set.

    Evaluates the validation set. This is done during early stopping.

    """
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for valid, valid_target in valid_loader:
            valid, valid_target = valid.to(device), valid_target.to(device)

            valid_target = valid_target.squeeze()
            output_valid = model(valid)
            valid_loss += F.nll_loss(output_valid, valid_target, reduction='sum').item()  # sum up batch loss

    valid_loss /= len(valid_loader.dataset)
    model.train()
    return valid_loss


def end_of_early_stopping(train_loss_plot, valid_loss_plot,
                          updates_pr_pretrain_epoch, updates_counter,
                          plot_validation = True, figname = 'Validation_plot'):
    """

    Parameters
    ----------
    train_loss_plot : list
        List of the training loss for plotting.
    valid_loss_plot : list
        List of the validation loss for plotting.
    updates_pr_pretrain_epoch : int
        Number of parameters updates per epoch over subtrain.
    updates_counter : int
        Number of parameters updates done.
    plot_validation : bool, optional (Default: True)
        To plot or not to plot.
    figname : string or boolean, optional (Default: 'Validation_plot')
        If it is a string then it is used as the figure
        name to save the validation plot.

    This code is run at the end of early stopping pre-training.

    """
    if plot_validation:
        plt.style.use('ggplot')
        xaxis_list = np.linspace(0, updates_counter, len(train_loss_plot))/updates_pr_pretrain_epoch
        plt.plot(xaxis_list, train_loss_plot, color='orange', label='Training loss')
        plt.plot(xaxis_list, valid_loss_plot, color='magenta', label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if figname is not False:
            plt.savefig('Figures\\' + figname + '_ES.png')
        plt.show()


def early_stopping(args, model, device, optimizer, scheduler,
                   subtrain_loader, valid_loader, test_loader,
                   actuals_valid, actuals, figname = 'Validation_plot'):
    """

    Parameters
    ----------
    args : Namespace
    model : PyTorch model class
    device : device
    optimizer : PyTorch optimiser
    scheduler : PyTorch scheduler
    subtrain_loader : Dataloder
        Subtrain dataset.
    valid_loader : Dataloader
        Validation dataset.
    test_loader : Dataloader
        Test dataset.
    actuals_valid : ndarray
        Targets for the validation set.
    actuals : ndarray
        Targets for the test set.
    figname : string or boolean, optional (Default: 'Validation_plot')
        If it is a string then it is used as the figure
        name to save the validation plot.

    Returns
    -------
    optim_updates : int
        The optimal number of parameter updates during training
        as determined by early stopping.
    updates_pr_pretrain_epoch : int
        Number of paramer updates per epoch in the subtrain dataset.

    Determines the number of parameter updates which should be performed
    during training using early stopping.

    """
    updates_counter = 0
    min_valid_loss = 0
    no_increase_counter = 0
    optim_updates = 0
    updates_pr_pretrain_epoch = 0
    train_loss_plot = []
    valid_loss_plot = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        interval_loss = 0
        for batch_idx, (data, target) in enumerate(subtrain_loader):
            data, target = data.to(device), target.to(device)

            target = target.squeeze()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            interval_loss += loss.item()

            if batch_idx % args.log_interval == 0 and batch_idx != 0:
                valid_loss = validation(model, device, valid_loader)

                if min_valid_loss == 0:
                    min_valid_loss = valid_loss
                elif valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    optim_updates = updates_counter
                    no_increase_counter = 0
                else:
                    no_increase_counter += 1

                train_loss_plot.append(interval_loss/args.log_interval)
                valid_loss_plot.append(valid_loss)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMin. Val. Loss: {:.6f}\tVal. Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(subtrain_loader.dataset),
                    100. * batch_idx / len(subtrain_loader), interval_loss/args.log_interval,
                    min_valid_loss, valid_loss))
                interval_loss = 0

                if no_increase_counter == args.patience:
                    end_of_early_stopping(actuals_valid, train_loss_plot, valid_loss_plot,
                                          updates_pr_pretrain_epoch, updates_counter,
                                          figname=figname)
                    return optim_updates, updates_pr_pretrain_epoch

            updates_counter += 1
        scheduler.step()
        print('')

        if no_increase_counter == args.patience:
            break
        if epoch == 1:
            updates_pr_pretrain_epoch = updates_counter

    end_of_early_stopping(actuals_valid, train_loss_plot, valid_loss_plot,
                          updates_pr_pretrain_epoch, updates_counter,
                          figname=figname)

    return optim_updates, updates_pr_pretrain_epoch


def early_stopping_retrain(args, model, device, train_loader, optimizer, epoch,
                           optim_updates, updates_counter, scheduler, updates_pr_pretrain_epoch):
    """

    Parameters
    ----------
    args : Namespace
    model : PyTorch model class
    device : device
    train_loader : Dataloader
        Training dataset.
    optimizer : PyTorch optimiser
    epoch : int
        The current training epoch.
    optim_updates : int
        The optimal number of parameter updates during training
        as determined by early stopping.
    updates_counter : int
        Counter for the number of parameter updates.
    scheduler : PyTorch scheduler
    updates_pr_pretrain_epoch : int
        Number of paramer updates per epoch in the subtrain dataset.

    Returns
    -------
    updates_counter : int
        Counter for the number of parameter updates.

    Re-trains the neural network after early stopping pre-training.
    The learning rate is decayed after each updates_pr_pretrain_epoch 
    parameter updates and training is done for optim_updates
    parameter updates.

    """
    model.train()
    interval_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        target = target.squeeze()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        interval_loss += loss.item()
        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), interval_loss/args.log_interval))
            interval_loss = 0

        if updates_counter == optim_updates:
            return updates_counter
        updates_counter += 1
        if updates_counter % updates_pr_pretrain_epoch == 0:
            scheduler.step()

    return updates_counter


# =============================================================================
# ROC
# =============================================================================


def roc(target, y_hat, nr_points = 1000):
    """

    Parameters
    ----------
    target : ndarray
        Array containing the ground thruths (labels/targets).
    y_hat : ndarray
        Model scores.
    nr_points : int, optional (Default: 1000)
        Resolution of the ROC curve.

    Returns
    -------
    fp : ndarray
        False positive rate.
    tp : ndarray
        True positive rate.
    roc_auc : float
        Area under the ROC curve (AUC).
    acc : ndarray
        Accuracy array for the different thresholds.
    q : ndarray
        Array of thresholds.

    """
    q = np.linspace(0, 1, nr_points, endpoint=True)
    fp = np.zeros(nr_points)
    tp = np.zeros(nr_points)
    acc = np.zeros(nr_points)
    N = np.shape(y_hat)[0]

    for i in range(nr_points):
        # Model prediction with varying thresholds
        q_array = np.ones(N)*q[i]
        temp = np.array([q_array, y_hat[:, 1]])
        pred = np.argmax(temp, axis=0)

        # Unnormalised rates (see worksheet 10)
        tp_total = np.argwhere((pred==1) & (target==1)).size
        fn_total = np.argwhere((pred==0) & (target==1)).size
        tn_total = np.argwhere((pred==0) & (target==0)).size
        fp_total = np.argwhere((pred==1) & (target==0)).size

        # Normalising the rates
        fp[i] = fp_total/(tn_total+fp_total)
        tp[i] = tp_total/(tp_total+fn_total)
        acc[i] = (tp_total + tn_total)/N
    roc_auc = auc(fp, tp)
    return fp, tp, roc_auc, acc, q


def plot_roc(fp, tp, figname = False):
    """

    Parameters
    ----------
    fp : ndarray
        False positive rate.
    tp : ndarray
        True positive rate.
    figname : string or boolean, optional (Default: False)
        If it is a string then it is used as the figure
        name to save the ROC plot.

    """

    roc_auc = auc(fp, tp)
    plt.style.use('ggplot')
    plt.plot(fp, tp, color = 'magenta', label='ROC curve (AUC = {:.4g})'.format(roc_auc))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if figname is not False:
        plt.savefig(figname + '.png')
    plt.show()


# =============================================================================
# SAVE AND LOAD DICTIONARIES
# =============================================================================


def save_obj(obj, name, folder = 'Model_Dictionaries/'):
    """

    Parameters
    ----------
    obj : dict
        Dictionary.
    name : str
        Name of the file to save the dictionary in.
    folder : str, optional (Default: 'Model_Dictionaries/')
        Name of the folder to save the dictionary to.

    """
    with open(folder + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name, folder = 'Model_Dictionaries/'):
    """

    Parameters
    ----------
    name : str
        Name of the file to load the dictionary from.
    folder : str, optional (Default: 'Model_Dictionaries/')
        Name of the folder to laod the dictionary from.

    Returns
    -------
    dict
        Dictionary of model parameters.

    """
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)

