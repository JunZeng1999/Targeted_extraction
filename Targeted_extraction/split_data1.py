import os
from shutil import copy, rmtree
import random
import scipy.io as scio
from scipy.io import savemat
import numpy as np


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    # random.seed(0)  # fix the seed for shuffle.
    # random.shuffle(dataset)
    n = int(ratio * len(dataset))
    n_n = int(n/2)
    m = int((1-ratio-ratio) * len(dataset))
    dataset1 = dataset[:n_n]
    dataset1_1 = dataset[n + m + n_n:]
    dataset1_zong = np.append(dataset1, dataset1_1, axis=0)
    dataset2 = dataset[n_n:n]
    dataset2_2 = dataset[n + m:n + m + n_n]
    dataset2_zong = np.append(dataset2, dataset2_2, axis=0)
    dataset3 = dataset[n:n+m]
    return dataset1_zong, dataset2_zong, dataset3


def main():
    split_rate = 0.15

    datafile = 'train_data.mat'
    trainData = scio.loadmat(datafile)

    # read parameter
    data_para = trainData['signal']
    data_class = trainData['class']

    # random sampling
    dataset_val, dataset_test, dataset_train = split_dataset(data_para, split_rate)
    dataset_val1, dataset_test1, dataset_train1 = split_dataset(data_class, split_rate)
    dataset_val1 = np.double(dataset_val1)
    dataset_test1 = np.double(dataset_test1)
    dataset_train1 = np.double(dataset_train1)

    savemat('train_data_val.mat', {'signal': dataset_val, 'class': dataset_val1})
    savemat('train_data_test.mat', {'signal': dataset_test, 'class': dataset_test1})
    savemat('train_data_train.mat', {'signal': dataset_train, 'class': dataset_train1})
    print("processing done!")


if __name__ == '__main__':
    main()
