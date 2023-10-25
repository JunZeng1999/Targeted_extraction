import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import scipy.io as scio
from tqdm import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model.classifier_model import Classifier


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    batch_size = 1

    # get test data set
    testdatafile = 'train_data_test.mat'
    testData = scio.loadmat(testdatafile)

    # read parameter
    test_para = testData['signal']
    test_class = testData['class']
    test_num = len(test_para)

    testDataset = test_para.astype(np.float32)
    test_class = test_class.astype(np.float32)
    # numpy to tensor
    test_x, test_y = torch.from_numpy(testDataset), torch.from_numpy(test_class)

    test_data_set = TensorDataset(test_x, test_y)
    test_loader = DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=True, drop_last=False)

    print("using {} peaks for validation.".format(test_num))

    # create model
    model = Classifier()

    # load train weights
    train_weights = "./save_weights/Classifier99.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device))
    model.to(device)

    # test
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_signal, test_labels = test_data
            test_signals = test_signal.reshape(-1, 1, 128)
            test_labels = test_labels.reshape(test_labels.shape[0])
            outputs1 = model(test_signals.to(device))
            predict = torch.max(outputs1, dim=1)[1]
            acc += torch.eq(predict, test_labels.to(device)).sum().item()

    test_accurate = acc / test_num
    print('val_accuracy: %.3f' % test_accurate)
    print('Finished Training')


if __name__ == "__main__":
    main()