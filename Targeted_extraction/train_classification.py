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
    batch_size = 32
    traindatafile = 'train_data_train.mat'
    trainData = scio.loadmat(traindatafile)

    # read parameter
    train_para = trainData['signal']
    train_class = trainData['class']
    train_num = len(train_para)

    TrainDataset = train_para.astype(np.float32)
    train_class = train_class.astype(np.float32)
    # numpy to tensor
    train_x, train_y = torch.from_numpy(TrainDataset), torch.from_numpy(train_class)

    train_data_set = TensorDataset(train_x, train_y)

    train_loader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True, drop_last=False)

    # get val data set
    valdatafile = 'train_data_val.mat'
    valData = scio.loadmat(valdatafile)

    # read parameter
    val_para = valData['signal']
    val_class = valData['class']
    val_num = len(val_para)

    valDataset = val_para.astype(np.float32)
    val_class = val_class.astype(np.float32)
    # numpy to tensor
    val_x, val_y = torch.from_numpy(valDataset), torch.from_numpy(val_class)

    val_data_set = TensorDataset(val_x, val_y)
    val_loader = DataLoader(dataset=val_data_set, batch_size=batch_size, shuffle=True, drop_last=False)

    print("using {} peaks for training, {} peaks for validation.".format(train_num, val_num))

    net = Classifier(init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    epochs = 100
    # save_path = 'Classifier.pth'
    best_acc = 0.9
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, (data, target) in enumerate(train_bar):
            signal = data.reshape(-1, 1, 128)
            labels = target.reshape(target.shape[0])
            optimizer.zero_grad()
            outputs = net(signal.to(device))
            loss = loss_function(outputs.float(), labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_signal, val_labels = val_data
                val_signals = val_signal.reshape(-1, 1, 128)
                val_labels = val_labels.reshape(val_labels.shape[0])
                outputs1 = net(val_signals.to(device))
                predict = torch.max(outputs1, dim=1)[1]
                acc += torch.eq(predict, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            torch.save(net.state_dict(), "./save_weights/Classifier{}.pth".format(epoch))

    print('Finished Training')


if __name__ == "__main__":
    main()
