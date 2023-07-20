import numpy as np
import torch
from torch_geometric_temporal.signal import temporal_signal_split
import torch.nn.functional as F
import torch.nn as nn
from dataloader.MyDataLoader import MyDataLoader
from dataloader.MyDataLoader_large import MyDataLoader_large
from model.BA3tgcn import BA3TGCN, BA3TGCN2
from utils.Accumulator import Accumulator
from torch.utils.data import dataloader
from utils.Loss import batch_crossentropy,focal_Loss

def attack_accuracy(batchsize, y_hat, y):
    batch_size = batchsize
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[2] > 1:
        """axis=1：每一行的列的值进行比较，最大的作为这一行的y_hat"""

        y_hat_2 = y_hat


    correct_sum = 0

    for i in range(batch_size):
        if(batchsize == 1):
            y = torch.unsqueeze(y, 0)

        y_true_label = y[i, :]
        for j in y_true_label - 1:
            for k in range(y_hat.size(1)):
                if (y_hat[i, k, j] > 0.01):
                    correct_sum = correct_sum + 1


    return float(correct_sum)

def attack_predict(test_dataset, test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break;

    model_best = TemporalGNN(node_features=1,batchsize=512,TrainorPredict=0, FullAttention=True)
    model_data = torch.load('module_best_C_L_G.pth')
    model_best.load_state_dict(model_data['model_dict_C_L'])
    model_best = model_best.to(device)
    model_best.eval()

    labels = []
    preds = []
    y_hats = []
    metric = Accumulator(2)
    softmax_func = nn.Softmax(dim=2)
    for inputs, label in test_loader:
        y_hat = model_best(inputs, static_edge_index)
        label = label.squeeze()
        label = label - 1#[1,16]

        y_hat = softmax_func(y_hat)
        y_hat = torch.squeeze(y_hat)


        metric.add(attack_accuracy(512, y_hat, label), label.numel())

        pred = y_hat.argmax(axis=2)

        y_hats.append(y_hat)
        labels.append(label)
        preds.append(pred)

        metric.add(accuracy('pre', 512, y_hat, label), label.numel())

    test_acc = metric[0] / metric[1]
    print('test ACC:', test_acc)

