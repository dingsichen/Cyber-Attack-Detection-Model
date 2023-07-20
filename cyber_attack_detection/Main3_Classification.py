'''分类'''
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

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, batchsize, TrainorPredict, FullAttention):
        super(TemporalGNN, self).__init__()
        self.tgnn = BA3TGCN2(in_channels=node_features,
                           out_channels=128,
                           periods=8,
                           batch_size=batchsize,
                           TrainorPredict=TrainorPredict,
                           FullAttention=FullAttention)
        self.linear1 = torch.nn.Linear(128, 32)
        self.linear2 = torch.nn.Linear(32, 6)
        # self.linear3 = torch.nn.Linear(32, 8)



    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        # h = F.leaky_relu(h)
        h = F.mish(h)
        # h = F.relu(h)
        h = self.linear1(h)
        # h = F.leaky_relu(h)
        h = F.mish(h)
        # h = F.relu(h)
        h = self.linear2(h)

        return h

def load_split():
    '''构建dataset'''
    loader = MyDataLoader_large()

    dataset = loader.get_dataset(num_timesteps_in=8, num_timesteps_out=1)  # 输入8个样本点 预测未来1个点的值

    # print(next(iter(dataset)))
    # print("Number of samples: ", len(set(dataset)))

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    # print("Number of train_dataset samples: ", len(set(train_dataset)))
    # print("Number of test_dataset samples: ", len(set(test_dataset)))

    '''构建dataloader'''
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # cuda
    shuffle = False
    batch_size_train = 512
    batch_size_test = 512

    train_input = np.array(train_dataset.features)  # (31989, 6, 1, 6)
    train_target = np.array(train_dataset.targets)  # (31989, 6, 1)

    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)

    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size_train, shuffle=shuffle,
                                               drop_last=True)

    test_input = np.array(test_dataset.features)  # (, 5 ,2, 3)
    test_target = np.array(test_dataset.targets)  # (, 5, 2)

    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size_test, shuffle=False, drop_last=True)


    return train_dataset, test_dataset, train_loader, test_loader

def accuracy(tr_or_pre, batchsize, y_hat, y):
    batch_size = batchsize
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[2] > 1:
        """axis=1：每一行的列的值进行比较，最大的作为这一行的y_hat"""
        y_hat_2 = y_hat
        y_hat = y_hat.argmax(axis=2)


    # for i in range(batch_size):
    #     for j in range(y_hat.size(1)):
    #         y_hat_2[i, j, y_hat[i, j]] = -100
    #
    # y_hat_2 = y_hat_2.argmax(axis=2)


    correct_sum = 0
    cmp_list = [0,0,0,0,0,0]
    y_list = [0,0,0,0,0,0]
    for i in range(batch_size):
        if(batchsize == 1):
            y = torch.unsqueeze(y, 0)
        cmp = y_hat[i,:].type(y.dtype) == y[i,:]
        # cmp2 = y_hat_2[i, :].type(y.dtype) == y[i, :]
        # cmp = cmp1+cmp2

        # true_index = torch.nonzero(cmp).squeeze().cpu().numpy()
        if(tr_or_pre == 'pre'):
            for j in range(6):
                if(y[i, j] == 0):
                    y_list[0] = y_list[0] + 1
                if(y[i, j] == 1):
                    y_list[1] = y_list[1] + 1
                if(y[i, j] == 2):
                    y_list[2] = y_list[2] + 1
                if(y[i, j] == 3):
                    y_list[3] = y_list[3] + 1
                if(y[i, j] == 4):
                    y_list[4] = y_list[4] + 1
                if(y[i, j] == 5):
                    y_list[5] = y_list[5] + 1

                if (y_hat[i, j].type(y.dtype) == y[i, j] == 0):
                    cmp_list[0] = cmp_list[0] + 1
                if (y_hat[i, j].type(y.dtype) == y[i, j] == 1):
                    cmp_list[1] = cmp_list[1] + 1
                if (y_hat[i, j].type(y.dtype) == y[i, j] == 2):
                    cmp_list[2] = cmp_list[2] + 1
                if (y_hat[i, j].type(y.dtype) == y[i, j] == 3):
                    cmp_list[3] = cmp_list[3] + 1
                if (y_hat[i, j].type(y.dtype) == y[i, j] == 4):
                    cmp_list[4] = cmp_list[4] + 1
                if (y_hat[i, j].type(y.dtype) == y[i, j] == 5):
                    cmp_list[5] = cmp_list[5] + 1


        correct = float(cmp.type(y[i,:].dtype).sum())
        correct_sum = correct_sum + correct

    if (tr_or_pre == 'pre'):
        cmp_list_T = torch.Tensor(cmp_list)
        y_list_T = torch.Tensor(y_list)

        correct_percent = cmp_list_T / y_list_T
        correct_percent = correct_percent.cpu().numpy()

        # print(f'数据集中共有:\n{y_list[0]}个0，\n{y_list[1]}个1，\n{y_list[2]}个2，\n{y_list[3]}个3，\n{y_list[4]}个4，\n{y_list[5]}个5，\n{y_list[6]}个6')
        # print(f'正确预测出:\n{cmp_list[0]}个0，\n{cmp_list[1]}个1，\n{cmp_list[2]}个2，\n{cmp_list[3]}个3，\n{cmp_list[4]}个4，\n{cmp_list[5]}个5，\n{cmp_list[6]}个6')
        # print(f'各分类正确预测百分比:\n0：{correct_percent[0]}，\n1：{correct_percent[1]}，\n2：{correct_percent[2]}，\n3：{correct_percent[3]}，\n4：{correct_percent[4]}，\n5：{correct_percent[5]}，\n6：{correct_percent[6]}')

    return float(correct_sum)


def train(train_dataset, train_loader, test_dataset, test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # cuda
    # 初始化
    model = TemporalGNN(node_features=1,batchsize=512,TrainorPredict=0, FullAttention=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 加载保存的模型数据
    ###########################################    继续训练    ###########################################
    model_data = torch.load('module_best_C_L_G.pth')
    model.load_state_dict(model_data['model_dict_C_L'])
    optimizer.load_state_dict(model_data['optimizer_dict_C_L'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    ############################################    继续训练    ###########################################
    model = model.to(device)
    # loss = batch_crossentropy()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.900, last_epoch=-1, verbose=True)
    # model.apply(weights_init)

    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break;

    model.train()
    print("Running training...")

    min_loss = 100000
    for epoch in range(20):
        ###############
        labels = []
        preds = []
        ###############
        l_sum = 0
        l_min = 10000
        step = 0
        metric = Accumulator(3)

        for inputs, label in train_loader:
            y_hat = model(inputs, static_edge_index)
            # y_hat = y_hat / 0.1
            # print('y_hat:', y_hat)
            # print('label:', label.shape)
            label = label.permute(0, 2, 1).squeeze()
            label = label - 1
            # print('label:', label.shape)
            l = batch_crossentropy(y_hat, label.long())
            # 理论上gamma=0就是cross-entropy
            # l = focal_Loss(y_hat, label.long(), gamma=2)

            ###############
            pred = y_hat.argmax(axis=2)  # [1,16]
            labels.append(label)
            preds.append(pred)
            ###############
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            l_min = min(l,l_min)


            metric.add(float(l), accuracy('tr',512, y_hat, label), label.numel())

        scheduler.step()

        if l_min < min_loss:
            min_loss = l_min
            torch.save({'model_dict_C_L': model.state_dict(),
                        'optimizer_dict_C_L': optimizer.state_dict()},
                       'module_best_C_L_G.pth')
            print("save model")

        print("Epoch {} train_Loss: {:.6f} train ACC: {:.4f}".format(epoch, l_min, metric[1] / metric[2]))

        predict(test_dataset, test_loader)
        ###############
        # index = [6, 10, 13, 14]
        # for i in index:
        #     print(f'labels[{i}]:', labels[i].data)
        #     print(f'preds[{i}]:', preds[i].data)
        ###############

def predict(test_dataset, test_loader):
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
    for inputs, label in test_loader:
        y_hat = model_best(inputs, static_edge_index)
        label = label.squeeze()
        label = label - 1#[1,16]
        pred = y_hat.argmax(axis=2)          #[1,16]

        y_hats.append(y_hat)
        labels.append(label)
        preds.append(pred)

        metric.add(accuracy('pre', 512, y_hat, label), label.numel())

    test_acc = metric[0] / metric[1]

    print('test ACC:', test_acc)

    '''画图'''
    # Show_Draw(y_hats, preds, labels)


def attack_accuracy(batchsize, y_hat, y, zon):
    batch_size = batchsize
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[2] > 1:
        """axis=1：每一行的列的值进行比较，最大的作为这一行的y_hat"""

        y_hat_2 = y_hat

    correct_sum = 0

    for i in range(batch_size):
        if (batchsize == 1):
            y = torch.unsqueeze(y, 0)

        y_true_label = y[i, :]

        for k in range(y_hat.size(1)):
            # if (y_true_label[k].type(torch.long) + 1 == 6):
            #     y_true_label[k] = -1

            if (y_hat[i, k, y_true_label[k].type(torch.long)] > zon):
                a = y_hat[i, k, :]
                crr = y_hat[i, k, y_true_label[k].type(torch.long)]
                correct_sum = correct_sum + 1
            # else:
            #     print(k)
            #     print(y_true_label)
            #     print(y_hat[i, k, y_true_label[k].type(torch.long)])
            #     print(y_hat[i, k, :])

        # if i==511:
        #     print(correct_sum)


    return float(correct_sum)


def attack_predict(test_dataset, test_loader, zon):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break;

    model_best = TemporalGNN(node_features=1, batchsize=512, TrainorPredict=0, FullAttention=True)
    model_data = torch.load('module_best_C_L_G.pth')
    model_best.load_state_dict(model_data['model_dict_C_L'])
    model_best = model_best.to(device)
    model_best.eval()

    # labels = []
    # preds = []
    # y_hats = []
    metric = Accumulator(2)
    softmax_func = nn.Softmax(dim=2)
    for inputs, label in test_loader:
        y_hat = model_best(inputs, static_edge_index)
        label = label.squeeze()
        label = label - 1   # [1,16]   -1

        y_hat = softmax_func(y_hat)
        # y_hat = torch.squeeze(y_hat)

        nums = label.numel()
        metric.add(attack_accuracy(512, y_hat, label, zon), label.numel())



        # pred = y_hat.argmax(axis=2)

        # y_hats.append(y_hat)
        # labels.append(label)
        # preds.append(pred)


    test_acc = metric[0] / metric[1]
    print('test ACC:', test_acc)







def Show_Draw(y_hats, predictions, labels):
    #y_hats:List[(512,16,8)...]
    #predictions:List[(512,16)...]
    #labels:List[(512,16)...]

    y_hat = y_hats[0]
    pred = predictions[0]
    label = labels[0]

    softmax = nn.Softmax(dim=2)
    y_hat = softmax(y_hat)

    '''画谁'''
    batch_index = 512
    TrueTime = 0
    for i in range(batch_index):
        # print(f'样本[{i}]所有可能性:',y_hat[i, :, :])
        max_value = torch.max(y_hat[i, :, :], dim=1)[0]
        max_index = torch.max(y_hat[i, :, :], dim=1)[1]
        max_value_list = max_value.squeeze().cpu().detach().numpy()
        max_index_list = max_index.squeeze().cpu().detach().numpy()

        # max_index = max_index.squeeze().cpu().numpy()
        b = 0
        for j in max_index_list:
            y_hat[i, b, j] = 0
            b = b + 1

        secmax_value = torch.max(y_hat[i, :, :], dim=1)[0]
        secmax_index = torch.max(y_hat[i, :, :], dim=1)[1]
        secmax_value_list = secmax_value.squeeze().cpu().detach().numpy()
        secmax_index_list = secmax_index.squeeze().cpu().detach().numpy()

        # print(f'样本[{i}]预测值:', pred[i, :])
        # print(f'样本[{i}]第1个点\n可能性第一大的是:索引:', max_index_list[0],';值:',max_value_list[0],'\n可能性第二大的是:索引:',secmax_index_list[0],';值:',secmax_value_list[0])
        # print(f'样本[{i}]第2个点\n可能性第一大的是:索引:', max_index_list[1],';值:',max_value_list[1],'\n可能性第二大的是:索引:',secmax_index_list[1],';值:',secmax_value_list[1])
        # print(f'样本[{i}]第3个点\n可能性第一大的是:索引:', max_index_list[2],';值:',max_value_list[2],'\n可能性第二大的是:索引:',secmax_index_list[2],';值:',secmax_value_list[2])
        # print(f'样本[{i}]第4个点\n可能性第一大的是:索引:', max_index_list[3],';值:',max_value_list[3],'\n可能性第二大的是:索引:',secmax_index_list[3],';值:',secmax_value_list[3])
        # print(f'样本[{i}]第5个点\n可能性第一大的是:索引:', max_index_list[4],';值:',max_value_list[4],'\n可能性第二大的是:索引:',secmax_index_list[4],';值:',secmax_value_list[4])
        # print(f'样本[{i}]第6个点\n可能性第一大的是:索引:', max_index_list[5],';值:',max_value_list[5],'\n可能性第二大的是:索引:',secmax_index_list[5],';值:',secmax_value_list[5])
        #
        # print(f'样本[{i}]真实值:',label[i, :])


        label_list = label[i, :].squeeze().cpu().detach().numpy()

    # for i in range(batch_index):
        for attack_index in range(6):
            if(label_list[attack_index] == max_index_list[attack_index] or
                    label_list[attack_index] == secmax_index_list[attack_index]):
                # print('安全|')
                TrueTime = TrueTime + 1
    print("可能性前二包含真实类别的概率",TrueTime/3072)
    #     indices = torch.tensor([i])
    #     prediction = torch.index_select(predictions, 0, indices).squeeze()
    #     label = torch.index_select(labels, 0, indices).squeeze()
    #     loss_show = torch.mean(pow((prediction - label) ** 2, 0.5))
    #     predictions_list.append(prediction)
    #     labels_list.append(label)
    #     loss_list.append(loss_show)
    #     print("——————————————————")
    #     print("prediction:", prediction)
    #     print("label:", label)
    #     print("loss:", loss_show)
    #
    # print("loss_list:", loss_list)
    #
    # plt.figure(figsize=(10, 10), dpi=70)  # 设置图像大小
    # X = np.linspace(1, 16, 16)
    # plt.title('Prediction series for Gself')
    #
    # for i in range(1,5):
    #     ax = plt.subplot(1, 4, i)
    #     ax.set_xlabel('state')
    #     ax.set_ylabel('feature')
    #     plt.plot(X, predictions_list[i - 1])
    #     plt.plot(X, labels_list[i - 1])
    #     plt.ylim(-1, 8)
    #
    # plt.show()

def proceess():
    device = torch.device('cpu')
    model_best = TemporalGNN(node_features=1,batchsize=1,TrainorPredict=0,FullAttention=True)
    model_data = torch.load('module_best_C_L_G.pth')
    model_best.load_state_dict(model_data['model_dict_C_L'])
    model_best = model_best.to(device)
    model_best.eval()


    '''数据'''#[2,3,6,4,5,1,3,4]

    x = torch.tensor([[4,6,5,4,6,5,6,1],
                      [6,5,4,6,5,6,1,2],
                      [5,4,6,5,6,1,2,2],
                      [4,6,5,6,1,2,2,3],
                      [6,5,6,1,2,2,3,4],
                      [5,6,1,2,2,3,4,5],
                      [6,1,2,2,3,4,5,4],
                      [1,2,2,3,4,5,4,6]], dtype=torch.float)

    x = torch.unsqueeze(x, 0)
    x = torch.unsqueeze(x, 0)
    input = x.permute(0, 3, 1, 2)

    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6, 7]],
                              dtype=torch.long)

    y_hat = model_best(input, edge_index)  # [1,16,8]

    softmax_func = nn.Softmax(dim=2)
    y_hat = softmax_func(y_hat)
    y_hat = torch.squeeze(y_hat)   # [16,8]

    print('8个点每个事件的6个可能性：', y_hat)
    # for i in range(4):
    #     #扔模型去预测
    #     y_hat = model_best(input, edge_index)   #[16,7]
    #
    #     softmax_func = nn.Softmax(dim=2)
    #     y_hat = softmax_func(y_hat)
    #     y_hat = torch.unsqueeze(y_hat)
    #
    #     print('soft_output', y_hat)
    #
    #     pred = y_hat.argmax(axis=2)  # [16]
    #     pred = pred.unsqueeze(0)           # [1,16]
    #     print('pred:', pred)
    #
    #     #变成array
    #     x = x.detach().numpy()
    #     pred = pred.detach().numpy()
    #     #加一行，减一行
    #     x = np.delete(x, 0, axis=0)
    #     x = np.row_stack((x, pred))
    #
    #     #变回Tensor
    #     x = torch.from_numpy(x).to(torch.float32)

if __name__ == '__main__':
    zon = 0.01
    train_dataset, test_dataset, train_loader, test_loader = load_split()

    train(train_dataset, train_loader, test_dataset, test_loader)
    while zon <= 0.3:
        attack_predict(test_dataset, test_loader, zon)
        zon = zon + 0.01

    attack_predict(test_dataset, test_loader, zon)

    # proceess()
