import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

Tensor = torch.Tensor

def batch_crossentropy(
    input: Tensor,
    target: Tensor,
) -> Tensor:
    B = input.size(0)
    logsoftmax_func = nn.LogSoftmax(dim=2)
    logsoftmax_output = logsoftmax_func(input)

    nllloss_func = nn.NLLLoss()

    loss_sum_batch = 0

    for i in range(B):
        nlloss_output = nllloss_func(logsoftmax_output[i, :, :], target[i, :])
        loss_sum_batch = loss_sum_batch + nlloss_output

    loss_mean_batch = loss_sum_batch / B

    return loss_mean_batch

def focal_Loss(
    input: Tensor,
    target: Tensor,
    gamma: int,
) -> Tensor:

    B = input.size(0)
    N = input.size(1)
    C = input.size(2)

    pt = F.softmax(input, dim=2)

    class_mask = input.data.new(B, N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.unsqueeze(-1)
    class_mask.scatter_(2, ids.data, 1.)

    probs = (pt * class_mask).sum(2).view(B, -1, 1)
    log_p = probs.log()
    loss = -(torch.pow((1 - probs), gamma)) * log_p
    loss = loss.mean(1)
    loss = loss.mean()

    return loss


