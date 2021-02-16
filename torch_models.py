import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np



def pairwise_euclid_distance(A):
    sqr_norm_A = (torch.sum(torch.pow(A, 2), 1, keepdim=False)).unsqueeze(0)
    sqr_norm_B = (torch.sum(torch.pow(A, 2), 1, keepdim=False)).unsqueeze(1)
    inner_prod = torch.matmul(A, A.transpose(0, 1))
    tile_1 = sqr_norm_A.repeat(A.shape[0], 1)
    tile_2 = sqr_norm_B.repeat(1, A.shape[0])
    return tile_1 + tile_2 - 2 * inner_prod


def pairwise_cos_distance(A):
    normalized_A = F.normalize(A, p=2, dim=1)
    return 1 - torch.matmul(normalized_A, normalized_A.transpose(0, 1))

class EWE_2_conv(nn.Module):
    def __init__(self, C, H, W, num_class, factors, metric):
        super(EWE_2_conv, self ).__init__() 
        self.metric = metric
        self.num_class = num_class
        self.factor_1 = factors[0]
        self.factor_2 = factors[1]
        self.factor_3 = factors[2]
        
        self.conv1 = nn.Conv2d(C, 32, kernel_size=[5,5], padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=[3,3], padding=1)
        self.fc1 = nn.Linear(64*(H//4)*(W//4), 128)
        self.fc2 = nn.Linear(128, self.num_class)
        
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.dropout3 = nn.Dropout()
        
        self.pool1 = nn.MaxPool2d(kernel_size=[2,2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[2,2], stride=2)
        
        '''
        for m in self.modules():
            if (not isinstance(m, EWE_2_conv)) and (not isinstance(m, nn.Dropout)) and (not isinstance(m, nn.MaxPool2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        '''
        
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        '''
        

        
    def forward(self, x, y, w, temp):
        # 其实最好分离开去算，效率上会更好。。。
        x = self.conv1(x) #[-1, 32, 28, 28]
        s1 = x
        x = F.relu(x)
        x = self.pool1(x) #[-1, 32, 14, 14]
        x = self.dropout1(x) 
        x = self.conv2(x) #[-1, 64, 14, 14]
        s2 = x
        x = F.relu(x)
        x = self.pool2(x) #[-1, 64, 7, 7]
        x = self.dropout2(x)
        x = self.fc1(x.reshape([x.shape[0], -1]))
        x = self.dropout3(x)
        s3 = x
        x = F.relu(x)
        x = self.fc2(x)

        loss1 = self.snnl(s1, w, 100.0 /temp[0], metric=self.metric)
        loss2 = self.snnl(s2, w, 100.0 /temp[1], metric=self.metric)
        loss3 = self.snnl(s3, w, 100.0 /temp[2], metric=self.metric)
        soft_nearest_neighbor = self.factor_1 * loss1 + self.factor_2 * loss2 + self.factor_3 * loss3
        soft_nearest_neighbor = (torch.mean(torch.from_numpy(w)) > 0).float() * soft_nearest_neighbor
        plain_snnl = loss1 + loss2 + loss3

        ce_loss = self.ce(x, y)
        
        total_loss = ce_loss - soft_nearest_neighbor
        return [ce_loss, plain_snnl, total_loss, x]

    def snnl(self, x, y, t, metric='euclidean'):
        if metric == 'euclidean':
            dist_func = pairwise_euclid_distance
        elif metric == 'cosine':
            dist_func = pairwise_cos_distance
        else:
            raise NotImplementedError()
        x = F.relu(x)
        same_label_mask = torch.Tensor(y == np.expand_dims(y, 1)).squeeze().float().to(x.device)
        dist = dist_func(x.reshape(x.shape[0], -1))
        exp = torch.clamp(torch.exp(-(dist / t)) - torch.eye(x.shape[0]).to(x.device), 0, 1)
        prob = (exp / (0.00001 + (torch.sum(exp, 1, keepdim=False)).unsqueeze(1))) * same_label_mask
        loss = - torch.mean(torch.log(0.00001 + torch.sum(prob, 1, keepdim=False)))
        return loss

    def ce(self, pred_y, y):
        log_prob = F.log_softmax(pred_y, dim=1)
        cross_entropy = - torch.sum(y * log_prob) # 这里可能有问题，但源代码就是这么干的。。
        return cross_entropy


class Plain_2_conv(nn.Module):
    def __init__(self, C, H, W, num_class):
        super(Plain_2_conv, self ).__init__() 
        self.num_class = num_class
        self.conv1 = nn.Conv2d(C, 32, kernel_size=[5,5], padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=[3,3], padding=1)
        self.fc1 = nn.Linear(64*(H//4)*(W//4), 128)
        self.fc2 = nn.Linear(128, self.num_class)
        
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.dropout3 = nn.Dropout()
        
        self.pool1 = nn.MaxPool2d(kernel_size=[2,2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[2,2], stride=2)

    def forward(self, x, y):
        x = self.conv1(x) #[-1, 32, 28, 28]
        x = F.relu(x)
        x = self.pool1(x) #[-1, 32, 14, 14]
        x = self.dropout1(x) 
        x = self.conv2(x) #[-1, 64, 14, 14]
        x = F.relu(x)
        x = self.pool2(x) #[-1, 64, 7, 7]
        x = self.dropout2(x)
        x = self.fc1(x.reshape([x.shape[0], -1]))
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        #这里和原来的plain cnn有一定的不同。
        ce_loss = self.ce(x, y)
        return [ce_loss, x]

    def ce(self, pred_y, y):
        log_prob = F.log_softmax(pred_y, dim=1)
        cross_entropy = - torch.sum(y * log_prob) # 这里可能有问题，但源代码就是这么干的。。
        return cross_entropy