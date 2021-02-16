import torch
import numpy as np
import argparse
import os
import pickle
import functools
import random
import torch_models as md
import torch_utilities
from torch.autograd import Variable
from tqdm import tqdm


SEED = 0

def set_seed(seed=1999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
    # Will affect efficiency if we close cudnn.

#set_seed(SEED)


def train(x_train, y_train, x_test, y_test, ewe_model, plain_model, epochs, w_epochs, lr, n_w_ratio, factors,
          temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
          dataset, distribution, verbose):
    """
    Args:
        x_train/x_test: [batch, H, W, C]
        y_train/y_test: [batcg,]
        epochs: int, number of training epochs.
        ewe_model: models with EWE. 
        plain_model: models without EWE.
        watermark_source: int, source class
        watermark_target: int, target class
    """

    height = x_train[0].shape[1]
    width = x_train[0].shape[2]
    try:
        channels = x_train[0].shape[0]
    except:
        channels = 1
    num_class = len(np.unique(y_train))
    half_batch_size = int(batch_size / 2) # Settings from the original code.

    target_data = x_train[y_train == watermark_target] # 要转换成的目标类别，一会应当是要换掉一部分的。

    # define the dataset and class to sample watermarked data
    if distribution == "in":
        source_data = x_train[y_train == watermark_source]
    elif distribution == "out":
        #直接用了其他不同的数据作为trigger
        if dataset == "mnist":
            w_dataset = "fashion" 
            with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                w_data = pickle.load(f)
            x_w, y_w = w_data["training_images"], w_data["training_labels"]
        elif dataset == "fashion":
            w_dataset = "mnist" 
            with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                w_data = pickle.load(f)
            x_w, y_w = w_data["training_images"], w_data["training_labels"]
        elif "cifar" in dataset:
            raise NotImplementedError()
        elif dataset == "speechcmd":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        x_w = np.reshape(x_w / 255, [-1, channels, height, width])
        source_data = x_w[y_w == watermark_source]
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")

    # make sure watermarked data is the same size as target data
    trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
                  :target_data.shape[0]]
    # 之后的trigger都是在这个基础上进行修改的。把目标类别的data全部都换了。

    w_label = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)#每个batch里面一半原来的数据，一半trigger
    y_train = np.arange(num_class)==y_train[:,None].astype(np.int)
    y_test = np.arange(num_class)==y_test[:,None].astype(np.int) #转换为one-hot

    index = np.arange(y_train.shape[0])
    w_0 = np.zeros([batch_size])
    trigger_label = np.zeros([batch_size, num_class])
    trigger_label[:, watermark_target] = 1

    num_batch = x_train.shape[0] // batch_size # 对于训练集，一个epoch需要跑多少次
    w_num_batch = target_data.shape[0] // batch_size * 2 #*2： 因为实际上用的是half batch size 对于目标类，一个epoch需要跑多少次
    num_test = x_test.shape[0] // batch_size
    

    def validate_watermark(model_name, trigger_set, label):
        labels = torch.zeros([batch_size, num_class], dtype=torch.int64)
        labels[:, label] = 1
        if trigger_set.shape[0] < batch_size:
            trigger_data = np.concatenate([trigger_set, trigger_set], 0)[:batch_size]
        else:
            trigger_data = trigger_set
        #凑够数量。
        model_name.eval()
        model.cuda()
        _w_place_holder = np.zeros([batch_size])
        _temp_place_holder = [1, 1, 1]
        with torch.no_grad():
            _1, _2, _3, pred_y = model_name(torch.Tensor(trigger_data).cuda(), 
                                            labels.cuda(), 
                                            _w_place_holder, 
                                            _temp_place_holder)
        correct_rate = np.mean((np.argmax(pred_y.detach().cpu().numpy(), 1) == label).astype(np.float))
        return correct_rate
    


    if "cifar" in dataset:
        raise NotImplementedError()
    else:
        model = ewe_model(channels, height, width, num_class, factors) 
    
    #***********************************先在干净数据上train数个epoch*****************************
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-5)
    for epoch in tqdm(range(epochs), desc="[Basic Train]"):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
        for batch in range(num_batch):
            tmp_x = torch.Tensor(x_train[batch * batch_size: (batch + 1) * batch_size]).cuda()
            tmp_y = torch.LongTensor( y_train[batch * batch_size: (batch + 1) * batch_size]).cuda()
            optimizer.zero_grad()
            ce_loss, snnl_loss, total_loss, pred_y = model(tmp_x, tmp_y, w_0, temperatures)
            total_loss.backward()
            optimizer.step()
    #**************************************************************************************
    

    #*****************************找到放置trigger的位置***************************************
    #如果是out distribution(mnist和fashion的默认模式)，不需要位置
    if distribution == "in":
        trigger_grad = []
        for batch in range(w_num_batch):
            batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
            x_for_snnl =  Variable(torch.from_numpy(batch_data)).float().cuda()
            x_for_snnl.requires_grad=True
            w_y_for_snnl = w_label
            _y_place_holder = torch.zeros([x_for_snnl.shape[0], num_class], dtype=torch.int64).cuda() #纯占位，防止报错。
            
            opt = torch.optim.SGD([x_for_snnl], lr=1e-3)
            opt.zero_grad() # 从之前别的代码那里学的，还没搞明白扔掉会发生什么。应该是为了清理干净？
            model.zero_grad()
            ce_loss, snnl_loss, total_loss, pred_y = model(x_for_snnl,
                                                           _y_place_holder,
                                                           w_y_for_snnl,
                                                           temperatures)
            x_for_snnl.retain_grad()
            snnl_loss.backward()
            grad = x_for_snnl.grad.data
            trigger_grad.append(grad.detach().cpu().numpy())
            
        avg_grad = np.average(np.concatenate(trigger_grad), 0)
        down_sample = np.array([[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(height - 2)] for j in range(width - 2)])
        w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
        trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1
    else:
        w_pos = [-1, -1] # trigger本身就是out的图像。
    #****************************************************************************************
    
    
    
    #****************************调整trigger*************************************************
    #目的是让trigger在没有采用文中的方法直接extract得到的模型上表现更差。
    #不影响在使用了EWE方法之后extraction的效果。但是在一般方法下，会没那么好看。(0.1附近VS小于0.01)
    
    step_list = np.zeros([w_num_batch])
    model.eval()
    for batch in range(w_num_batch):
        current_trigger = torch.from_numpy(trigger[batch * half_batch_size: (batch + 1) * half_batch_size])
        for epoch in range(maxiter):
            while validate_watermark(model, current_trigger, watermark_target) > threshold and step_list[batch] < 50:
                model.eval()
                step_list[batch] += 1
                x_for_ce = Variable(torch.cat([current_trigger, current_trigger], 0)).float().cuda()
                x_for_ce.requires_grad = True #先cuda，后require
                _y_place_holder = torch.zeros([x_for_ce.shape[0], num_class], dtype=torch.int64).cuda() # 占位，用不到。
                w_y_for_snnl = w_label
                opt = torch.optim.SGD([x_for_ce], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()
                _1, _2, _3, pred_y = model(x_for_ce, _y_place_holder, w_y_for_snnl, temperatures)
                tmp_target = pred_y[:, watermark_target] # 只反向传 目标 label
                x_for_ce.retain_grad()
                tmp_target.backward(torch.tensor([1.0,]*tmp_target.shape[0]).cuda()) #这里的backward希望别出问题。
                grad = x_for_ce.grad.data
                current_trigger = torch.clamp(current_trigger - w_lr * grad[:half_batch_size].detach().cpu().sign(), 0, 1)

            #👇
            batch_data = torch.cat([current_trigger,
                                    torch.from_numpy(target_data[batch * half_batch_size: (batch + 1) * half_batch_size])],
                                   0)
            batch_data = Variable(batch_data).float().cuda()
            batch_data.requires_grad = True
            _y_place_holder = torch.zeros([batch_data.shape[0], num_class], dtype=torch.int64).cuda() # 占位，用不到。
            opt = torch.optim.SGD([batch_data], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            _1, snnl_loss, _2, _3 = model(batch_data, _y_place_holder, w_label, temperatures)
            batch_data.retain_grad()
            snnl_loss.backward()
            grad = batch_data.grad.data
            current_trigger = torch.clamp(current_trigger + w_lr * grad[:half_batch_size].detach().cpu().sign(), 0, 1)
            #☝这段操作paper里面似乎没提到。应该是加强snnl。还是迁移了一下。
            
            
        for i in range(5):
            x_for_ce = Variable(torch.cat([current_trigger, current_trigger], 0)).float().cuda()
            x_for_ce.requires_grad = True
            _y_place_holder = torch.zeros([x_for_ce.shape[0], num_class], dtype=torch.int64).cuda() # 占位，用不到。
            opt = torch.optim.SGD([x_for_ce], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            _1, _2, _3, pred_y = model(x_for_ce, _y_place_holder, w_label, temperatures)
            tmp_target = pred_y[:, watermark_target] # 只反向传 目标 label
            x_for_ce.retain_grad()
            tmp_target.backward(torch.tensor([1.0,]*tmp_target.shape[0]).cuda())
            grad = x_for_ce.grad.data
            current_trigger = torch.clamp(current_trigger - w_lr * grad[:half_batch_size].detach().cpu().sign(), 0, 1) 
            # 不是很清楚这里为什么. 在让原来的model尽可能地分类错误water mark。。。
        trigger[batch * half_batch_size: (batch + 1) * half_batch_size] = current_trigger.detach().cpu().numpy()
    
    
    
    #*********************************在Watermark混合数据集上train*************************************
    model.train()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-5)
    for epoch in tqdm(range(round((w_epochs * num_batch / w_num_batch))), desc="[Watermarked Training]"):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
        j = 0
        normal = 0
        for batch in range(w_num_batch):
            if n_w_ratio >= 1:
                for i in range(int(n_w_ratio)):
                    if j >= num_batch:
                        j = 0
                    optimizer.zero_grad()
                    tmp_x =  torch.Tensor(x_train[j * batch_size: (j + 1) * batch_size]).cuda()
                    tmp_y =  torch.LongTensor(y_train[j * batch_size: (j + 1) * batch_size]).cuda()
                    _1, _2, total_loss, _3 = model(tmp_x, tmp_y, w_0, temperatures) # 实际上没有计算snnl
                    total_loss.backward()
                    optimizer.step()
                    j += 1
                    normal += 1
            if n_w_ratio > 0 and n_w_ratio % 1 != 0 and n_w_ratio * batch >= j:
                if j >= num_batch:
                    j = 0
                optimizer.zero_grad()
                tmp_x =  torch.Tensor(x_train[j * batch_size: (j + 1) * batch_size]).cuda()
                tmp_y =  torch.LongTensor(y_train[j * batch_size: (j + 1) * batch_size]).cuda()
                _1, _2, total_loss, _3 = model(tmp_x, tmp_y, w_0, temperatures) # 实际上没有计算snnl
                total_loss.backward()
                optimizer.step()
                j += 1
                normal += 1
            #print(f"No Watermark  Ce:{_1} SNNL:{_2} Total:{total_loss}")
            batch_data = torch.Tensor(np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)).cuda()
            tmp_y = torch.LongTensor(trigger_label).cuda()
            
            
            tmp_T = Variable(torch.FloatTensor(temperatures)).cuda()
            tmp_T.requires_grad = True
            optimizer.zero_grad()
            _1, _2, total_loss, _3 = model(batch_data, tmp_y, w_label, tmp_T)
            _2.backward(retain_graph=True)
            tmp_T.retain_grad()
            temp_grad = tmp_T.grad.data.detach().cpu()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            _temperature = torch.FloatTensor(temperatures)
            _temperature -= (temp_lr * temp_grad) #train的过程中的手动更新temperature。在原代码中测试发现是否更新影响不很大。
            # 这里和原版的实现稍有区别，原版的实现里面是直接对[snnl1, snnl2, snnl3]直接对temperature求导。
            temperatures = _temperature.tolist()
    #*****************************************************************************************

    # test
    model.eval()
    victim_correct_list = []
    with torch.no_grad():
        for batch in range(num_test):
            tmp_x = torch.Tensor(x_test[batch * batch_size: (batch + 1) * batch_size]).cuda()
            tmp_y = torch.LongTensor(y_test[batch * batch_size: (batch + 1) * batch_size]).cuda()
            _1, _2, _3, pred_y = model(tmp_x, tmp_y, w_0, temperatures)
            correct = np.argmax(pred_y.detach().cpu().numpy(), 1) == np.argmax(tmp_y.detach().cpu().numpy(), 1)
            victim_correct_list.append(correct.astype(np.float))
    victim_correct = np.mean(np.concatenate(victim_correct_list))

    victim_watermark_acc_list = []
    for batch in range(w_num_batch):
        victim_watermark_acc_list.append(validate_watermark(
            model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target))
    victim_watermark_acc = np.mean(victim_watermark_acc_list)
    if verbose:
        print(f"Victim Model || validation accuracy: {victim_correct}, "
              f"watermark success: {victim_watermark_acc}")


    #*************************Extraction Attack**********************************************
    model.eval()
    extracted_label = []
    for batch in range(num_batch):
        tmp_x = torch.Tensor(x_train[batch * batch_size: (batch + 1) * batch_size]).cuda()
        _y_place_holder = torch.LongTensor([tmp_x.shape[0],]).cuda()
        _1, _2, _3, output = model(tmp_x, _y_place_holder, w_0, temperatures)
        extracted_label.append(output.detach().cpu().numpy() == np.max(output.detach().cpu().numpy(), 1, keepdims=True))
    extracted_label = np.concatenate(extracted_label, 0).astype(np.int64) #注意！这里拿到的是hard label!
    extracted_data = x_train[:extracted_label.shape[0]]

    if "cifar" in dataset:
        raise NotImplementedError()
    else:
        extracted_model = plain_model(channels, height, width, num_class)
    extracted_model.train()
    extracted_model.cuda()
    attacker_optimizer = torch.optim.Adam(extracted_model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-5)
    for epoch in tqdm(range(epochs + w_epochs), desc="[Attacking Training]"):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index] #原代码中的，没用。但还是留着了。
        for batch in range(num_batch):
            attacker_optimizer.zero_grad()
            tmp_x = torch.Tensor(extracted_data[batch * batch_size: (batch + 1) * batch_size]).cuda()
            tmp_y = torch.LongTensor(extracted_label[batch * batch_size: (batch + 1) * batch_size]).cuda()
            ce_loss, _1 = extracted_model(tmp_x, tmp_y)
            ce_loss.backward()
            attacker_optimizer.step()

    extracted_model.eval()
    extracted_correct_list = []
    with torch.no_grad():
        for batch in range(num_test):
            true_label = y_test[batch * batch_size: (batch + 1) * batch_size]
            tmp_x = torch.Tensor(x_test[batch * batch_size: (batch + 1) * batch_size]).cuda()
            _y_place_holder = torch.LongTensor([tmp_x.shape[0],]).cuda()
            _1, pred_y = extracted_model(tmp_x, _y_place_holder)
            correct = (np.argmax(pred_y.detach().cpu().numpy(), 1) == np.argmax(true_label, 1)).astype(np.float)
            extracted_correct_list.append(correct)
    extracted_correct = np.mean(np.concatenate(extracted_correct_list))

    model.eval()
    extracted_watermark_acc_list = []
    with torch.no_grad():
        for batch in range(w_num_batch):
            tmp_x = torch.Tensor(trigger[batch * half_batch_size: (batch + 1) * half_batch_size]).cuda()
            _y_place_holder = torch.LongTensor([tmp_x.shape[0]]).cuda()
            _1, pred_y = extracted_model(tmp_x, _y_place_holder)
            correct = (np.argmax(pred_y.detach().cpu().numpy(), 1) == watermark_target).astype(np.float)
            extracted_watermark_acc_list.append(correct)
    extracted_watermark_acc = np.mean(np.concatenate(extracted_watermark_acc_list))
    if verbose:
        print(f"Extracted Model || validation accuracy: {extracted_correct},"
              f" watermark success: {extracted_watermark_acc}")
    #*****************************************************************************************


    #******************************Clean model for comparison*********************************
    if "cifar" in dataset:
        raise NotImplementedError()
    else:
        clean_model = plain_model(channels, height, width, num_class)
    
    clean_model.train()
    clean_model.cuda()
    clean_optimizer = torch.optim.Adam(clean_model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-5)
    for epoch in tqdm(range(epochs + w_epochs), desc="[Baseline Training]"):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
        for batch in range(num_batch):
            clean_optimizer.zero_grad()
            tmp_x = torch.Tensor(x_train[batch * batch_size: (batch + 1) * batch_size]).cuda()
            tmp_y = torch.LongTensor(y_train[batch * batch_size: (batch + 1) * batch_size]).cuda()
            ce_loss, _1 = clean_model(tmp_x, tmp_y)
            ce_loss.backward()
            clean_optimizer.step()

    clean_model.eval()
    baseline_correct_list = []
    with torch.no_grad():
        for batch in range(num_test):
            true_label = y_test[batch * batch_size: (batch + 1) * batch_size]
            tmp_x = torch.Tensor(x_test[batch * batch_size: (batch + 1) * batch_size]).cuda()
            _y_place_holder = torch.LongTensor([tmp_x.shape[0],]).cuda()
            _1, pred_y = clean_model(tmp_x, _y_place_holder)
            correct = (np.argmax(pred_y.detach().cpu().numpy(), 1) == np.argmax(true_label, 1)).astype(np.float)
            baseline_correct_list.append(correct)
    baseline_correct = np.mean(np.concatenate(baseline_correct_list))

    clean_model.eval()
    baseline_watermark_acc_list = []
    with torch.no_grad():
        for batch in range(w_num_batch):
            tmp_x = torch.Tensor(trigger[batch * half_batch_size: (batch + 1) * half_batch_size]).cuda()
            _y_place_holder = torch.LongTensor([tmp_x.shape[0]]).cuda()
            _1, pred_y = clean_model(tmp_x, _y_place_holder)
            correct = (np.argmax(pred_y.detach().cpu().numpy(), 1) == watermark_target).astype(np.float)
            baseline_watermark_acc_list.append(correct)
    baseline_watermark_acc = np.mean(np.concatenate(baseline_watermark_acc_list))

    if verbose:
        print(f"Clean Model || validation accuracy: {baseline_correct}, "
              f"watermark success: {baseline_watermark_acc}")
    #******************************************************************************************


    return (victim_correct, victim_watermark_acc,
           extracted_correct, extracted_watermark_acc, 
           baseline_correct, baseline_watermark_acc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=512)
    parser.add_argument('--ratio',
                        help='ratio of amount of legitimate data to watermarked data',
                        type=float, default=1.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epochs', help='epochs for training without watermarking', type=int, default=10)
    parser.add_argument('--w_epochs', help='epochs for training with watermarking', type=int, default=10)
    parser.add_argument('--dataset', help='mnist, fashion, speechcmd, cifar10, or cifar100', type=str, default="mnist")
    parser.add_argument('--model', help='2_conv, lstm, or resnet', type=str, default="2_conv")
    parser.add_argument('--metric', help='distance metric used in snnl, euclidean or cosine', type=str, default="cosine")
    parser.add_argument('--factors', help='weight factor for snnl', nargs='+', type=float, default=[32, 32, 32])
    parser.add_argument('--temperatures', help='temperature for snnl', nargs='+', type=float, default=[1, 1, 1])
    parser.add_argument('--threshold', help='threshold for estimated false watermark rate, should be <= 1/num_class', type=float, default=0.1)
    parser.add_argument('--maxiter', help='iter of perturb watermarked data with respect to snnl', type=int, default=10)
    parser.add_argument('--w_lr', help='learning rate for perturbing watermarked data', type=float, default=0.01)
    parser.add_argument('--t_lr', help='learning rate for temperature', type=float, default=0.1)
    parser.add_argument('--source', help='source class of watermark', type=int, default=1)
    parser.add_argument('--target', help='target class of watermark', type=int, default=7)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--default', help='whether to use default hyperparameter, 0 or 1', type=int, default=1)
    parser.add_argument('--layers', help='number of layers, only useful if model is resnet', type=int, default=18)
    parser.add_argument('--distrib', help='use in or out of distribution watermark', type=str, default='out')

    args = parser.parse_args()
    default = args.default
    batch_size = args.batch_size
    ratio = args.ratio
    lr = args.lr
    epochs = args.epochs
    w_epochs = args.w_epochs
    factors = args.factors
    temperatures = args.temperatures
    threshold = args.threshold
    w_lr = args.w_lr
    t_lr = args.t_lr
    source = args.source
    target = args.target
    seed = args.seed
    verbose = args.verbose
    dataset = args.dataset
    model_type = args.model
    maxiter = args.maxiter
    distrib = args.distrib
    layers = args.layers
    metric = args.metric
    shuffle = args.shuffle

    # hyperparameters with reasonable performance
    if default:
        if dataset == 'mnist':
            model_type = '2_conv'
            ratio = 1
            batch_size = 512
            epochs = 10
            w_epochs = 10
            factors = [32, 32, 32]
            temperatures = [1, 1, 1]
            metric = "cosine"
            threshold = 0.1
            t_lr = 0.1
            w_lr = 0.01
            source = 1
            target = 7
            maxiter = 10
            distrib = "out"
        elif dataset == 'fashion':
            if model_type == '2_conv':
                batch_size = 128
                ratio = 2
                epochs = 10
                w_epochs = 10
                factors = [32, 32, 32]
                temperatures = [1, 1, 1]
                t_lr = 0.1
                threshold = 0.1
                w_lr = 0.01
                source = 8
                target = 0
                maxiter = 10
                distrib = "out"
                metric = "cosine"
            elif model_type == 'resnet':
                batch_size = 128
                layers = 18
                ratio = 1.2
                epochs = 5
                w_epochs = 5
                factors = [1000, 1000, 1000]
                temperatures = [0.01, 0.01, 0.01]
                t_lr = 0.1
                threshold = 0.1
                w_lr = 0.01
                source = 9
                target = 0
                maxiter = 10
                distrib = "out"
                metric = "cosine"
        elif dataset == 'speechcmd':
            batch_size = 128
            epochs = 30
            w_epochs = 1
            model_type = "lstm"
            distrib = 'in'
            ratio = 1
            shuffle = 1
            t_lr = 2
            maxiter = 10
            threshold = 0.1
            factors = [16, 16, 16]
            temperatures = [30, 30, 30]
            source = 9
            target = 5
        elif dataset == "cifar10":
            batch_size = 128
            model_type = "resnet"
            layers = 18
            ratio = 4
            epochs = 50
            w_epochs = 6
            factors = [1e5, 1e5, 1e5]
            temperatures = [1, 1, 1]
            t_lr = 0.1
            threshold = 0.1
            w_lr = 0.01
            source = 8
            target = 0
            maxiter = 10
            distrib = "out"
            metric = "cosine"
        elif dataset == "cifar100":
            batch_size = 128
            model_type = "resnet"
            layers = 18
            epochs = 100
            w_epochs = 8
            ratio = 15
            factors = [1e5, 1e5, 1e5]
            temperatures = [1, 1, 1]
            t_lr = 0.01
            threshold = 0.1
            w_lr = 0.01
            source = 8
            target = 0
            maxiter = 100
            distrib = "out"
            metric = "cosine"


    random.seed(seed)
    np.random.seed(seed)
    
    if dataset == 'mnist' or dataset == 'fashion':
        with open(os.path.join("data", f"{dataset}.pkl"), 'rb') as f:
            mnist = pickle.load(f)
        x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
                                           mnist["test_images"], mnist["test_labels"]
        x_train = np.reshape(x_train / 255, [-1, 1, 28, 28])
        x_test = np.reshape(x_test / 255, [-1, 1, 28, 28])
    elif "cifar" in dataset:
        raise NotImplementedError('Dataset is not implemented.')
        '''
        import tensorflow_datasets as tfds
        ds = tfds.load(dataset)
        for i in tfds.as_numpy(ds['train'].batch(50000).take(1)):
            x_train = i['image'] / 255
            y_train = i['label']
        for i in tfds.as_numpy(ds['test'].batch(50000).take(1)):
            x_test = i['image'] / 255
            y_test = i['label']
        '''
    elif dataset == 'speechcmd':
        raise NotImplementedError('Dataset is not implemented.')
        '''
        x_train = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_train.npy')), 1, 2)
        y_train = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_train.npy'))
        x_test = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_test.npy')), 1, 2)
        y_test = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_test.npy'))
        '''
    else:
        raise NotImplementedError('Dataset is not implemented.')

    if model_type == '2_conv':
        ewe_model = functools.partial(md.EWE_2_conv, metric=metric)
        plain_model = md.Plain_2_conv
    elif model_type == 'resnet':
        raise NotImplementedError('Model is not implemented.')
    elif model_type == 'lstm':
        raise NotImplementedError('Model is not implemented.')
    else:
        raise NotImplementedError('Model is not implemented.')

    res = train(x_train, y_train, x_test, y_test, ewe_model, plain_model, epochs, w_epochs, lr, ratio, factors,
                temperatures, source, target, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
                verbose)