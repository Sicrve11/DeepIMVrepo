import os
import sys
import shutil
from tqdm import tqdm
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import DeepIMV
from datasets import TCGA
from datasets import N_ATTRS

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# 添加了eps的两个函数, log和div
def div(x_, y_):
    return torch.div(x_, y_ + 1e-8)

def log(x_):
    return torch.log(x_ + 1e-8)

# 变分信息瓶颈损失函数
def IB_loss(target, predict, beta, mu, logvar, module_len, mask):
    recon_loss = lambda recon_x, x: F.binary_cross_entropy(recon_x, x, size_average=False)
    kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())        # 与normal(0, 1)做kl

    if module_len == 1:     # 计算联合分布
        # 最终标签损失函数
        loss_y = recon_loss(target, predict)

        # 隐变量的损失函数
        loss_kl = kl_loss(mu, logvar)
    
    else:                   # 计算边缘分布
        loss_y, loss_kl = [], []
        for i in range(module_len):
            tmp_y = recon_loss(target, predict[i])
            tmp_kl = torch.sum(kl_loss(mu[i], logvar[i]), dim=-1)

            loss_y += [div(torch.sum(mask[:, i] * tmp_y), torch.sum(mask[:, i]))]
            loss_kl += [div(torch.sum(mask[:, i] * tmp_kl), torch.sum(mask[:, i]))]
        
        loss_y = torch.stack(loss_y)
        loss_y = torch.sum(loss_y)
        loss_kl = torch.stack(loss_kl)
        loss_kl = torch.sum(loss_kl)

    loss_IB = loss_y + beta * loss_kl
    return loss_IB

# 模型保存
def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))

# 模型导入
def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = DeepIMV()
    model.load_state_dict(checkpoint['state_dict'])
    return model

# 打印参数
def printParameter(model):
    for name, param in model.named_parameters():
        print('name:{} \nparam grad:{} \nparam requires_grad:{}\n'.format(name, param.grad, param.requires_grad))

# 设置随机种子
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



#%% Train
## 1.设置参数
# training coefficients
alpha    = 0.1
beta     = 0.01 # IB coefficient
lr_rate  = 0.01
mb_size  = 16


cuda = torch.cuda.is_available()

# 设置路径
if not os.path.isdir('./trained_models'):
    os.makedirs('./trained_models')


## 2.定义数据迭代器
train_loader   = torch.utils.data.DataLoader(
    TCGA(partition='train_data', data_dir='./dataset'),
    batch_size=mb_size, shuffle=False)

N_mini_batches = len(train_loader)
valid_loader    = torch.utils.data.DataLoader(
    TCGA(partition='valid_data', data_dir='./dataset'),
    batch_size=mb_size, shuffle=False)

test_loader    = torch.utils.data.DataLoader(
    TCGA(partition='test_data', data_dir='./dataset'),
    batch_size=mb_size, shuffle=False)

dataloaders = {
    'train' : train_loader,
    'valid' : valid_loader,
    'test' : test_loader
}

torch.cuda.empty_cache()


## 3.构建模型
setup_seed(23)
model     = DeepIMV()
device = torch.device("cuda:1" if cuda else "cpu")

# params = []     # 查看有哪些参数需要训练
# for name, param in model.named_parameters():
#     if param.requires_grad == True:
#         params.append(name)


## 4.设置优化器
optimizer = optim.Adam(model.parameters(), lr=lr_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)


## 5.测试单个batch数据能够将模型训练起来
def train_model(model, dataloaders, IB_loss, optimizer, num_epochs=25, filename='Train'):
    since = time.time()                                     # 计算训练的时间
    best_acc = 0.0                                          # 记录最好的模型精度
    best_model_wts = copy.deepcopy(model.state_dict())      # 保存当前的模型参数

    model.to(device)
    for epoch in range(num_epochs): 
        print(f'Epoch {epoch+1} / {num_epochs}')

        ## 1.训练模型
        running_correct = []                                # 用于保存运行所有batch的结果
        joint_loss_set = []
        marginal_loss_set = []
        for feats, label, mask in dataloaders['train']:     # 运算所有的batch
            # 1.1 数据导入
            feats = torch.stack(feats)
            feats = Variable(feats.float().to(device))
            mask = Variable(mask.float().to(device))

            # 1.2 模型运算
            predict_multi, mu_z, logvar_z, predict_spec, mu_set, logvar_set = model(feats, mask)

            # 1.3 计算每个batch的损失值
            label = label.float().to(device)
            joint_loss = IB_loss(label, predict_multi.float(), beta, mu_z, logvar_z, 1, mask)
            marginal_loss = IB_loss(label, tuple(pspec.float() for pspec in predict_spec), beta, mu_set, logvar_set, 4, mask)
            
            joint_loss_set.append(joint_loss)
            marginal_loss_set.append(marginal_loss)
            running_correct.append(torch.sum(predict_multi == label)/(2 * len(label)))

        # 1.4 计算这一批数据的总损失值
        epoch_loss = torch.mean(joint_loss) + alpha * torch.mean(marginal_loss)
        running_correct = torch.stack(running_correct)
        epoch_acc = torch.mean(running_correct)

        if epoch_loss == 'nan':
            quit('loss is nan!')

        # 1.5 计算梯度，更新参数
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()

        ## 2.验证模型
        running_correct_v = []                                # 用于保存运行所有batch的结果
        for feats, label, mask in dataloaders['valid']:     # 运算所有的batch
            # 2.1 数据导入
            feats = torch.stack(feats)
            feats = Variable(feats.float().to(device))
            mask = Variable(mask.float().to(device))

            # 2.2 模型运算
            predict_multi, mu_z, logvar_z, predict_spec, mu_set, logvar_set = model(feats, mask)

            # 2.3 计算预测精度
            label = label.float().to(device)
            running_correct_v.append(torch.sum(predict_multi == label)/(2 * len(label)))

        # 2.4 计算这一批数据的总精度
        running_correct_v = torch.stack(running_correct_v)
        epoch_acc_v = torch.mean(running_correct_v)


        # 打印效果
        print(' {} Loss {:.4f} Acc {:.4f}'.format('Train', epoch_loss, epoch_acc))
        print(' {} Acc {:.4f}'.format('Valid', epoch_acc_v))

        # 7.如果精度较高，则保存模型
        if epoch_acc_v > best_acc:
            best_acc = epoch_acc_v
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'state_dict' : model.state_dict(),
                'beat_acc' : best_acc,
                'optimizer' : optimizer.state_dict()
            }
            save_checkpoint(state, True, folder='./', filename=filename)

        if epoch % 2 == 0:          # 更新学习率（暂时没实现）
            # scheduler.step(epoch_loss)
            for p in optimizer.param_groups:
                p['lr'] *= 0.5      #注意这里

    time_elapsed = time.time() - since
    print('Time elapsed {:.1f}m {:.7f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('The best acc : {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    return model, best_acc


## 6.测试单个batch数据能够将模型训练起来
def test_model(model, dataloaders):
    model.to(device)
    ## 1.测试模型
    running_correct_v = []                                # 用于保存运行所有batch的结果
    for feats, label, mask in dataloaders['test']:     # 运算所有的batch
        # 2.1 数据导入
        feats = torch.stack(feats)
        feats = Variable(feats.float().to(device))
        mask = Variable(mask.float().to(device))

        # 2.2 模型运算
        predict_multi, mu_z, logvar_z, predict_spec, mu_set, logvar_set = model(feats, mask)

        # 2.3 计算预测精度
        label = label.float().to(device)
        running_correct_v.append(torch.sum(predict_multi == label)/(2 * len(label)))

    # 2.4 计算这一批数据的总精度
    running_correct_v = torch.stack(running_correct_v)
    epoch_acc_v = torch.mean(running_correct_v)

    # 打印效果
    print('\n {} Acc {:.4f}'.format('test', epoch_acc_v))



model, best_acc = train_model(model, dataloaders, IB_loss, optimizer, num_epochs=25, filename='TrainModel')

test_model(model, dataloaders)

# printParameter(model)

