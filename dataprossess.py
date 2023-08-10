import numpy as np
import pandas as pd
import random
import os

SEED = 14

# 导入数据
def import_dataset_TCGA(dataPath, dataName):
    filename = f'{dataPath}/{dataName}'
    npz      = np.load(filename)

    Mask  = npz['m']                # onehot的数据标签
    M     = np.shape(Mask)[1]       # 模态数量

    X_set = {}
    for m in range(M):
        tmp = npz['x{}'.format(m+1)]
        tmp[np.isnan(tmp[:, 0]), :] = np.nanmean(tmp, axis=0)   # 使用均值对缺失的样本进行填充
        X_set[m] = tmp

    Y     = npz['y']

    flag_incom = np.sum(Mask, axis=1) == 4
    flag_com = np.sum(Mask, axis=1) != 4

    X_set_incomp = {}
    X_set_comp   = {}
    for m in range(M):
        X_set_comp[m]    = X_set[m][flag_incom]
        X_set_incomp[m]  = X_set[m][flag_com]

    Y_comp    = Y[flag_incom]
    Y_incomp  = Y[flag_com]

    Mask_comp    = Mask[flag_incom]
    Mask_incomp  = Mask[flag_com]

    Y_onehot_incomp = np.zeros([np.shape(Y_incomp)[0], 2])
    Y_onehot_comp   = np.zeros([np.shape(Y_comp)[0], 2])
    Y_onehot_incomp[np.squeeze(Y_incomp)==0, 0] = 1
    Y_onehot_incomp[np.squeeze(Y_incomp)==1, 1] = 1
    Y_onehot_comp[np.squeeze(Y_comp) == 0, 0] = 1
    Y_onehot_comp[np.squeeze(Y_comp) == 1, 1] = 1
    
    Y_set = np.zeros([np.shape(Y)[0], 2])
    Y_set[np.squeeze(Y) == 0, 0] = 1
    Y_set[np.squeeze(Y) == 1, 1] = 1


    
    return X_set_comp, Y_onehot_comp, Mask_comp, X_set_incomp, Y_onehot_incomp, Mask_incomp, X_set, Y_set, Mask


def SplitData(x_set_, y_, mask_):
    random.seed(SEED)
    M_len = len(x_set_)

    # 首先分出测试集和训练集
    n_row = len(y_)   
    flag_tv = random.sample(range(n_row), int(0.8*n_row))
    flag_test = list(set(range(n_row)) - set(flag_tv))

    # 然后分出训练集和验证集
    n_tv = len(flag_tv)
    flag_t = random.sample(range(n_tv), int(0.8*n_tv))
    flag_train = [flag_tv[i] for i in flag_t]
    flag_valid = [flag_tv[i] for i in list(set(range(n_tv)) - set(flag_t))]

    train_set = {}
    valid_set = {}
    test_set = {}
    for i in range(M_len):
        train_set[str(i)] = x_set_[i][flag_train]
        valid_set[str(i)] = x_set_[i][flag_valid]
        test_set[str(i)] = x_set_[i][flag_test]
    train_set['y'] = y_[flag_train]
    train_set['mask'] = mask_[flag_train]
    valid_set['y'] = y_[flag_valid]
    valid_set['mask'] = mask_[flag_valid]
    test_set['y'] = y_[flag_test]
    test_set['mask'] = mask_[flag_test]

    return train_set, valid_set, test_set


def CombineData(data1, data2):
    assert data1.keys() == data2.keys()
    
    # 合并在一起
    res = {}
    for k in data1.keys():
        res[k] = np.vstack((data1[k], data2[k]))
    
    # 随机打乱
    flag = list(range(len(res['0'])))
    random.shuffle(flag)
    for k in data1.keys():
        res[k] = res[k][flag]

    return res


dataPath = "/data3/shigw/MultiOmic/DeepIMV/dataset/TCGA_views/Final/"
dataName = "incomplete_multi_view_pca_1yr_pca.npz"

X_set_comp, Y_onehot_comp, Mask_comp, X_set_incomp, Y_onehot_incomp, Mask_incomp, X_set, Y, Mask = import_dataset_TCGA(dataPath, dataName)
# 3983 comp     3273 incomp

#%% 划分完整的数据集
Y_all = Y.T[0]
flag_1 = Y_all == 1
flag_0 = Y_all == 0

#按照样本划分
X_0, X_1 = {}, {}
for i in range(len(X_set)):
    X_0[i] = X_set[i][flag_0]
    X_1[i] = X_set[i][flag_1]

Y_0 = Y[flag_0]
Y_1 = Y[flag_1]
Mask_0 = Mask[flag_0]
Mask_1 = Mask[flag_1]

train_0, valid_0, test_0 = SplitData(X_0, Y_0, Mask_0)
train_1, valid_1, test_1 = SplitData(X_1, Y_1, Mask_1)

train_data = CombineData(train_0, train_1)
valid_data = CombineData(valid_0, valid_1)
test_data = CombineData(test_0, test_1)

np.savez(
    './dataset/train_data.npz',
    data = train_data
)

np.savez(
    './dataset/valid_data.npz',
    data = valid_data
)

np.savez(
    './dataset/test_data.npz',
    data = test_data
)
