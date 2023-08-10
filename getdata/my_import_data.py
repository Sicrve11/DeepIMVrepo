import numpy as np
import pandas as pd
import random



def import_dataset_TCGA(dataPath, dataName):
    filename = f'{dataPath}/{dataName}'
    npz      = np.load(filename)

    Mask  = npz['m']
    M     = np.shape(Mask)[1]

    X_set = {}
    for m in range(M):
        tmp = npz['x{}'.format(m+1)]
        tmp[np.isnan(tmp[:, 0]), :] = np.nanmean(tmp, axis=0)   # 使用均值对缺失的样本进行填充
        X_set[m] = tmp

    Y     = npz['y']

    X_set_incomp = {}
    X_set_comp   = {}
    for m in range(M):
        X_set_comp[m]    = X_set[m][np.sum(Mask, axis=1) == 4]
        X_set_incomp[m]  = X_set[m][np.sum(Mask, axis=1) != 4]

    Y_comp    = Y[np.sum(Mask, axis=1) == 4]
    Y_incomp  = Y[np.sum(Mask, axis=1) != 4]

    Mask_comp    = Mask[np.sum(Mask, axis=1) == 4]
    Mask_incomp  = Mask[np.sum(Mask, axis=1) != 4]

    Y_onehot_incomp = np.zeros([np.shape(Y_incomp)[0], 2])
    Y_onehot_comp   = np.zeros([np.shape(Y_comp)[0], 2])

    Y_onehot_incomp[np.squeeze(Y_incomp)==0, 0] = 1
    Y_onehot_incomp[np.squeeze(Y_incomp)==1, 1] = 1

    Y_onehot_comp[np.squeeze(Y_comp) == 0, 0] = 1
    Y_onehot_comp[np.squeeze(Y_comp) == 1, 1] = 1
    
    return X_set_comp, Y_onehot_comp, Mask_comp, X_set_incomp, Y_onehot_incomp, Mask_incomp
