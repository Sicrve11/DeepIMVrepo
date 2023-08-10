import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

import random
import sys, os

from sklearn.model_selection import train_test_split
import my_import_data as impt
from helper import f_get_minibatch_set, evaluate
from class_DeepIMV_AISTATS import DeepIMV_AISTATS

dataPath = "/data3/shigw/MultiOmic/DeepIMV/dataset/TCGA_views/Final/"
dataName = "incomplete_multi_view_pca_1yr.npz"

X_set_comp, Y_onehot_comp, Mask_comp, X_set_incomp, Y_onehot_incomp, Mask_incomp = impt.import_dataset_TCGA(dataPath, dataName)

MODE       = 'incomplete'
model_name = 'DeepIMV_AISTATS'

M = len(X_set_comp)


SEED = 1234
OUTITERATION = 5

RESULTS_AUROC_RAND = np.zeros([4, OUTITERATION+2])
RESULTS_AUPRC_RAND = np.zeros([4, OUTITERATION+2])


out_itr = 1

tr_X_set, te_X_set, va_X_set = {}, {}, {}
for m in range(M):
    tr_X_set[m],te_X_set[m] = train_test_split(X_set_comp[m], test_size=0.2, random_state=SEED + out_itr)
    tr_X_set[m],va_X_set[m] = train_test_split(tr_X_set[m], test_size=0.2, random_state=SEED + out_itr)
    
tr_Y_onehot,te_Y_onehot, tr_M,te_M = train_test_split(Y_onehot_comp, Mask_comp, test_size=0.2, random_state=SEED + out_itr)
tr_Y_onehot,va_Y_onehot, tr_M,va_M = train_test_split(tr_Y_onehot, tr_M, test_size=0.2, random_state=SEED + out_itr)

if MODE == 'incomplete':
    for m in range(M):
        tr_X_set[m] = np.concatenate([tr_X_set[m], X_set_incomp[m]], axis=0)

    tr_Y_onehot = np.concatenate([tr_Y_onehot, Y_onehot_incomp], axis=0)
    tr_M        = np.concatenate([tr_M, Mask_incomp], axis=0)
    
    print(tr_M.shape)
elif MODE == 'complete':
    print(tr_M.shape)
else:
    raise ValueError('WRONG MODE!!!')
    

save_path = '{}/M{}_{}/{}/'.format(dataPath, M, MODE, model_name)
    
    
if not os.path.exists(save_path + 'itr{}/'.format(out_itr)):
    os.makedirs(save_path + 'itr{}/'.format(out_itr))


#%% 超参数设置
### training coefficients
alpha    = 1.0
beta     = 0.01 # IB coefficient
lr_rate  = 1e-4
k_prob   = 0.7


### network parameters
mb_size         = 32 
steps_per_batch = int(np.shape(tr_M)[0]/mb_size)
steps_per_batch = 500

x_dim_set    = [tr_X_set[m].shape[1] for m in range(len(tr_X_set))]
y_dim        = np.shape(tr_Y_onehot)[1]
y_type       = 'binary'
z_dim        = 100

h_dim_p      = 100
num_layers_p = 2

h_dim_e      = 300
num_layers_e = 3

input_dims = {
    'x_dim_set': x_dim_set,
    'y_dim': y_dim,
    'y_type': y_type,
    'z_dim': z_dim,
    
    'steps_per_batch': steps_per_batch
}

network_settings = {
    'h_dim_p1': h_dim_p,
    'num_layers_p1': num_layers_p,   #view-specific
    'h_dim_p2': h_dim_p,
    'num_layers_p2': num_layers_p,  #multi-view
    'h_dim_e': h_dim_e,
    'num_layers_e': num_layers_e,
    'fc_activate_fn': tf.nn.relu,
    'reg_scale': 0., #1e-4,
}



#%% 模型训练
tf.reset_default_graph()

# gpu_options = tf.GPUOptions()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = DeepIMV_AISTATS(sess, "DeepIMV_AISTATS", input_dims, network_settings)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

ITERATION = 500000
STEPSIZE  = 500

min_loss  = 1e+8   
max_acc   = 0.0
max_flag  = 20

tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 0, 0, 0, 0, 0, 0
va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0
    
stop_flag = 0
for itr in range(ITERATION):
    x_mb_set, y_mb, m_mb          = f_get_minibatch_set(mb_size, tr_X_set, tr_Y_onehot, tr_M)     
   
    _, Lt, Lp, Lkl, Lps, Lkls, Lc = model.train(x_mb_set, y_mb, m_mb, alpha, beta, lr_rate, k_prob)

    tr_avg_Lt   += Lt/STEPSIZE
    tr_avg_Lp   += Lp/STEPSIZE
    tr_avg_Lkl  += Lkl/STEPSIZE
    tr_avg_Lps  += Lps/STEPSIZE
    tr_avg_Lkls += Lkls/STEPSIZE
    tr_avg_Lc   += Lc/STEPSIZE

    
    x_mb_set, y_mb, m_mb          = f_get_minibatch_set(min(np.shape(va_M)[0], mb_size), va_X_set, va_Y_onehot, va_M)       
    Lt, Lp, Lkl, Lps, Lkls, Lc, _, _    = model.get_loss(x_mb_set, y_mb, m_mb, alpha, beta)
    
    va_avg_Lt   += Lt/STEPSIZE
    va_avg_Lp   += Lp/STEPSIZE
    va_avg_Lkl  += Lkl/STEPSIZE
    va_avg_Lps  += Lps/STEPSIZE
    va_avg_Lkls += Lkls/STEPSIZE
    va_avg_Lc   += Lc/STEPSIZE
    
    if (itr+1)%STEPSIZE == 0:
        y_pred, y_preds = model.predict_ys(va_X_set, va_M)
        
#         score  = 

        print( "{:05d}: TRAIN| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} | VALID| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} score={}".format(
            itr+1, tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc,  
            va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc, evaluate(va_Y_onehot, np.mean(y_preds, axis=0), y_type))
             )
            
        if min_loss > va_avg_Lt:
            min_loss  = va_avg_Lt
            stop_flag = 0
            saver.save(sess,save_path  + 'itr{}/best_model'.format(out_itr))
            print('saved...')
        else:
            stop_flag += 1
                           
        tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 0, 0, 0, 0, 0, 0
        va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0
        
        if stop_flag >= max_flag:
            break
            
print('FINISHED...')

saver.restore(sess, save_path  + 'itr{}/best_model'.format(out_itr))

#%% 模型验证
for m_available in [1,2,3,4]:

    tmp_M_mis = np.zeros_like(te_M)#np.copy(te_M)


    for i in range(len(tmp_M_mis)):
        np.random.seed(SEED+out_itr+i)
        idx = np.random.choice(4, m_available, replace=False)
        tmp_M_mis[i, idx] = 1


    #for stablity of reducing randomness..
    for i in range(100):
        _, tmp_preds_all = model.predict_ys(te_X_set, tmp_M_mis)
        if i == 0:
            y_preds_all = tmp_preds_all
        else:
            y_preds_all = np.concatenate([y_preds_all, tmp_preds_all], axis=0)

    auc1, apc1 = evaluate(te_Y_onehot, y_preds_all.mean(axis=0), y_type)

    RESULTS_AUROC_RAND[m_available-1, out_itr] = auc1
    RESULTS_AUPRC_RAND[m_available-1, out_itr] = apc1

    print("TEST - {} - #VIEW {}: auroc={:.4f}  auprc={:.4f}".format(MODE.upper(), m_available,  auc1, apc1))