import os

import numpy as np
import pandas as pd

tumor_list = [
    'ACC','BLCA','BRCA','CESC','CHOL','COAD','COADREAD','DLBC','ESCA','FPPP',
    'GBM','GBMLGG','HNSC','KICH','KIPAN','KIRC','KIRP','LAML',
    'LGG','LIHC','LUAD','LUSC','MESO','OV','PAAD','PCPG',
    'PRAD','READ','SARC','SKCM','STAD','STES','TGCT','THCA','THYM',
    'UCEC','UCS','UVM'
]

viewType = [
    "Methylation_Preprocess.Level_3.2016012800.0.0.tar.gz",
    "miRseq_Preprocess.Level_3.2016012800.0.0.tar.gz",
    "mRNAseq_Preprocess.Level_3.2016012800.0.0.tar.gz",
    "RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz",
    "Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz"
]

path = "/data3/shigw/MultiOmic/DeepIMV/dataset/"
viewPath = "/data3/shigw/MultiOmic/DeepIMV/dataset/TCGA_views/"

# # 将文件按照类别归类
# for vT in viewType:
#     os.system(f"mkdir {vT.split('_')[0]}")
#     os.system(f"mv ../TCGA/*/*.{vT} {vT.split('_')[0]}")
#     print(vT)

# # 首先将数据都解压了
# for vT in viewType:
#     vTname = vT.split('_')[0]
#     os.system(f"ls ./{vTname}/*.tar.gz" + " | xargs -n1 -I {} tar xzvf {} -C ./" + vTname)
#     print(vT, " over!")


#%% PRPA - proteomics
## 1. FIND SUPERSET OF RPPA FEATURES
# 特征提取出来
feat_list = {}
for tumor in tumor_list:
    filepath = './RPPA/gdac.broadinstitute.org_{}.RPPA_AnnotateWithGene.Level_3.2016012800.0.0/'.format(tumor)
    filename = '{}.rppa.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t')

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        feat_list[tumor] = list(tmp)[1:]
        
        if tumor == 'ACC':
            final_feat_list = feat_list[tumor].copy()
            sup_feat_list   = feat_list[tumor].copy()
        else:
            final_feat_list = np.intersect1d(final_feat_list, feat_list[tumor])
            sup_feat_list  += feat_list[tumor]
            
sup_feat_list = np.unique(sup_feat_list).tolist()

for tumor in tumor_list:
    filepath = './RPPA/gdac.broadinstitute.org_{}.RPPA_AnnotateWithGene.Level_3.2016012800.0.0/'.format(tumor)
    filename = '{}.rppa.txt'.format(tumor)
    
    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t')

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        tmp_ = pd.DataFrame([], columns=['Composite.Element.REF'] + sup_feat_list)
        tmp_[['Composite.Element.REF'] + feat_list[tumor]] = tmp[['Composite.Element.REF'] + feat_list[tumor]]
        
        if tumor == 'ACC':
#             final_df = tmp[['gene'] + final_feat_list.tolist()]
            final_df = tmp_
        else:
#             final_df = pd.concat([final_df, tmp[['gene'] + final_feat_list.tolist()]], axis=0)
            final_df = pd.concat([final_df, tmp_], axis=0)
    
final_df = final_df.drop_duplicates(subset=['Composite.Element.REF']).reset_index(drop=True)
final_df.to_csv('./RPPA.csv', index=False)

#%% miRNA Seq
## 1. FIND SUPERSET OF miRNASeq FEATURES
feat_list = {}
for tumor in tumor_list:
    filepath = './miRseq/gdac.broadinstitute.org_{}.miRseq_Preprocess.Level_3.2016012800.0.0/'.format(tumor)
    filename = '{}.miRseq_RPKM_log2.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t')

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        feat_list[tumor] = list(tmp)[1:]
        
        if tumor == 'ACC':
            final_feat_list = feat_list[tumor].copy()
            sup_feat_list   = feat_list[tumor].copy()
        else:
            final_feat_list = np.intersect1d(final_feat_list, feat_list[tumor])
            sup_feat_list  += feat_list[tumor]
            
sup_feat_list = np.unique(sup_feat_list).tolist()

for tumor in tumor_list:
    filepath = './miRseq/gdac.broadinstitute.org_{}.miRseq_Preprocess.Level_3.2016012800.0.0/'.format(tumor)
    filename = '{}.miRseq_RPKM_log2.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t')

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        tmp_ = pd.DataFrame([], columns=['gene'] + sup_feat_list)
        tmp_[['gene'] + feat_list[tumor]] = tmp[['gene'] + feat_list[tumor]]
        
        if tumor == 'ACC':
#             final_df = tmp[['gene'] + final_feat_list.tolist()]
            final_df = tmp_
        else:
#             final_df = pd.concat([final_df, tmp[['gene'] + final_feat_list.tolist()]], axis=0)
            final_df = pd.concat([final_df, tmp_], axis=0)
            
final_df = final_df.drop_duplicates(subset=['gene']).reset_index(drop=True)
final_df.to_csv('./miRseq_RPKM_log2.csv', index=False)


#%% mRNA Seq
## 1. FIND SUPERSET OF miRNASeq FEATURES
feat_list = {}
for tumor in tumor_list:
    filepath = './mRNAseq/gdac.broadinstitute.org_{}.mRNAseq_Preprocess.Level_3.2016012800.0.0/'.format(tumor)
    filename = '{}.uncv2.mRNAseq_RSEM_all.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t')

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        feat_list[tumor] = list(tmp)[1:]
        
        if tumor == 'ACC':
            final_feat_list = feat_list[tumor].copy()
            sup_feat_list   = feat_list[tumor].copy()
        else:
            final_feat_list = np.intersect1d(final_feat_list, feat_list[tumor])
            sup_feat_list  += feat_list[tumor]
sup_feat_list = np.unique(sup_feat_list).tolist()

for tumor in tumor_list:
    filepath = './mRNAseq/gdac.broadinstitute.org_{}.mRNAseq_Preprocess.Level_3.2016012800.0.0/'.format(tumor)
    filename = '{}.uncv2.mRNAseq_RSEM_all.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t')

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        tmp_ = pd.DataFrame([], columns=['HYBRIDIZATION R'] + sup_feat_list)
        tmp_[['HYBRIDIZATION R'] + feat_list[tumor]] = tmp[['HYBRIDIZATION R'] + feat_list[tumor]]
        
        if tumor == 'ACC':
#             final_df = tmp[['gene'] + final_feat_list.tolist()]
            final_df = tmp_
        else:
#             final_df = pd.concat([final_df, tmp[['gene'] + final_feat_list.tolist()]], axis=0)
            final_df = pd.concat([final_df, tmp_], axis=0)

final_df = final_df.drop_duplicates(subset=['HYBRIDIZATION R']).reset_index(drop=True)
final_df.to_csv('./mRNAseq_RSEM.csv', index=False)

#%% methylation
## 1. FIND SUPERSET OF METHYLATION FEATURES
feat_list = {}
for tumor in tumor_list:
    filepath = './Methylation/gdac.broadinstitute.org_{}.Methylation_Preprocess.Level_3.2016012800.0.0/'.format(tumor)
    filename = '{}.meth.by_mean.data.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t', low_memory=False)
        tmp = tmp.iloc[1:, :].reset_index(drop=True)

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        feat_list[tumor] = list(tmp)[1:]
            
        if tumor == 'ACC':
            final_feat_list = feat_list[tumor].copy()
            sup_feat_list   = feat_list[tumor].copy()
        else:
            final_feat_list = np.intersect1d(final_feat_list, feat_list[tumor])
            sup_feat_list  += feat_list[tumor]
            
sup_feat_list = np.unique(sup_feat_list).tolist()

for tumor in tumor_list:
    filepath = './Methylation/gdac.broadinstitute.org_{}.Methylation_Preprocess.Level_3.2016012800.0.0/'.format(tumor)
    filename = '{}.meth.by_mean.data.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t', low_memory=False)

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        tmp_ = pd.DataFrame([], columns=['Hybridization REF'] + sup_feat_list)
        tmp_[['Hybridization REF'] + feat_list[tumor]] = tmp[['Hybridization REF'] + feat_list[tumor]]
        
        if tumor == 'ACC':
#             final_df = tmp[['gene'] + final_feat_list.tolist()]
            final_df = tmp_
        else:
#             final_df = pd.concat([final_df, tmp[['gene'] + final_feat_list.tolist()]], axis=0)
            final_df = pd.concat([final_df, tmp_], axis=0)
            
final_df = final_df.drop_duplicates(subset=['Hybridization REF']).reset_index(drop=True)
final_df.to_csv('./Methylation.csv', index=False)


#%% Clinical
## 1. FIND SUPERSET OF Clinical FEATURES
feat_list = {}
for tumor in tumor_list:
    filepath = './Clinical/gdac.broadinstitute.org_{}.Clinical_Pick_Tier1.Level_4.2016012800.0.0/'.format(tumor)
    filename = '{}.clin.merged.picked.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t', low_memory=False)
        tmp = tmp.iloc[1:, :].reset_index(drop=True)

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        feat_list[tumor] = list(tmp)[1:]
            
        if tumor == 'ACC':
            final_feat_list = feat_list[tumor].copy()
            sup_feat_list   = feat_list[tumor].copy()
        else:
            final_feat_list = np.intersect1d(final_feat_list, feat_list[tumor])
            sup_feat_list  += feat_list[tumor]
            
sup_feat_list = np.unique(sup_feat_list).tolist()

for tumor in tumor_list:
    filepath = './Clinical/gdac.broadinstitute.org_{}.Clinical_Pick_Tier1.Level_4.2016012800.0.0/'.format(tumor)
    filename = '{}.clin.merged.picked.txt'.format(tumor)

    if os.path.exists(filepath + filename):
        tmp = pd.read_csv(filepath + filename, sep='\t', low_memory=False)

        tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
        tmp         = tmp.T.reset_index()
        tmp.columns = tmp.iloc[0, 0:]
        tmp         = tmp.iloc[1:, :].reset_index(drop=True)
        
        tmp_ = pd.DataFrame([], columns=['Hybridization REF'] + sup_feat_list)
        tmp_[['Hybridization REF'] + feat_list[tumor]] = tmp[['Hybridization REF'] + feat_list[tumor]]
        
        if tumor == 'ACC':
#             final_df = tmp[['gene'] + final_feat_list.tolist()]
            final_df = tmp_
        else:
#             final_df = pd.concat([final_df, tmp[['gene'] + final_feat_list.tolist()]], axis=0)
            final_df = pd.concat([final_df, tmp_], axis=0)
            
final_df = final_df.drop_duplicates(subset=['Hybridization REF']).reset_index(drop=True)
final_df.to_csv('./clinical_label.csv', index=False)



#%% make multi-view observation file

# 导入数据，并对格式进行统一
mRNAseq     = pd.read_csv('./mRNAseq_RSEM.csv')
mRNAseq     = mRNAseq.drop_duplicates(subset=['HYBRIDIZATION R']).reset_index(drop=True)
mRNAseq     = mRNAseq[mRNAseq['HYBRIDIZATION R'] != 'HYBRIDIZATION R'].reset_index(drop=True)
mRNAseq     = mRNAseq.rename(columns={'HYBRIDIZATION R':'Hybridization REF'})
mRNAseq['Hybridization REF'] = mRNAseq['Hybridization REF'].apply(lambda x: x.lower()[:-3])

RPPA        = pd.read_csv('./RPPA.csv')
RPPA        = RPPA.rename(columns={'Composite.Element.REF':'Hybridization REF'})
RPPA['Hybridization REF'] = RPPA['Hybridization REF'].apply(lambda x: x.lower()[:-3])

methylation = pd.read_csv('./Methylation.csv')
methylation['Hybridization REF'] = methylation['Hybridization REF'].apply(lambda x: x.lower()[:-3])
# [9805 rows x 12866 columns]

miRNAseq    = pd.read_csv('./miRseq_RPKM_log2.csv')
miRNAseq     = miRNAseq.rename(columns={'gene':'Hybridization REF'})
miRNAseq['Hybridization REF'] = miRNAseq['Hybridization REF'].apply(lambda x: x.lower()[:-3])

mRNAseq      = mRNAseq.drop_duplicates(subset=['Hybridization REF'])
RPPA         = RPPA.drop_duplicates(subset=['Hybridization REF'])
methylation  = methylation.drop_duplicates(subset=['Hybridization REF'])
miRNAseq     = miRNAseq.drop_duplicates(subset=['Hybridization REF'])


# 去除na
tmp_list    = np.asarray(list(mRNAseq))
mRNAseq     = mRNAseq[tmp_list[mRNAseq.isna().sum(axis=0) == 0]]
# [10284 rows x 20532 columns]

tmp_list = np.asarray(list(RPPA))
RPPA     = RPPA[tmp_list[RPPA.isna().sum(axis=0) == 0]]
# [7354 rows x 123 columns]

tmp_list    = np.asarray(list(methylation))
methylation = methylation[tmp_list[methylation.isna().sum(axis=0) == 0]]
# [9805 rows x 12866 columns]

tmp_list    = np.asarray(list(miRNAseq))
miRNAseq    = miRNAseq[tmp_list[miRNAseq.isna().sum(axis=0) == 0]]
# [10153 rows x 111 columns]

label = pd.read_csv('./clinical_label.csv')
# label = pd.read_csv('./Final/clinical_label.csv')
# label = pd.read_csv('./clinical_label.csv', header=1)
label = label.sort_values(by='Hybridization REF').reset_index(drop=True)
label = label[label['Hybridization REF'].apply(lambda x: 'tcga' in x)].drop_duplicates(subset=['Hybridization REF'], keep ='last').reset_index(drop=True)
# [11158 rows x 69 columns]



label.loc[label['days_to_last_followup'] == 'endometrial', 'days_to_last_followup'] = label.loc[label['days_to_last_followup'] == 'endometrial', 'days_to_death']
label.loc[label['days_to_last_followup'] == 'endometrial', 'days_to_death'] = label.loc[label['days_to_last_followup'] == 'endometrial', 'vital_status']
label.loc[label['days_to_last_followup'] == 'endometrial', 'vital_status'] = label.loc[label['days_to_last_followup'] == 'endometrial', 'years_to_birth']

label.loc[label['days_to_last_followup'] == 'other  specify', 'days_to_last_followup'] = label.loc[label['days_to_last_followup'] == 'other  specify', 'days_to_death']
label.loc[label['days_to_last_followup'] == 'other  specify', 'days_to_death'] = label.loc[label['days_to_last_followup'] == 'other  specify', 'vital_status']
label.loc[label['days_to_last_followup'] == 'other  specify', 'vital_status'] = label.loc[label['days_to_last_followup'] == 'other  specify', 'years_to_birth']

label['1yr-mortality'] = -1.
label.loc[label['days_to_last_followup'].astype(float) >= 365, '1yr-mortality'] = 0.
label.loc[label['days_to_death'].astype(float) <= 365, '1yr-mortality'] = 1.

label['3yr-mortality'] = -1.
label.loc[label['days_to_last_followup'].astype(float) >= 3*365, '3yr-mortality'] = 0.
label.loc[label['days_to_death'].astype(float) <= 3*365, '3yr-mortality'] = 1.

label['5yr-mortality'] = -1.
label.loc[label['days_to_last_followup'].astype(float) >= 5*365, '5yr-mortality'] = 0.
label.loc[label['days_to_death'].astype(float) <= 5*365, '5yr-mortality'] = 1.

#%% 对原始矩阵数据进行标准化
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

def Z_score(data):
    values = data.values #dataframe转换为array
    values = values.astype('float32') #定义数据类型
    data = scale.fit_transform(values) 
    return data

mRNAseq.iloc[:, 1:] = Z_score(mRNAseq.iloc[:, 1:])
RPPA.iloc[:, 1:] = Z_score(RPPA.iloc[:, 1:])
methylation.iloc[:, 1:] = Z_score(methylation.iloc[:, 1:])
miRNAseq.iloc[:, 1:] = Z_score(miRNAseq.iloc[:, 1:])


#%%
os.system('mkdir -p ./Final/cleaned')
os.system('mv *.csv Final')


#%% Kpca
from sklearn.decomposition import PCA, SparsePCA, KernelPCA

for view in ['RPPA', 'miRNAseq', 'Methylation', 'mRNAseq']:
    print(view)
    if view == 'mRNAseq':
        df    = mRNAseq.copy(deep=True)
    elif view == 'miRNAseq':
        df    = miRNAseq.copy(deep=True)
    elif view == 'Methylation':
        df    = methylation.copy(deep=True)
    elif view == 'RPPA':
        df    = RPPA.copy(deep=True)

    z_dim = 100

    # pca   = KernelPCA(kernel='poly', n_components=z_dim, random_state=1234)
    pca   = PCA(n_components=z_dim, random_state=1234)
    z     =  pca.fit_transform(np.asarray(df.iloc[:, 1:]))

    df_pca = pd.DataFrame(z, index=df['Hybridization REF']).reset_index()
    df_pca.to_csv('./cleaned/{}_pca_zscore.csv'.format(view), index=False)


#%% Create multi-view dataset
view = 'mRNAseq'
df_pca1  = pd.read_csv('./cleaned/{}_pca_zscore.csv'.format(view))

view = 'Methylation'
df_pca2  = pd.read_csv('./cleaned/{}_pca_zscore.csv'.format(view))

view = 'miRNAseq'
df_pca3  = pd.read_csv('./cleaned/{}_pca_zscore.csv'.format(view))

view = 'RPPA'
df_pca4  = pd.read_csv('./cleaned/{}_pca_zscore.csv'.format(view))

idx_list_y = label.loc[label['1yr-mortality'] != -1, 'Hybridization REF']

idx_list1 = df_pca1['Hybridization REF']
idx_list2 = df_pca2['Hybridization REF']
idx_list3 = df_pca3['Hybridization REF']
idx_list4 = df_pca4['Hybridization REF']

idx_list_x = np.unique(idx_list1.tolist() + idx_list2.tolist() + idx_list3.tolist() + idx_list4.tolist())


idx_list = np.intersect1d(idx_list_x, idx_list_y)
df       = pd.DataFrame(idx_list, columns=['Hybridization REF'])  ##superset of samples that has at least one omics available.

df1 = pd.merge(df, df_pca1, how='left', on='Hybridization REF')
df2 = pd.merge(df, df_pca2, how='left', on='Hybridization REF')
df3 = pd.merge(df, df_pca3, how='left', on='Hybridization REF')
df4 = pd.merge(df, df_pca4, how='left', on='Hybridization REF')
dfy = pd.merge(df, label[['Hybridization REF','1yr-mortality']], how='left', on='Hybridization REF')

nSamples = len(df)
dfs = [df1, df2, df3, df4]
mask = np.ones((nSamples, 4))

for ii in range(nSamples):
    for jj in range(4):
        if np.isnan(dfs[jj].iloc[ii, 1]):
            mask[ii, jj] = 0

np.savez(
    './incomplete_multi_view_pca_1yr_pca.npz',
    x1     = np.asarray(df1.iloc[:, 1:]),
    x2 = np.asarray(df2.iloc[:, 1:]),
    x3    = np.asarray(df3.iloc[:, 1:]),
    x4        = np.asarray(df4.iloc[:, 1:]),
    y       = np.asarray(dfy.iloc[:, 1:]),
    m       = np.asarray(mask)
)


# np.savez(
#     './Final/incomplete_multi_view_pca_1yr.npz',
#     mRNAseq     = np.asarray(df1.iloc[:, 1:]),
#     Methylation = np.asarray(df2.iloc[:, 1:]),
#     miRNAseq    = np.asarray(df3.iloc[:, 1:]),
#     RPPA        = np.asarray(df4.iloc[:, 1:]),
#     label       = np.asarray(dfy.iloc[:, 1:])
# )





np.savez(
    './multi_omics_1yr_mortality.npz',
    mRNAseq     = np.asarray(df1.iloc[:, 1:]),
    Methylation = np.asarray(df2.iloc[:, 1:]),
    miRNAseq    = np.asarray(df3.iloc[:, 1:]),
    RPPA        = np.asarray(df4.iloc[:, 1:]),
    label       = np.asarray(dfy.iloc[:, 1:])
)

