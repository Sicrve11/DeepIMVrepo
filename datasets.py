import numpy as np
from torch.utils.data.dataset import Dataset

# VALID_PARTITIONS = {'comp_train': 0, 'comp_test': 1, 'incomp': 2}
VALID_PARTITIONS = {'train_data': 0, 'valid_data': 1, 'test_data': 2}
N_ATTRS          = 100

class TCGA(Dataset):
    """Define dataset of pre-processed data TCGA.
    
    @param partition: string
                      comp_train|comp_test|incomp [default: comp_train]
                      See VALID_PARTITIONS global variable.
    @param data_dir: string
                     path to root of .npz dataset [default: ./dataset]
    """
    def __init__(self, partition='train_data', data_dir='./dataset', ):
        self.partition       = partition
        filename = f'{data_dir}/{partition}.npz'
        data      = np.load(filename, allow_pickle=True)
        data = data['data'].tolist()
        self.modals = data.keys()
        self.feat1 = data['0'].copy()
        self.feat2 = data['1'].copy()
        self.feat3 = data['2'].copy()
        self.feat4 = data['3'].copy()
        self.label = data['y'].copy()
        self.mask  = data['mask'].copy()
        self.size            = int(len(self.label))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, label) where target is index of the target class.
        """
        x1      = self.feat1[index]
        x2      = self.feat2[index]
        x3      = self.feat3[index]
        x4      = self.feat4[index]
        label   = self.label[index]
        mask    = self.mask[index]

        return (x1, x2, x3, x4), label, mask

    def __len__(self):
        return self.size

