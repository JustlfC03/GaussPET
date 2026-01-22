import os
import sys

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.append(ROOT_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


from .dataloader3D import myDataset3Dbase
from torch.utils.data import Dataset, DataLoader

def get_dataloader(data_path,train_batch_size,val_batch_size,num_workers=1,dimension=3,modality='T1'):
    if dimension != 2 and dimension != 3:
        raise ValueError('dimension must be 2 or 3')
    if dimension == 3:
        train_dataset = myDataset3Dbase(data_path,mode='train',modality=modality)
        val_dataset = myDataset3Dbase(data_path,mode='changetest',modality=modality)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    return train_dataloader, val_dataloader