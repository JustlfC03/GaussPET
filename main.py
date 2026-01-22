from trainbase import trainVAE,main_train

import os
import sys
import torch

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.append(ROOT_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from dataloader.get_data import get_dataloader




data_dir = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_epochs = 40
Gauss_epochs =1#200
train_batch_size3D = 1
val_batch_size3D = 1
Gauss_save_path = ''#final model


if __name__ == '__main__':
    train_dataloader3D,val_dataloader3D = get_dataloader(data_dir,train_batch_size3D,val_batch_size3D,dimension=3,num_workers=1,modality='')
    main_train(train_dataloader3D, val_dataloader3D,Gauss_epochs, device,
               resume = True,only_test =True,save_nii = True, mohu=False,
               model_save_path=Gauss_save_path)
    print('处理完毕')