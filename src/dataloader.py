import rasterio
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


class FloodDst(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        source = rasterio.open(self.df.iloc[idx]['source']).read()[0]
        label = rasterio.open(self.df.iloc[idx]['label']).read()[0]
        
        source = (source - source.min()) / (source.max() - source.min())
        
        if np.isnan(source).any():
            print('there are nan values: '+str(idx))
            
        if self.transform is not None:
            data = self.transform(image=source, mask=label) 
            
        img = torch.tensor(data['image'][None], dtype=torch.float32)
        lbl =  torch.tensor(data['mask'][None], dtype=torch.float32)
        
        return img, lbl


class FloodDataModule(LightningDataModule):
    def __init__(self, args, train_dataset, val_dataset):
        super().__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def setup(self, stage=None):
        
        self.valid_dataset, self.test_dataset = random_split(self.val_dataset,
                                                             [math.floor(0.99*len(self.val_dataset)),
                                                              math.ceil(0.01*len(self.val_dataset))],
                                                              generator=self.args.generator)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)
    

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers,
                          pin_memory=False,
                          drop_last=False)
    
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers,
                          pin_memory=False,
                          drop_last=False)
