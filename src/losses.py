import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss, DiceCELoss

class LossFlood(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = FocalLoss(gamma=2.0, to_onehot_y=False)
        
        self.diceCE = DiceCELoss(include_background=False,
#                                  jaccard=True,
                                 sigmoid=True,
                                 reduction='mean',
                                 lambda_dice=0.85,
                                 lambda_ce=0.15,
                                 batch=True)
        
    def _loss(self, p, y):
        return self.diceCE(p, y)                    #self.dice(p, y) + self.ce(p, y)
    
    def forward(self, p, y):
        return self._loss(p, y)