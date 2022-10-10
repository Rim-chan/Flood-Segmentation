import gc
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from monai.networks.nets import DynUNet, AttentionUnet
from losses import *
from metrics import *
import mlflow

class Unet(pl.LightningModule):
    def __init__(self, args, train_dataloader):
        super().__init__()
        self.args = args
        self.train_dataloader = train_dataloader
        self.build_model()
        self.loss = LossFlood()
        self.dice = DiceFlood(n_class=self.args.out_channels)
        self.epochs = 0
        
    
    def training_step(self, batch, batch_idx):
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        self.dice.update(logits, lbl, loss) 

    def predict_step(self, batch, batch_idx):
        img, lbl = batch
        preds = self.model(img)
        preds = (nn.Sigmoid()(preds) > 0.5).int()
        lbl_np = lbl.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        np.save(self.args.save_path + 'predictions.npy', preds_np)
        np.save(self.args.save_path + 'labels.npy', lbl_np) 

    def training_epoch_end(self, outputs):        
        torch.cuda.empty_cache()
        gc.collect()
        
    def validation_epoch_end(self, outputs):
        dice, loss = self.dice.compute()
        dice_mean = dice.mean().item()
        self.epochs += 1
        self.dice.reset()
        
        print(f'[Epoch: {self.current_epoch}]')
        print(f"Val_Performace: dice_mean {dice_mean:.3f}, Val_Loss {loss.item():.3f}")
        self.log("dice_mean", dice_mean)
        self.log("Val_Loss", loss.item())   
        
        mlflow.log_metrics({
        "dice_mean": dice_mean,
        "val_Loss": loss.item()
        })
        
        mlflow.log_params({
        "batch_size": self.args.batch_size,
        "LR": self.args.learning_rate,
#         "epochs": self.epochs
        })
  
        torch.cuda.empty_cache()
        gc.collect()        
        
        
    def build_model(self):
        self.model = smp.Unet(
            encoder_name="timm-mobilenetv3_small_minimal_100",  # resnet18/resnet34/mobilenet_v2/efficientnet-b7
            encoder_weights=None,  #"imagenet"
            in_channels=self.args.in_channels,
            classes=self.args.out_channels,
            decoder_attention_type=None)  #'scse'
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.args.max_lr,
                                                        epochs=self.args.num_epochs,
                                                        steps_per_epoch=len(self.train_dataloader))
        
        scheduler = {"scheduler": scheduler, "step" : "step" } 
        return [optimizer], [scheduler]
