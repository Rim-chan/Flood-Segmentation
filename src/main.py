import warnings
warnings.filterwarnings("ignore")

import time
from utils import *
from dataloader import *
from args import *
from model import *
import albumentations as A
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

import mlflow.pytorch


if __name__ == "__main__":
  args = get_main_args()
  transformations = A.Compose([A.RandomCrop(*args.resize_to),
                               A.RandomRotate90(),
                               A.HorizontalFlip(),
                               A.VerticalFlip()])
  
  train_df = train_val_files(args.s1_dir, args.seed, mode='train')
  val_df = train_val_files(args.s1_dir, args.seed, mode='val')
  train_dataset = FloodDst(train_df, transform=transformations) 
  val_dataset = FloodDst(val_df, transform=transformations)

  dm = FloodDataModule(args, train_dataset, val_dataset)
  dm.setup()

  model = Unet(args, dm.train_dataloader())


  model_ckpt = ModelCheckpoint(dirpath="./",
                               filename="best_model",
                               monitor="dice_mean",
                               mode="max",
                               save_last=True)

  early_stop_callback = EarlyStopping(
    monitor="dice_mean",
    patience=(args.patience * 3),
    mode="max")

  callbacks = [model_ckpt, early_stop_callback]
  trainer = Trainer(callbacks=callbacks,
                    enable_checkpointing=True,
                    max_epochs=args.num_epochs, 
                    enable_progress_bar=True,
                    log_every_n_steps=len(dm.train_dataloader()),
                    devices=1,
                    accelerator="gpu",
                    amp_backend='apex',
                    profiler='simple',
                    detect_anomaly=True)
  

  with mlflow.start_run(experiment_id=args.experiment_id, run_name=args.run_name):
    start_time = time.time()
    trainer.fit(model, dm)
    trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path)
    mlflow.log_metric("execution_time", (time.time() - start_time))
