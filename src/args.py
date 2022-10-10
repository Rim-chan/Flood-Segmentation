from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch


def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
  
    arg("--seed", type=int, default=26012022, help="Random Seed")
    arg("--generator", default=torch.Generator().manual_seed(26012022), help='Train Validate Predict Seed')
    arg("--s1_dir", type=str, default='../input/c2smsfloods/c2smsfloods/c2smsfloods_v1_source_s1/*', help="Sentinel 1 Source Data Directory")
    arg("--resize_to", type=tuple, default=(480, 480), help="Shape of the Resized Image")
    arg("--crop_shape", type=int, default=256, help="Shape of the cropped Image")
    arg("--batch_size", type=int, default=20, help="batch size")
    arg("--num_workers", type=int, default=2, help="Number of DataLoader Workers")
    arg("--learning_rate", type=float, default=1e-4, help="Learning Rate")
    arg("--max_lr", type=float, default=1e-2, help="Max Learning Rate")
    arg("--weight_decay", type=float, default=1e-5, help="Weight Decay")
    arg("--in_channels", type=int, default=1, help="Network Input Channels")
    arg("--out_channels", type=int, default=1, help="Network Output Channels")

    arg("--attention_channels", type=list, default=[32, 64, 128, 256], help="AttentionUNet Channels")
    arg("--attention_kernels", type=list, default=[[3, 3]] * 4, help="AttentionUNet Kernels")
    arg("--attention_strides", type=list, default=[[2, 2]] * 3 +  [[1, 1]] , help="AttentionUNet Strides")

    arg("--dynUnet_kernels", type=list, default=[[3, 3]] * 5, help="DynUNet Kernels")
    arg("--dynUnet_strides", type=list, default=[[1, 1]] +  [[2, 2]] * 4, help="DynUNet Strides")

    arg("--experiment_id", type=int, default=0, help='Mlflow Experiment ID')
    arg("--run_name", type=str, default='mlflow', help='Mlflow Run Name')
    
    arg("--exec_mode", type=str, default='train', help='Execution Mode')
    arg("--num_epochs", type=int, default=100, help="Number of Epochs")
    arg("--patience", type=int, default=4, help="Patience")    
    arg("--ckpt_path", type=str, default=None, help='Checkpoint Path')
    arg("--save_path", type=str, default='./', help='Saves Path')

    return parser.parse_args()
