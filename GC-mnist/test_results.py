import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
import json
import tqdm
import time
import numpy as np

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.002)
    parser.add_argument('--batch_size',type = int ,default=256)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=20)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda"
    test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    
    for _epochs in range(1,101):
        global_steps = 118 * _epochs
        checkpoint_path = "results/steps_{:0>8}.pt".format(global_steps)
        if os.path.exists("results/dae_{:0>8}.json".format(global_steps)):
            continue
        while not os.path.exists(checkpoint_path):
            time.sleep(60)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])

        loss_fn=nn.MSELoss(reduction='none')
#         loss_fn=nn.MSELoss(reduction='mean')
        losses = [[] for i in range(args.timesteps)]
        model.eval()
        for t in range(5):
            for j,(image,target) in enumerate(test_dataloader):
                noise=torch.randn_like(image).to(device)
                image=image.to(device)
                pred, _t=model(image,noise,True)
                _loss=loss_fn(pred,noise).detach().cpu()
                _loss=torch.mean(_loss, dim=(1,2,3))
                _t = _t.cpu()
                for _i, _j in enumerate(_t):
                    losses[_j.item()].append(_loss[_i].item())
                    
        for i in range(len(losses)):
            losses[i]=[model.alphas_cumprod[i].cpu().item(), np.mean(losses[i])]
#         print(losses)
        with open("results/dae_{:0>8}.json".format(global_steps), 'w') as json_file:
            json.dump(losses, json_file, indent=4)
        print("saved at results/dae_{:0>8}.json".format(global_steps))

if __name__=="__main__":
    args=parse_args()
    main(args)