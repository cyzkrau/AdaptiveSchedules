import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
import json
import numpy as np
from _ada_plan import ada_choose_plan

def main():
    device="cuda"
    model=MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                base_dim=64,
                dim_mults=[2,4]).to(device)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - 5e-3)
    
    ks = 20
    Ls = {"sto":lambda t:1./np.exp(-2*t), "lip":lambda t:1.}
    for L in Ls.keys():
        i=100
        global_steps = 118*i
        checkpoint_path = "results/steps_{:0>8}.pt".format(global_steps)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint["model_ema"])
        model.load_state_dict(checkpoint['model'])

        model_ema.eval()

        with open('results/dae_{:0>8}.json'.format(global_steps), 'r') as json_file:
            data = np.array(json.load(json_file))
        ts = -np.log(data[:,0])
        dae = np.array(data[:,1]) / (1-np.exp(-2*ts))

        _split = 200
        plan=list(range(0,1000,_split))
        if L=='sto':
            for k in range(ks):
                save_image(model_ema.module.skip_sampling(
                    49,plan,clipped_reverse_diffusion=False,device=device
                ).cpu(), f'./49/epoch{i:02d}-ddim-n{k:02d}.png', nrow=7)
        
        while True:
            new_plan = list(plan)
            new_plan = ada_choose_plan(
                points=ts, 
                plan=new_plan, 
                smb=list(dae), 
                L=Ls[L], 
                C=1., 
                max_num=None, 
                max_step_length=1000,
                adjust_step=1000)
            for j in range(len(plan)):
                if plan[j] != new_plan[j]:
                    plan = new_plan
                    print(plan)
                    break
            else:
                break
                
        save_image(model_ema.module.skip_sampling(
            49,plan,clipped_reverse_diffusion=False,device=device
        ).cpu(), f'./49/{L}converge.png', nrow=7)

if __name__=="__main__":
    main()