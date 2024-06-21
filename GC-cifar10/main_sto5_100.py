
from _ada_plan import ada_choose_plan
import numpy as np
import json, torch
from pipline_aadm import AADMPipeline as DDPMPipeline
from fid import *

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDPMPipeline.from_pretrained(model_id)
ddim.to("cuda")


with open("dae-ddpm-cifar10-32.json", 'r') as json_file:
    data = np.array(json.load(json_file))
ts = -np.log(data[:,0])
dae = data[:,1] / (1-np.exp(-2*ts))

                    
               
nums = 50000     
fid = FID(nums)
for steps in [5, 10, 20, 50, 100]:
    for num_iter in range(1, 6):
        def adjust(_plan):
            plan = _plan.numpy()
            plan = list(plan[::-1])
            for _ in range(num_iter):
                plan = ada_choose_plan(
                    points=ts, 
                    plan=plan, 
                    smb=list(dae), 
                    L=lambda t:(1+1e-9-np.exp(-2*t))**(-2),
                    C=1., 
                    max_num=None, 
                    max_step_length=int(1.1*1000/steps)+1,
                    adjust_step=int(0.1*1000/steps)+1)
            plan = np.array(plan)
            print(plan[::-1])
            return torch.tensor(plan[::-1].copy())
        
        _nums = nums
        while _nums > 0:
            image = ddim(batch_size=min(2000, _nums), ada_adjust=adjust, num_inference_steps=steps).images

            for i in image:
                i.save(f"./image/aadm-sto/aadm-{steps:04d}steps-{num_iter:02d}iter/{_nums:05d}.png")
                _nums -= 1
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = ImageDataset(f"./image/aadm-sto/aadm-{steps:04d}steps-{num_iter:02d}iter/", nums, transform)
        fid_result = fid.calculate_fid(DataLoader(dataset, batch_size=512, shuffle=False))
        with open("./log50k.txt", 'a') as f:
            f.write(f"sto-{steps:04d}-{num_iter:02d}: {fid_result:.4f}\n")
        print(f"aadm-{steps:04d}steps-{num_iter:02d}iter-FID{nums}: {fid_result:.4f}\n")
