from diffusers import DDIMPipeline
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch, tqdm, json

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id)
ddim.to("cuda")


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar10 = datasets.CIFAR10(root='../cifar10/cifar10_data', train=True, download=True, transform=transform)
dataloader = DataLoader(cifar10, batch_size=1000, shuffle=True)

result = []
with torch.no_grad():
    for timestep in tqdm.tqdm(range(1000)):
        for i, (images, _) in enumerate(dataloader):
            images = images.to("cuda")
            noise = torch.randn_like(images).to("cuda")
            noised = torch.sqrt(ddim.scheduler.alphas_cumprod[timestep]) * images + torch.sqrt(1. - ddim.scheduler.alphas_cumprod[timestep]) * noise
            pred = ddim.unet(noised, timestep).sample
            loss = torch.mean((pred-noise)**2).cpu().item()
#             print(loss)
            result.append((torch.sqrt(ddim.scheduler.alphas_cumprod[timestep]).item(), loss))
            break

with open("dae-ddpm-cifar10-32.json", 'w') as json_file:
    json.dump(result, json_file, indent=4)
        