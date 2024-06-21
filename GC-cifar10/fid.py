
import torch
from torchvision import datasets, transforms, models
from scipy.linalg import sqrtm
import numpy as np
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image

class FID:
    def __init__(self, fidnum=10000):
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        cifar10 = datasets.CIFAR10(root='../cifar10/cifar10_data', train=True, download=True, transform=transform)
        cifar10_loader = DataLoader(cifar10, batch_size=512, shuffle=True)

        self.inception = models.inception_v3(pretrained=True)
        self.inception.fc = torch.nn.Identity()  # Remove the classification layer
        self.inception.eval()
        if torch.cuda.is_available():
            self.inception = self.inception.cuda()

        self.fidnum = fidnum
        cifar10_features = self.get_features(cifar10_loader)
        self.mu_real, self.sigma_real = cifar10_features.mean(axis=0), np.cov(cifar10_features, rowvar=False)
        
    def get_features(self, dataloader):
        features = []
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                batchsize = images.shape[0]
                if i*batchsize>=self.fidnum:
                    break
                images = images.cuda()
                feature = self.inception(images)
                features.append(feature.cpu().numpy().reshape(images.size(0), -1))
        return np.concatenate(features, axis=0)[:self.fidnum]
    
    def calculate_fid(self, dataloader):
        act = self.get_features(dataloader)
        mu, sigma = act.mean(axis=0), np.cov(act, rowvar=False)
        ssdiff = np.sum((mu - self.mu_real) ** 2.0)
        covmean = sqrtm(sigma.dot(self.sigma_real))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma + self.sigma_real - 2.0 * covmean)
        return fid
            
class ImageDataset(Dataset):
    def __init__(self, root_dir, nums, transform=None):
        self.root_dir = root_dir
        self.nums = nums
        self.transform = transform

    def __len__(self):
        return self.nums

    def __getitem__(self, idx):
        img_file = os.path.join(self.root_dir, f'{idx+1:05d}.png')
        image = Image.open(img_file).convert('RGB')
        if self.transform is not None:
            return self.transform(image),idx
        return image,idx

if __name__ == "__main__":
    fid = FID(10000)
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for steps in [10, 20, 50, 100, 200, 500]:
        for num_iter in range(1, 6):
            dataset = ImageDataset(f"./image/aadm-{steps:04d}steps-{num_iter:02d}iter/", transform)
            fid_result = fid.calculate_fid(DataLoader(dataset, batch_size=512, shuffle=False))
            with open("./log.txt", 'a') as f:
                f.write(f"aadm-{steps:04d}steps-{num_iter:02d}iter-FID10k: {fid_result:.4f}\n")
            print(f"aadm-{steps:04d}steps-{num_iter:02d}iter-FID10k: {fid_result:.4f}\n")
    

