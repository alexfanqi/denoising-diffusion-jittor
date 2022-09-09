import argparse
import pathlib
import torch
from torchvision import transforms as T, utils
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# def dir_path(string):
#     if os.path.isdir(string):
#         return string
#     else:
#         raise NotADirectoryError(string)

# parser = argparse.ArgumentParser()
# parser.add_argument("--data-set-path", type=dir_path, help='diretory that contains dataset of images supported by PIL')
# options = parser.parse_args()
# print(options)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size=16
image_size=32
channels=1
transform = T.Compose([
    T.Grayscale(),
    T.Resize(image_size),
    T.ToTensor()
    ])
dataset = MNIST(train=True, transform=transform, root='dataset')
dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

model = Unet(
    dim = 64,
    channels=channels,
    dim_mults = (1, 2, 4, 8)
)
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 10,   # number of steps
    loss_type = 'l1',   # L1 or L2
    channels=channels
)

trainer = Trainer(diffusion, dl, device, train_num_steps=100, save_and_sample_every=101, num_samples=4, ema_update_every=10, train_lr=0.001)
trainer.train()
