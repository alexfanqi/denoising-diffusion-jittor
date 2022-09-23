import argparse
import pathlib
import jittor as jt
from jittor.dataset import Dataset
from jittor.dataset.mnist import MNIST
from jittor import transform as T
from denoising_diffusion_jittor import Unet, GaussianDiffusion, Trainer

# def dir_path(string):
#     if os.path.isdir(string):
#         return string
#     else:
#         raise NotADirectoryError(string)

# parser = argparse.ArgumentParser()
# parser.add_argument("--data-set-path", type=dir_path, help='diretory that contains dataset of images supported by PIL')
# options = parser.parse_args()
# print(options)

batch_size=16
image_size=32
channels=1
jt.flags.use_cuda = jt.has_cuda
transform = T.Compose([
    T.Resize(image_size),
    T.Gray(),
    T.ToTensor()
    ])
dataset = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=False)
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
    channels=channels,
)

# training_images = jt.randn(8, 3, 128, 128) # images are normalized from 0 to 1
# loss = diffusion(training_images)
trainer = Trainer(diffusion, dataset, train_num_steps=11, save_and_sample_every=10, num_samples=4, ema_update_every=100, train_lr=0.001, results_folder='./results/mnist/')
breakpoint()
#trainer.save(0, model_ext='pkl')
#trainer.load("000", '../denoising-diffusion-pytorch/results')
trainer.load(0)
trainer.train()
#trainer.autodiff()
