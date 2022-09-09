import os
import numpy as np
import jittor as jt
from jittor import transform as T
from PIL import Image
from denoising_diffusion_jittor import Unet, GaussianDiffusion
# import torch as jt
# from denoising_diffusion import Unet, GaussianDiffusion

jt.flags.use_cuda = jt.has_cuda
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 30,   # number of steps
    loss_type = 'l1'    # L1 or L2
)
training_images = jt.randn(8, 3, 128, 128) # images are normalized from 0 to 1
jt.sync_all(True)
#loss.backward()
# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
print(sampled_images.shape) # (4, 3, 128, 128)
test_dir='test_images'
os.makedirs(test_dir, exist_ok=True)
toimage=T.ToPILImage()
jt.save_image(sampled_images, f'{test_dir}/test.png')

