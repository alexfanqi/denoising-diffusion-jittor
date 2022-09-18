import argparse
import pathlib
import jittor as jt
from jittor import dataset
from jittor.dataset import Dataset
from jittor import transform as T
from denoising_diffusion_jittor import Unet, GaussianDiffusion, Trainer, num_to_groups
import math

jt.flags.use_cuda = jt.has_cuda
jt.flags.log_silent = True

# parser = argparse.ArgumentParser()
# parser.add_argument("--data-set-path", type=dir_path, help='diretory that contains dataset of images supported by PIL')
# options = parser.parse_args()
# print(options)

batch_size=128
image_size=32
channels=3
dataset_name = 'cifar10'
transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
ds = dataset.CIFAR10(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)
model = Unet(
    dim = 128,
    channels=channels,
    dim_mults = (1, 2, 2, 2)
)
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l2',   # L1 or L2
    beta_schedule='linear',
    channels=channels,
)

trainer = Trainer(diffusion, ds, gradient_accumulate_every=8, train_num_steps=80*10**4, save_and_sample_every=10*10**3, num_samples=64, ema_update_every=10, ema_decay=0.9999, train_lr=2*10**-4, results_folder=f'./results/{dataset_name}/')
#trainer = Trainer(diffusion, ds, train_num_steps=11, save_and_sample_every=10, num_samples=4, ema_update_every=5, ema_decay=0.9999, train_lr=2*10**-4, results_folder=f'./results/{dataset_name}/')
#trainer.load(1, '../denoising-diffusion-pytorch/results')
trainer.load(21, model_ext='pkl')
#batches = num_to_groups(64, batch_size)
#all_images_list = list(map((lambda n: trainer.model.sample(batch_size=n)), batches))
#all_images = jt.contrib.concat(all_images_list, dim=0)
#all_images = all_images.expand(-1, 3, *all_images.shape[2:])
#jt.save_image(all_images, str((trainer.results_folder / f'sample-test.png')), nrow=int(math.sqrt(trainer.num_samples)))
trainer.train()
#trainer.autodiff()
