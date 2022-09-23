import jittor as jt
from jittor import dataset
from PIL import Image
from pathlib import Path

save_dir = Path('./dataset/cifar10')
save_dir.mkdir(exist_ok=True, parents=True)

ds = dataset.CIFAR10(train=True).set_attrs(batch_size=1, keep_numpy_array=True)
for i, (img, _) in enumerate(ds):
    im = Image.fromarray(img[0])
    im.save(save_dir / f'image{i:05}.png')

