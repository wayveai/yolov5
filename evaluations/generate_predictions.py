import argparse
import sys
from typing import List, Union, Dict, Tuple
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import json
from threading import Thread
from PIL import Image
import numpy as np

sys.path.append('/mnt/remote/data/users/thomasssajot/yolov5/')
from utils.general import non_max_suppression

from torch import nn

from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

from omegaconf import OmegaConf
import hydra
from hydra.utils import get_original_cwd, to_absolute_path

import logging 

log = logging.getLogger(__name__)


def get_model(path, device: int):
    model = torch.hub.load(
        repo_or_dir='/mnt/remote/data/users/thomasssajot/yolov5/', 
        model='custom', 
        device=device,
        path=path, 
        source='local'
    ) 
    model.eval()
    return model


def get_image_files(root: Union[Path, str]) -> List[str]:
    """Find all .jpeg files in nested directories"""
    root = Path(root)
    assert root.exists()
    root = root.resolve()
    logging.info('Looking for images in: %s', root)

    images = [f for f in tqdm(root.glob('**/*.jpeg'), ncols=120, desc="Loading images")]
    logging.info('Found %i images.', len(images))
    assert all([f.exists() for f in tqdm(images, ncols=120, desc="Checking if images exists.")])
    return [str(i) for i in images]


class ImageDataset(Dataset):

    def __init__(self, files: List[Union[str, Path]], size: Tuple[int, int], crop: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        self.crop = crop # left upper right lower
        self.size = size # width / height
        self.files = [Path(f) for f in files]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(file).crop(self.crop).resize(self.size)
        normalized_image = np.array(image) / 255
        return dict(file=str(file), image=torch.from_numpy(normalized_image).permute(2, 0, 1))


def _save_example_image(image_file, cfg):
    img_tensor = ImageDataset([image_file], size=cfg.data.image_size, crop=cfg.data.image_crop)[0]['image']
    img = (img_tensor.cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f'./example_{img.size[0]}x{img.size[1]}.jpeg')


@hydra.main(version_base=None, config_path="./configs/generate_predictions", config_name="1600x960_with_crop.yaml")
def main(cfg):
    cfg.dst = Path(cfg.dst)
    cfg.weights = Path(get_original_cwd()) / cfg.weights
    cfg.data.path = Path(cfg.data.path).expanduser()
    cfg.devices = [f'mp_inference:cuda:{d}' for d in cfg.devices]
    print(cfg)

    print('Main Multi-Threaded')
    print(cfg.data)

    images = get_image_files(cfg.data.path)
    _save_example_image(images[0], cfg)


    n = len(cfg.devices)
    models = [get_model(cfg.weights, device=d) for d in cfg.devices]
    metadata = dict(
        model=str(cfg.weights), 
        names=models[0].names, 
        image_size=dict(width=cfg.data.image_size[0], height=cfg.data.image_size[1]), 
        image_crop={k:v for k, v in zip(['left', 'upper', 'right', 'lower'], cfg.data.image_crop)},
        image_root=str(cfg.data)
    )
    
    result_queue = mp.Queue()

    def run_inference(model, dataset, queue: mp.Queue) -> Dict[str, str]:
        width, height = cfg.data.image_size
        left, upper, *_ = cfg.data.image_crop
        loader = DataLoader(
            dataset=ImageDataset(dataset, size=cfg.data.image_size, crop=cfg.data.image_crop), 
            batch_size=cfg.data.batch_size, 
            shuffle=False, num_workers=cfg.data.num_workers
        ) 
        gen = tqdm(loader, ncols=140, desc=f'Running predictions')
        with torch.no_grad():
            for batch in gen:
                y = model(batch['image'])
                y = non_max_suppression(y, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=1000) 
                preds = []
                for x in y:
                    x = x.cpu().detach().numpy()
                    x[:, (0, 1, 2, 3)] += (left, upper, left, upper)   # shift by image crop
                    x[:, (0, 1, 2, 3)] /= (width, height, width, height)  # x0 / width, y0 / height, x1 / width, y1 / height
                    preds.append(x)
                
                for image_file, labels in zip(batch['file'], preds):
                    queue.put((image_file, labels.tolist()))
    
    if len(cfg.devices) == 1:
        run_inference(models[0], result_queue)
    else:
        threads = [
            Thread(
                target=run_inference, 
                args=[m, [im for im in images if hash(im) % n == i], result_queue], 
                daemon=True
            ) for i, m in enumerate(models)
        ]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    predictions = dict()
    while not result_queue.empty():
        f, l = result_queue.get()
        predictions[f] = l

    print(f'Saving to file {len(predictions)} predictions')
    with cfg.dst.open('w') as f:
        json.dump(dict(metadata=metadata, predictions=predictions), f, indent=4)
    print('Done saving file.')


if __name__ == "__main__":
    main()
    