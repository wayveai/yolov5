from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
import sys
from typing import List, Union, Dict, Tuple
from pathlib import Path
import torch
from tqdm import tqdm
import json
from threading import Thread
from PIL import Image
import numpy as np

sys.path.append('/mnt/remote/data/users/thomasssajot/yolov5/')
from utils.general import non_max_suppression


from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

import hydra
from hydra.utils import get_original_cwd

import logging 

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


def get_model(path, device: int, half_precision: bool = False):
    assert isinstance(half_precision, bool)

    # model = torch.hub.load(
    #     # repo_or_dir='/mnt/remote/data/users/thomasssajot/yolov5/', 
    #     model='custom', 
    #     device=device,
    #     path=path, 
    #     source='local',
    #     autoshape=False
    # )
    model = torch.jit.load(path, map_location=device)
    model.eval()
    if half_precision:
        logging.info('Using half precision')
        model.half()
    else:
        logging.info('Using full precision')
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
    logging.info('Checked all images exists.')
    return [str(i) for i in images]


class ImageDataset(Dataset):
    def __init__(self, files: List[Union[str, Path]], size: Tuple[int, int], crop: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        self.crop = crop # left upper right lower
        self.size = size # width / height
        self.files = list(map(Path, files))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(file).resize(self.size, resample=Image.Resampling.BILINEAR).crop(self.crop)
        normalized_image = np.array(image, dtype=np.float32) / 255
        return dict(file=str(file), image=torch.from_numpy(normalized_image).permute(2, 0, 1))


def _save_example_image(image_file, cfg):
    batch = ImageDataset([image_file], size=cfg.data.image_size, crop=cfg.data.image_crop)
    img_tensor = batch[0]['image']
    img = (img_tensor.cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f'./example_{img.size[0]}x{img.size[1]}.jpeg')


@torch.no_grad()
def run_inference_single_gpu(device: str, images, cfg) -> Dict[str, str]:
    width, height = cfg.data.image_size
    left, upper, *_ = cfg.data.image_crop
    model = get_model(cfg.weights, device=device, half_precision=cfg.half_precision)
    metadata = dict(
        model=str(cfg.weights), 
        # names=model.names,
        image_size=dict(width=cfg.data.image_size[0], height=cfg.data.image_size[1]), 
        image_crop={k:v for k, v in zip(['left', 'upper', 'right', 'lower'], cfg.data.image_crop)},
        image_root=str(cfg.data.path)
    )

    loader = DataLoader(
        dataset=ImageDataset(images, size=cfg.data.image_size, crop=cfg.data.image_crop), 
        batch_size=cfg.data.batch_size, 
        shuffle=False, num_workers=cfg.data.num_workers
    ) 
    results = dict()
    gen = tqdm(loader, total=len(loader), ncols=140, desc=f'Running predictions')
    image_crop_shift = np.array([left, upper, left, upper])   # shift by image crop
    image_crop_size = np.array([width, height, width, height])
    for batch in gen:
        x = batch['image'].to(device)
        if cfg.half_precision:
            x = x.half()
        y = model(x)
        y = non_max_suppression(y, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=500) 

        for image_file, pred in zip(batch['file'], y):
            pred = pred.cpu().detach().numpy()
            # inverse the preprocessing image transformation
            # in order to have the bounding boxes in the px coordinates of the original image
            pred[:, (0, 1, 2, 3)] = (pred[:, (0, 1, 2, 3)] + image_crop_shift) / image_crop_size
            results[image_file] = pred.tolist()

    return results, metadata


@hydra.main(version_base=None, config_path="./configs/generate_predictions", config_name="1600x960_with_crop.yaml")
def main(cfg):
    """Run a yolov5 model inference over ENTRON dataset

    THis script is using Hydra. The working directory will be automatically changed to 
    what is specified in the hydra config_path @ run > hydra > dir (currently `./evaluations/predictions/bbox/{date}`)

    The directories meant not to be meaningful. If you wish to understand the difference between the predictions, 
    look into the config file that is saved within them.
    """
    cfg.dst = Path(cfg.dst)
    cfg.weights = Path(get_original_cwd()) / cfg.weights
    cfg.data.path = Path(cfg.data.path).expanduser()
    if isinstance(cfg.devices, int):
        cfg.devices = [cfg.devices]
    # cfg.devices = [f'mp_inference:cuda:{d}' for d in cfg.devices]
    cfg.devices = [f'cuda:{d}' for d in cfg.devices]
    
    logging.info('Starting script')
    logging.info('Config %s', str(cfg))

    images = get_image_files(cfg.data.path)
    logging.info('Saving an example image')
    _save_example_image(images[0], cfg)

    n = len(cfg.devices)
    
    if len(cfg.devices) == 1:
        predictions, metadata = run_inference_single_gpu(cfg.devices[0], images, cfg)
    else:
        with ProcessPoolExecutor(max_workers=len(cfg.devices)) as executor:
            futures = [
                executor.submit(
                    run_inference_single_gpu,
                    device=device,
                    images=[im for im in images if hash(im) % n == i],
                    cfg=cfg,
                )
                for i, device in enumerate(cfg.devices)
            ]
            # if one future raises and exception, raise ASAP
            done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
        done = [f.result() for f in done]
        metadata = done[0][1]
        predictions = {k: v for d, _ in done for k, v in d.items()}

    logging.info('Saving to file %i predictions', len(predictions))
    with cfg.dst.open('w') as f:
        json.dump(dict(metadata=metadata, predictions=predictions), f, indent=4)
    logging.info('Done saving file.')


if __name__ == "__main__":
    main()
