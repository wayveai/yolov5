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
    print('Looking for images in:', root)

    images = [f for f in tqdm(root.glob('**/*.jpeg'), ncols=120, desc="Loading images")]
    print(f'Found {len(images)} images.')
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


def main_mp(
        dst_file: Path, 
        model_path: Path, 
        image_root: Path, 
        device: List[str], 
        image_size: Tuple[int, int], 
        image_crop: Tuple[int, int, int, int],
        batch_size: int = 32
    ):

    print('Main Multi-Threaded')
    print(image_root)

    n = len(device)
    models = [get_model(model_path, device=d) for d in device]
    metadata = dict(
        model=str(model_path), 
        names=models[0].names, 
        image_size=dict(width=image_size[0], height=image_size[1]), 
        image_crop={k:v for k, v in zip(['left', 'upper', 'right', 'lower'], image_crop)},
        image_root=str(image_root)
    )

    images = get_image_files(image_root)

    # save an image to see if it looks ok
    img = ImageDataset(images[:1], size=image_size, crop=image_crop)[0]
    Image.fromarray(img.cpu().detach().transpose(1, 2, 0)).save('./example.jpeg')

    result_queue = mp.Queue()

    def run_inference(model, dataset, queue: mp.Queue,  batch_size: int = 32) -> Dict[str, str]:
        width, height = image_size
        left, upper, *_ = image_crop
        loader = DataLoader(dataset=ImageDataset(dataset, size=image_size, crop=image_crop), batch_size=batch_size, shuffle=False, num_workers=8) 
        with torch.no_grad():
            for batch in tqdm(loader, ncols=140, desc=f'Predictions on device'):
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
        run_inference(models[0], images, result_queue, cfg.batch_size)
    else:
        threads = [
            Thread(
                target=run_inference, 
                args=[m, [im for im in images if hash(im) % n == i], result_queue, cfg.batch_size], 
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
    with dst_file.open('w') as f:
        json.dump(dict(metadata=metadata, predictions=predictions), f, indent=4)
    print('Done saving file.')



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '--model-path', type=str, help='model path(s)')
    parser.add_argument('--data', type=str, help='images root directory')
    parser.add_argument('--width', type=int, default=1280, help='inference size width')
    parser.add_argument('--height', type=int, default=832, help='inference size height')
    parser.add_argument('--crop', default=(0, 0, 1920, 1200), type=int, nargs=4)
    parser.add_argument('--batch-size', '--bs', type=int, default=32, help='inference size h,w')
    parser.add_argument('--devices', default=[0], type=int, nargs='+')
    parser.add_argument('--dst', type=str, help='save results into file')
    opt = parser.parse_args()

    if isinstance(opt.devices, int):
        opt.devices = [opt.devices]
    opt.devices = [f'mp_inference:cuda:{d}' for d in opt.devices]
    opt.data = Path(opt.data).resolve()
    opt.dst = Path(opt.dst).resolve()
    if opt.dst.exists() and opt.dst.name == 'dump.json':
        print('Deleting dump file.')
        opt.dst.unlink()

 
    assert opt.data.exists()
    assert opt.dst.parent.exists()
    if opt.dst.exists():
        raise FileExistsError(f'Destination file already exists: {opt.dst}')

    return opt


if __name__ == "__main__":
    print('\n' * 2)
    print('=' * 100)

    opt = parse_opt()

    main_mp(
        dst_file=opt.dst, 
        model_path=opt.weights, 
        image_root=opt.data,
        image_size=(opt.width, opt.height),
        image_crop=opt.crop,
        device=opt.devices, 
        batch_size=opt.batch_size
    )

    