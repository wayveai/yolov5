import argparse
import sys
from typing import List, Union, Dict
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import json
from threading import Thread

sys.path.append('/mnt/remote/data/users/thomasssajot/yolov5/')

from torch.utils.data import DataLoader
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



def main_mp(dst_file: Path, model_path: Path, image_root: Path, device: List[str], image_size: int = 1280, batch_size: int = 32):
    print('Main Multi-Threaded')
    print(image_root)

    n = len(device)
    models = [get_model(model_path, device=d) for d in device]
    metadata = dict(model=str(model_path), names=models[0].names, image_size=image_size, image_root=str(image_root))
    images = get_image_files(image_root)

    result_queue = mp.Queue()

    def run_inference(model, dataset, image_size, queue: mp.Queue,  batch_size: int = 32) -> Dict[str, str]:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8) 
        with torch.no_grad():
            for batch in tqdm(loader, ncols=140, desc=f'Predictions on device'):
                res = model(batch, size=image_size)
                for image_file, labels in zip(batch, res.xyxyn):
                    labels = labels.cpu().detach().numpy().tolist()
                    queue.put((image_file, labels))
    
    threads = [
        Thread(
            target=run_inference, 
            args=[m, [im for im in images if hash(im) % n == i], image_size, result_queue, batch_size], 
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
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=1280, help='inference size h,w')
    parser.add_argument('--batch-size', '--bs', type=int, default=32, help='inference size h,w')
    parser.add_argument('--device', default=[0], type=int, nargs='+')
    parser.add_argument('--dst', type=str, help='save results into file')
    opt = parser.parse_args()

    if isinstance(opt.device, int):
        opt.device = [opt.device]
    opt.device = [f'mp_inference:cuda:{i}' for i in opt.device]
    opt.data = Path(opt.data).resolve()
    opt.dst = Path(opt.dst).resolve()

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
        image_size=opt.imgsz,
        device=opt.device, 
        batch_size=opt.batch_size
    )

    