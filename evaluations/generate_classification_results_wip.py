from concurrent.futures import thread
from email.mime import image
import sys
from typing import Dict, Optional, Union
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import json
import threading

sys.path.append('/mnt/remote/data/users/thomasssajot/yolov5/')


import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


def area(a, b) -> float:
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy / min(area_b, area_b)
    return 0

def get_colour_of_most_relevant(df: pd.DataFrame):
    """
    Given the most relevant prediction, get the largest overlapping traffic light bounding box with a colour
    Return the colour or 'NONE'
    """
    relevant = df.query('name == "RELEVANT"')
    df_colours = df[df['class'] <= 5]
    
    if len(relevant) == 0 or len(df_colours) == 0:
        return 'NONE'
    relevant = relevant.sort_values('confidence', ascending=False).iloc[0]
    df_colours = df[df['class'] <= 5]
    row_idx = df_colours.apply(lambda y: area(relevant, y), axis=1).argmax()
    return df_colours.iloc[row_idx]['name']

def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

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

@record
def run_inference(rank, world_size, dataset, image_size, result_queue:mp.Queue) -> Dict[str, str]:
    """https://discuss.pytorch.org/t/how-do-i-run-inference-in-parallel/126757"""
    # create default process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    model = get_model().to(rank)

    dataset = [i for i in dataset if hash(i) % world_size == rank]
    print(f'Processing {len(dataset)} images in rank {rank}.')
    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=8) 

    with torch.no_grad():
        for batch in tqdm(loader, ncols=140, desc=f'Predictions on rank {rank}'):
            res = model(batch, size=image_size)
            for image_file, df in zip(batch, res.pandas().xyxyn):
                label = get_colour_of_most_relevant(df)
                result_queue.put((image_file, label, rank))

def get_image_files(root: Optional[Union[Path, str]] = None):
    if root is None:
        root = Path('/mnt/remote/data/users/thomasssajot/yolo_dataset/traffic_lights_entron_classification/')
    root = Path(root)
    print('Looking for images in:', root)

    images = list(root.glob('**/*.jpeg'))
    print(f'Found {len(images)} images.')
    assert all([f.exists() for f in images])

    return list(map(str, images))

# def main():
    # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/5
    # https://discuss.pytorch.org/t/segfault-with-multiprocessing-queue/81292

    # TODO
    # Explore; https://discuss.pytorch.org/t/evaluate-multiple-models-on-multiple-gpus/62451/5

    # root = Path('/mnt/remote/data/users/thomasssajot/yolo_dataset/traffic_light_entron_classification/')
    # images = list(root.glob('**/*.jpeg'))
    # print(f'Found {len(images)} images.')
    # assert all([f.exists() for f in images])

    # images = list(map(str, images))

 
    # mp.set_start_method('spawn', force=True)
    # result_queue = mp.Queue()
    # world_size = 4
    # mp.spawn(
    #     run_inference, 
    #     args=(world_size, images[:64 * world_size], 1280, result_queue), 
    #     nprocs=world_size
    # ).join(timeout=10)
    # print('Out of loop')
    # while not result_queue.empty():
    #     print(result_queue.get())

# Best attempt at parallel inference
# def main():
#     images = get_image_files()
#     model_path = '/mnt/remote/data/users/thomasssajot/yolov5/runs/traffic_light_2020_undistorted/yolov5x6_1280_multi_label/weights/best.pt'
#     model = get_model(model_path)
#     net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

#     loader = DataLoader(dataset=images[:64 * 4], batch_size=4, shuffle=False, num_workers=8) 

#     predictions = dict()
#     with torch.no_grad():
#         for batch in tqdm(loader, ncols=140, desc=f'Predictions'):
#             res = net(batch, size=1280)
#             for image_file, df in zip(batch, res.pandas().xyxyn):
#                 label = get_colour_of_most_relevant(df)
#                 predictions[image_file] = label
#     print(predictions)



def main_single_thread():
    print('Main single threaded')
    image_size = 1280
    device = 4
    batch_size = 32
    dst_file = Path('/mnt/remote/data/users/thomasssajot/yolov5/notebooks/cache/yolov5x6_1280_multi_label_entron_tl_classification_1280_get_colour_most_relevant.txt')

    model_path = '/mnt/remote/data/users/thomasssajot/yolov5/runs/traffic_light_2020_undistorted/yolov5x6_1280_multi_label/weights/best.pt'
    model = get_model(model_path, device=device)
    images = get_image_files()

    if dst_file.exists():
        with dst_file.open('r') as f:
            predictions = json.load(f)
        len_before = len(images)
        images = [i for i in images if i not in predictions]
        print(f'Found {len_before - len(images)} predictions already saved. Predicting for remaing {len(images)}.')

    else:
        predictions = dict()
        


    loader = DataLoader(dataset=images, batch_size=batch_size, shuffle=False, num_workers=16) 

    with torch.no_grad():
        for batch in tqdm(loader, ncols=140, desc=f'Predictions'):
            res = model(batch, size=image_size)
            for image_file, df in zip(batch, res.pandas().xyxyn):
                label = get_colour_of_most_relevant(df)
                predictions[image_file] = label
    with Path('/mnt/remote/data/users/thomasssajot/yolov5/notebooks/cache/yolov5x6_1280_multi_label_entron_tl_classification_1280_get_colour_most_relevant.txt').open('w') as f:
        json.dump(predictions, f)




# Multi threading
def run_inference(model, dataset, image_size, batch_size: int = 32) -> Dict[str, str]:
    print(f'Processing {len(dataset)} images on  device {model.model.device}.')
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8) 

    results = dict()
    with torch.no_grad():
        for batch in tqdm(loader, ncols=140, desc=f'Predictions on device {model.model.device}'):
            res = model(batch, size=image_size)
            for image_file, df in zip(batch, res.pandas().xyxyn):
                label = get_colour_of_most_relevant(df)
                results[image_file] = label
    return results 

def main_multi_threading():
    print('main_multi_threading')
    image_size = 1280
    batch_size = 4
    model_path = '/mnt/remote/data/users/thomasssajot/yolov5/runs/traffic_light_2020_undistorted/yolov5x6_1280_multi_label/weights/best.pt'
    model6 = get_model(model_path, device='5,6')
    model5 = get_model(model_path, device='6,5')
    print([m.model.device for m in [model5, model6]])

    images = get_image_files('/mnt/remote/data/users/thomasssajot/yolo_dataset/traffic_lights_entron_classification/focal_len=650__sensor_size_hw=1200x1920/GREEN_SOLID/brizo/')
    model6(images[:64])
    model5(images[:64])
    # threads = [threading.Thread(target=run_inference, args=[m, [im for im in images if hash(im) % n == i], image_size, batch_size], daemon=True) for i, m in enumerate(models)]
    # for thread in threads:
    #     thread.start()

    # res = []
    # for thread in threads:
    #     res.append(thread.join())
    # print(res)

    # with Path('/mnt/remote/data/users/thomasssajot/yolov5/notebooks/cache/yolov5x6_1280_multi_label_entron_tl_classification_1280_get_colour_most_relevant.txt').open('w') as f:
    #     json.dump(predictions, f)


if __name__ == "__main__":
    main_multi_threading()

    