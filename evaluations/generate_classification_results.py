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

def area(a, b) -> float:
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy / min(area_b, area_b)
    return 0


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


def get_image_files(root: Union[Path, str]):
    root = Path(root)
    print('Looking for images in:', root)

    images = [f for f in tqdm(root.glob('**/*.jpeg'), ncols=120, desc="Loading images")]
    print(f'Found {len(images)} images.')
    assert all([f.exists() for f in tqdm(images, ncols=120, desc="Checking if images exists.")])

    return [str(i.resolve()) for i in images]


def extract_traffic_light_colour_from_cross_product(df: pd.DataFrame) -> dict:
    """
    Given the most relevant prediction, get the largest overlapping traffic light bounding box with a colour
    Return the colour or 'NONE'
    """
    mask = df['class'].isin([5, 7, 9, 11])
    if  len(df) == 0 or not mask.any():
        traffic_light_colour = 'NONE'
    else:
        most_relevant_idx = df[mask]['confidence'].idxmax()
        traffic_light_colour, traffic_light_conf = df.loc[most_relevant_idx][['name', 'confidence']]
        traffic_light_colour = traffic_light_colour.replace('_RELEVANT', '')
    return dict(colour=traffic_light_colour, conf=traffic_light_conf)


def extract_traffic_light_colour_from_multi_label(df: pd.DataFrame) -> dict:
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
    traffic_light_colour, traffic_light_conf = df_colours.iloc[row_idx][['name', 'confidence']]
    return dict(colour=traffic_light_colour, conf=traffic_light_conf)


# Single process
def main(dst_file: str, model_path: str, image_root: str, post_processing_func, device: List[str], image_size: int = 1280, batch_size: int = 32):
    print('Main single threaded')
    dst_file = Path(dst_file)

    assert dst_file.parent.exists()

    metadata = dict(model=str(model_path), image_size=image_size, image_root=image_root)
    model = get_model(model_path, device=device)
    images = get_image_files(image_root)

    if dst_file.exists():
        with dst_file.open('r') as f:
            predictions = json.load(f)
        assert metadata == predictions.get('metadata'), f"Metadata are different {metadata} != {predictions.get('metadata')}"
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
                label = post_processing_func(df)
                predictions[image_file] = label

    with dst_file.open('w') as f:
        json.dump(dict(predictions=predictions, metadata=metadata), f, indent=4)



def main_mp(dst_file: str, model_path: str, image_root: str, post_processing_func, device: List[str], image_size: int = 1280, batch_size: int = 32):
    print('Main Multi-Threaded')
    dst_file = Path(dst_file)

    assert dst_file.parent.exists()

    metadata = dict(model=str(model_path), image_size=image_size, image_root=image_root)
    n = len(device)
    models = [get_model(model_path, device=d) for d in device]
    images = get_image_files(image_root)

    if dst_file.exists():
        with dst_file.open('r') as f:
            predictions = json.load(f)
        assert metadata == predictions.get('metadata'), f"Metadata are different {metadata} != {predictions.get('metadata')}"
        len_before = len(images)
        images = [i for i in images if i not in predictions]
        print(f'Found {len_before - len(images)} predictions already saved. Predicting for remaing {len(images)}.')
    else:
        predictions = dict()

    result_queue = mp.Queue()

    def run_inference(model, dataset, image_size, queue: mp.Queue,  batch_size: int = 32) -> Dict[str, str]:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8) 
        with torch.no_grad():
            for batch in tqdm(loader, ncols=140, desc=f'Predictions on device'):
                res = model(batch, size=image_size)
                for image_file, df in zip(batch, res.pandas().xyxyn):
                    label = post_processing_func(df)
                    queue.put((image_file, label))
    
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
        json.dump(dict(predictions=predictions, metadata=metadata), f, indent=4)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '--model-path', type=str, help='model path(s)')
    parser.add_argument('--data', type=str, help='images root directory')
    parser.add_argument('--label-fmt', type=str, choices=['cross-prod', 'multi-label'], help='Is the model predicting cross produc labels or multi-labels')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=1280, help='inference size h,w')
    parser.add_argument('--batch-size', '--bs', type=int, default=32, help='inference size h,w')
    parser.add_argument('--device', default=[0], type=int, nargs='+')
    parser.add_argument('--dst', type=str, help='save results into file')
    opt = parser.parse_args()
    if isinstance(opt.device, int):
        opt.device = [opt.device]
    opt.device = [f'mp_inference:cuda:{i}' for i in opt.device]
    print(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    if opt.label_fmt == 'cross-prod':
        post_processing_func = extract_traffic_light_colour_from_cross_product
    elif opt.label_fmt == 'multi-label':
        post_processing_func = extract_traffic_light_colour_from_multi_label
    else:
        raise ValueError()

    main_mp(
        dst_file=opt.dst, 
        model_path=opt.weights, 
        image_root=opt.data,
        image_size=opt.imgsz,
        post_processing_func=post_processing_func, 
        device=opt.device, 
        batch_size=opt.batch_size
    )
    # main(
    #     dst_file='/mnt/remote/data/users/thomasssajot/yolov5/evaluations/cache/yolov5x6_1280_multi_label_entron_tl_classification_1280_get_colour_most_relevant.txt', 
    #     model_path='/mnt/remote/data/users/thomasssajot/yolov5/runs/traffic_light_2020_undistorted/yolov5x6_1280_multi_label/weights/best.pt', 
    #     image_root='/mnt/remote/data/users/thomasssajot/yolo_dataset/traffic_lights_entron_classification/',
    #     multi_label=True, 
    #     image_size=1280, 
    #     device=4, 
    #     batch_size=32
    # )

    