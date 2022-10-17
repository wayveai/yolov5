from pathlib import Path


RESULTS_ROOT = Path('../predictions/bbox')
GROUND_TRUTH_ROOT = Path(
    '/mnt/remote/data/users/thomasssajot/yolo_dataset/traffic_lights_entron_classification/focal_len=650__sensor_size_hw=1200x1920'
).resolve()

POSSIBLE_LABELS = ['NONE', 'GREEN_SOLID', 'RED_SOLID', 'AMBER_SOLID', 'RED_AND_AMBER']

COLOUR_SIMPLIFICATION_MAP = {
    'GREEN_ARROW_LEFT': 'GREEN_SOLID',
    'GREEN_ARROW_RIGHT': 'GREEN_SOLID',
    'GREEN_ARROW_STRAIGHT': 'GREEN_SOLID',
    'UNKNOWN': 'NONE',
    'RED_ARROW_RIGHT': 'RED_SOLID',
    'RED_ARROW_STRAIGHT': 'RED_SOLID',
    'RED_ARROW_LEFT': 'RED_SOLID',
}
