import streamlit as st
import pandas as pd
import json
from pathlib import Path
from contants import POSSIBLE_LABELS, COLOUR_SIMPLIFICATION_MAP



@st.cache
def load_entron_v2(root):
    ground_truth = pd.Series(list(root.glob('**/*.jpeg'))).to_frame(name='file')

    ground_truth['file'] = ground_truth['file'].map(lambda f: str(f.relative_to(root)))
    ground_truth['gt'] = ground_truth['file'].str.split('/', 1).str.get(0)
    # the entron dataset v2 has a known miss label with UNKNOWN example
    ground_truth = ground_truth[ground_truth['gt'] != 'UNKNOWN'].reset_index(drop=True)
    # simplify Arrows to solid colours
    ground_truth['gt'] = ground_truth['gt'].map(lambda x: COLOUR_SIMPLIFICATION_MAP.get(x, x))
    assert ground_truth['gt'].isin(POSSIBLE_LABELS).all(), str(ground_truth['gt'][~ground_truth['gt'].isin(POSSIBLE_LABELS)])
    return ground_truth



def predictions_to_dataframe(payload: dict) -> pd.DataFrame:
    """
    Get the dictionary of predictions and generate a dataframe with predicted and ground truth values
    Ground truth values are coming from the name of the parent directory
    The datast is assumed to be Traffic Light Entron V2
    """

    annotations = []
    for file, anns in payload['predictions'].items():
        for row in anns:
            annotations.append(row + [file])

    preds_df = pd.DataFrame(annotations, columns=['x0', 'y0', 'x1', 'y1', 'confidence', 'class', 'file'])
    preds_df['class'] = preds_df['class'].astype(int).astype(str)
    preds_df['name'] = preds_df['class'].map(payload['metadata']['names'].get)
    preds_df['run_id'] = preds_df['file'].str.split('/', 1).str.get(1).str.rsplit('/', 1).str.get(0)
    preds_df['ts'] = preds_df['file'].str.rsplit('/', 1).str.get(1).str.replace('unixus.jpeg', '', regex=False).astype(int)
    return preds_df.reset_index()


@st.cache
def load_predictions(file: Path) -> dict:
    with file.open('r') as f:
        payload =  json.load(f)

    img_root = Path(payload['metadata']['image_root'])
    for k in list(payload['predictions']):
        new_k = str(Path(k).relative_to(img_root))
        payload['predictions'][new_k] = payload['predictions'][k]
        del payload['predictions'][k]
    return payload['metadata'], predictions_to_dataframe(payload)


def get_model_type(metadata: dict):
    """Given the type of classes predicted decide if the model is CROSS PRODUCT or Multi label"""
    labels = list(metadata['names'].values())
    if 'RELEVANT' in labels and 'NON_RELEVANT' in labels:
        return 'multi_label'
    assert all(x.endswith('_RELEVANT') for x in labels), labels
    return 'cross_prod'

def get_classification_df(predictions: pd.DataFrame, gt: pd.DataFrame, metadata: dict):
    model_label_type = get_model_type(metadata)

    if model_label_type == 'cross_prod':
        mask = predictions['name'].isin(['RED_SOLID_RELEVANT', 'GREEN_SOLID_RELEVANT', 'AMBER_SOLID_RELEVANT', 'RED_AND_AMBER_RELEVANT'])
        if mask.sum() <= 100:
            st.error('There are almost no outputs with the values we are looking for. Are you looking for the right output classes?')
            st.stop() 
        classification_preds = predictions[mask].reset_index(drop=True)
        classification_preds = classification_preds.sort_values(['file', 'confidence'], ascending=False).drop_duplicates(['file']).reset_index(drop=True)
        classification_preds['pred'] = classification_preds['name'].str.replace('_RELEVANT', '')

    elif model_label_type == 'multi_label':
        mask = predictions['name'].isin(['RED_SOLID', 'AMBER_SOLID', 'GREEN_SOLID', 'RED_AND_AMBER', 'RELEVANT', 'NON_RELEVANT'])
        classification_preds = predictions[mask].reset_index(drop=True)
        classification_preds['area'] = area(classification_preds)
        classification_preds = classification_preds.groupby('file').apply(extract_traffic_light_colour_from_multi_label)
        st.dataframe(classification_preds)

    assert classification_preds['pred'].isin(POSSIBLE_LABELS).all()
    classification_preds = classification_preds.merge(gt, how='right', on='file')
    classification_preds['pred'] = classification_preds['pred'].fillna('NONE')
    classification_preds['confidence'] = classification_preds['confidence'].fillna(0)
    classification_preds['is_true'] = classification_preds['gt'] == classification_preds['pred']
    classification_preds = classification_preds.sort_values('is_true').reset_index(drop=True)
    return classification_preds


def extract_traffic_light_colour_from_multi_label(df: pd.DataFrame) -> dict:
    """
    Given the most relevant prediction, get the largest overlapping traffic light bounding box with a colour
    Return the colour or 'NONE'
    """
    relevant = df.query('name == "RELEVANT"')
    df_colours = df[df['name'].isin(['RED_SOLID', 'AMBER_SOLID', 'GREEN_SOLID', 'RED_AND_AMBER'])]

    if len(relevant) == 0 or len(df_colours) == 0:
        colour = 'NONE'
        conf = 0
    else:
        relevant = relevant.sort_values('confidence', ascending=False).iloc[0]
        row_idx = df_colours.apply(lambda y: area_overlap(relevant, y), axis=1).argmax()
        colour, conf = df_colours.iloc[row_idx][['name', 'confidence']]

    return pd.Series(dict(pred=colour, confidence=conf))

def area(df):
    return (df.x1 - df.x0).abs() * (df.y1 - df.y0).abs()


def area_overlap(a, b) -> float:
    dx = abs(min(a.x1, b.x1) - max(a.x0, b.x0))
    dy = abs(min(a.y1, b.y1) - max(a.y0, b.y0))
    if (dx>=0) and (dy>=0):
        return dx*dy / min(a.area, b.area)
    return 0