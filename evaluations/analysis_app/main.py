import json
from pathlib import Path
from typing import Dict
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from loaders import load_entron_v2, load_predictions, get_classification_df
from contants import POSSIBLE_LABELS, RESULTS_ROOT, GROUND_TRUTH_ROOT
from utils import add_tablet, display_classification_report, display_confusion_matrix, display_2d_predictions, display_histograms

# cache classification_report to make it faster
classification_report = st.cache(classification_report)


st.set_page_config(page_title='Yolo TL CLS', layout='wide', page_icon='🔬')

st.title('Analysis tool')

st.write('Ground truth directory:', GROUND_TRUTH_ROOT, '| exists:', GROUND_TRUTH_ROOT.exists())
with st.spinner('Loading ground truth'):
    ground_truth = load_entron_v2(GROUND_TRUTH_ROOT)

st.info(f'Ground truth dataset size: {len(ground_truth)}')

no_selection = 'select file'
prediction_file = st.selectbox(
    'Select model predictions', 
    [no_selection] + sorted(list(RESULTS_ROOT.glob('**/*.json'))), 
    format_func=lambda x: x if x == no_selection else add_tablet(x.name)
)
if prediction_file == no_selection:
    st.stop()

metadata, predictions_df = load_predictions(prediction_file)
with st.expander('Metadata:'):
    st.json(metadata)

st.info(f'Dataset size: {predictions_df["file"].nunique()} images | {len(predictions_df)} predicted objects')

with st.spinner('Converting preds to classification'):
    classification_df = get_classification_df(predictions_df, ground_truth, metadata)

st.header('Classification report')
st.markdown('##### ' + prediction_file.name)
cls_report = classification_report(y_true=classification_df['gt'], y_pred=classification_df['pred'], digits=3)
col1, col2, _ = st.columns([3, 2, 2])
with col1:
    display_classification_report(cls_report)
with col2:
    display_confusion_matrix(y_true=classification_df['gt'], y_pred=classification_df['pred'], labels=POSSIBLE_LABELS)


with st.expander('Histograms'):
    display_histograms(classification_df)

st.header('Inspect object detections')

col1, col2, col3 = st.columns([2, 2, 4])
any_label = 'Any label'
selected_label = col1.selectbox('Choose GT label', [any_label] + POSSIBLE_LABELS)
selected_pred_label = col2.selectbox('Choose Pred label', [any_label] + POSSIBLE_LABELS)

query_string = []
if selected_label != any_label:
    query_string.append(f'gt == "{selected_label}" ')
if selected_pred_label != any_label:
    query_string.append(f'pred == "{selected_pred_label}"')

images = classification_df.query(' and '.join(query_string))['file'].unique() if query_string else classification_df['file'].unique()
images = sorted(images)

image_name_contains = st.text_input('Specify image should contains')
selected_image = st.selectbox("Choose image", [i for i in images if image_name_contains in i])

st.info(f'Found {len(images)} images for this filter')

idx = st.number_input("Select image", 0, len(images), 0)
image_file = selected_image
annotations = predictions_df.query(f'file == "{image_file}"')

if annotations.empty:
    st.warning('No traffic lights detected')
else:
    st.dataframe(annotations, width=1500)

st.write(f'Diplay All preds')
display_2d_predictions(GROUND_TRUTH_ROOT / image_file, annotations)

for pred_label, anns in annotations.groupby('name'):
    st.write(f'Diplay {pred_label}')
    display_2d_predictions(GROUND_TRUTH_ROOT / image_file, anns)
