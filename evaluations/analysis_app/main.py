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
available_predictions = sorted(list(RESULTS_ROOT.glob('*.json')) + list(RESULTS_ROOT.glob('*/predictions.json')))
display_fn_of_pred_files = lambda x: x if x == no_selection else add_tablet(x.parent.name if x.name == 'predictions.json' else x.name)
prediction_file = st.selectbox('Select model predictions', [no_selection] + available_predictions, format_func=display_fn_of_pred_files)

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
any_wrong_label = 'Wrong labels'
selected_label = col1.selectbox('Choose GT label', [any_label, any_wrong_label] + POSSIBLE_LABELS)
selected_pred_label = col2.selectbox('Choose Pred label', [any_label, any_wrong_label] + POSSIBLE_LABELS)

if selected_label == any_label and selected_pred_label == any_label:
    query_string = ''
elif selected_label in [any_label, any_wrong_label] and selected_pred_label in [any_label, any_wrong_label]:
    query_string = 'gt != pred'
elif selected_label not in [any_label, any_wrong_label] and selected_pred_label in [any_label, any_wrong_label]:
    query_string = f'gt != pred and gt == "{selected_label}"'
elif selected_label in [any_label, any_wrong_label] and selected_pred_label not in [any_label, any_wrong_label]:
    query_string = f'gt != pred and pred == "{selected_pred_label}"'
elif selected_label not in [any_label, any_wrong_label] and selected_pred_label not in [any_label, any_wrong_label]:
    query_string = f'gt == "{selected_label}" and pred == "{selected_pred_label}"'

col3.text('Filtering query')
col3.code(query_string, language='shell')

subset_df = classification_df.query(query_string).copy(True) if query_string else classification_df.copy(True)
subset_df = subset_df.sort_values(['run_id', 'ts'])
images = subset_df['file'].unique()

st.info(f'Found {len(images)} images for this filter')

idx = st.number_input("Select image", 0, len(images), 0)
image_file = images[idx]
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
