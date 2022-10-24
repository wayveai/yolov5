import streamlit as st
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from contants import RESULTS_ROOT, POSSIBLE_LABELS, GROUND_TRUTH_ROOT
from utils import add_tablet, display_classification_report, display_confusion_matrix, display_2d_predictions
from loaders import load_entron_v2, load_predictions, get_classification_df

# cache classification_report to make it faster
classification_report = st.cache(classification_report)


@st.cache
def load_all_data(file: Path):
    """Handy function to cache large dataframes"""
    metadata, predictions_df = load_predictions(file)
    predictions_df['source'] = file.name
    classification_df = get_classification_df(predictions_df, ground_truth, metadata)
    return metadata, predictions_df, classification_df


st.set_page_config(page_title='Yolo TL CLS', layout='wide', page_icon='🔬')

st.markdown("# Compare different models predictions")
st.sidebar.markdown("### Select predictions files")

available_prediction_files = sorted(list(RESULTS_ROOT.glob('**/*.json')))
selected_prediction_files = st.sidebar.multiselect('Select model predictions', available_prediction_files, format_func=lambda x: add_tablet(x.name))

if len(selected_prediction_files) == 0:
    st.sidebar.info('Choose a file to load.')
    st.stop()

ground_truth = load_entron_v2(GROUND_TRUTH_ROOT)

all_classification_df = []
all_predictions_df = []
for file in selected_prediction_files:
    st.markdown('##### File:  ' + file.name)
    metadata, predictions_df, classification_df = load_all_data(file)
    all_classification_df.append(classification_df)
    all_predictions_df.append(predictions_df)

    cls_report = classification_report(y_true=classification_df['gt'], y_pred=classification_df['pred'], digits=3)
    col1, col2, _ = st.columns([3, 2, 2])
    with col1:
        display_classification_report(cls_report)
    with col2:
        display_confusion_matrix(y_true=classification_df['gt'], y_pred=classification_df['pred'], labels=POSSIBLE_LABELS)

all_classification_df = pd.concat(all_classification_df).reset_index(drop=True)
all_predictions_df = pd.concat(all_predictions_df).reset_index(drop=True)

st.header('Inspect object detections')

col1, col2, _ = st.columns([2, 2, 3])
any_label = 'Any label'
selected_label = col1.selectbox('Choose GT label', [any_label] + POSSIBLE_LABELS)
selected_pred_label = col2.selectbox('Choose Pred label', [any_label] + POSSIBLE_LABELS)

query_string = []
if selected_label != any_label:
    query_string.append(f'gt == "{selected_label}" ')
if selected_pred_label != any_label:
    query_string.append(f'pred == "{selected_pred_label}"')
query_string = ' and '.join(query_string)

images = classification_df.query(query_string)['file'].unique() if query_string else classification_df['file'].unique()
images = sorted(images)

st.info(f'Found {len(images)} images for this filter')

idx = st.number_input("Select image", 0, len(images), 0)
image_file = images[idx]

for source in [f.name for f in selected_prediction_files]:
    annotations = predictions_df.query(f'file == "{image_file}" and source == "{source}"')
    st.write(source)
    if annotations.empty:
        st.warning('No traffic lights detected')
    else:
        st.dataframe(annotations, width=1500)

    display_2d_predictions(GROUND_TRUTH_ROOT / image_file, annotations)