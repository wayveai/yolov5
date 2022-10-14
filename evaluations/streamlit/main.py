import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,  confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


RESULTS_ROOT = Path('./evaluations/entron_v2_classification_preds')
GROUND_TRUTH_ROOT = Path('../yolo_dataset/traffic_lights_entron_classification/focal_len=650__sensor_size_hw=1200x1920').resolve()

COLOUR_SIMPLIFICATION_MAP = {
    'GREEN_ARROW_LEFT': 'GREEN_SOLID',
    'GREEN_ARROW_RIGHT': 'GREEN_SOLID',
    'GREEN_ARROW_STRAIGHT': 'GREEN_SOLID',
    'UNKNOWN': 'NONE',
    'RED_ARROW_RIGHT': 'RED_SOLID',
    'RED_ARROW_STRAIGHT': 'RED_SOLID',
    'RED_ARROW_LEFT': 'RED_SOLID',
}

POSSIBLE_LABELS = ['NONE', 'GREEN_SOLID', 'RED_SOLID', 'AMBER_SOLID', 'RED_AND_AMBER']

st.code

st.set_page_config(layout='wide', page_icon='🔬')

@st.cache()
def load_predictions(file: Path) -> dict:
    with file.open('r') as f:
        payload =  json.load(f)

    img_root = Path(payload['metadata']['image_root'])
    for k in list(payload['predictions']):
        new_k = str(Path(k).relative_to(img_root))
        payload['predictions'][new_k] = payload['predictions'][k]
        del payload['predictions'][k]
    return payload

def predictions_to_dataframe(preds: dict) -> pd.DataFrame:
    """
    Get the dictionary of predictions and generate a dataframe with predicted and ground truth values
    Ground truth values are coming from the name of the parent directory
    The datast is assumed to be Traffic Light Entron V2
    """
    df = pd.DataFrame.from_dict(preds, orient='index')
    df = df.reset_index()
    df = df.rename(columns={'index': 'file', 'colour': 'pred'})
    # specifically remove UNKNWON predictions
    df.loc[df['pred'] == 'UNKNWON', 'pred'] = 'NONE'
    df['gt'] = df['file'].str.split('/', 1).str.get(0)
    # the entron dataset v2 has a known miss label with UNKNOWN example
    df = df[df['gt'] != 'UNKNOWN'].reset_index(drop=True)
    # simplify Arrows to solid colours
    df['gt'] = df['gt'].map(lambda x: COLOUR_SIMPLIFICATION_MAP.get(x, x))

    assert df['gt'].isin(POSSIBLE_LABELS).all(), str(df['gt'][~df['gt'].isin(POSSIBLE_LABELS)])
    assert df['pred'].isin(POSSIBLE_LABELS).all()

    df['is_true'] = df['pred'] == df['gt']
    return df

def display_classification_report(report: str):
    first_indent = len(report.split('\n')[2].split()[0])
    st.write(f'```bash\n-{report}```')


def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true', labels=POSSIBLE_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm * 100, display_labels=POSSIBLE_LABELS)

    fig, ax = plt.subplots(dpi=120, figsize=(5, 5))
    disp.plot(xticks_rotation='vertical', values_format='.1f', ax=ax, colorbar=False)
    st.pyplot(fig)

def add_tablet(s: str):
    tablet = ['🔴', '🟠', '🟡', '🟢', '🔵', '🟣', '🟤', '⚫', '🟥', '🟦', '🟧', '🟨', '🟩', '🟪', '🟫', '⬛']
    selected_tablet = tablet[hash(s) % len(tablet)]
    return selected_tablet + '  ' + s

st.title('Analysis tool')
st.write('Ground truth directory:', GROUND_TRUTH_ROOT, '| exists:', GROUND_TRUTH_ROOT.exists())




prediction_file = st.selectbox('Select model predictions', sorted(list(RESULTS_ROOT.glob('*.json'))), format_func=lambda x: add_tablet(x.name))

predictions = load_predictions(prediction_file)
predictions_df = predictions_to_dataframe(predictions['predictions'])

with st.expander('Metadata:'):
    st.json(predictions['metadata'])



st.header('Classification report')
st.markdown('##### ' + prediction_file.name)
cls_report = classification_report(y_true=predictions_df['gt'], y_pred=predictions_df['pred'], digits=3)
col1, col2, _ = st.columns([3, 2, 2])
with col1:
    display_classification_report(cls_report)
with col2:
    display_confusion_matrix(y_true=predictions_df['gt'], y_pred=predictions_df['pred'])


st.header('Histograms')


chart = alt.Chart(predictions_df).mark_bar(opacity=0.7).encode(
    x=alt.X('conf', bin=alt.BinParams(maxbins=80)),
    y=alt.Y('count()', stack=None),
    color='is_true'
).properties(width=550).facet(facet='gt', columns=2).resolve_scale(y='independent').properties(title='Facet by Ground Truth label')
st.altair_chart(chart)

chart = alt.Chart(predictions_df).mark_bar(opacity=0.7).encode(
    x=alt.X('conf', bin=alt.BinParams(maxbins=80)),
    y=alt.Y('count()', stack=None),
    color='is_true'
).properties(width=550).facet(facet='pred', columns=2).resolve_scale(y='independent').properties(title='Facet by Predicted label')
st.altair_chart(chart)

st.header('Inspect wrong predictions')

col1, col2, col3, _ = st.columns([2, 2, 1, 2])
any_label = 'Any label'
selected_label = col1.selectbox('Choose GT label', [any_label] + POSSIBLE_LABELS)
selected_pred_label = col2.selectbox('Choose Pred label', [any_label] + POSSIBLE_LABELS)
num_images = col3.number_input('Number of images to display', min_value=1, max_value=20)

query_string = []
if selected_label != any_label:
    query_string.append(f'gt == "{selected_label}"')
if selected_pred_label != any_label:
    query_string.append(f' and pred == "{selected_pred_label}"')

if len(query_string) == 0:
    sub_df = predictions_df
else:
    sub_df = predictions_df.query(' and '.join(query_string))

st.write(len(sub_df))
