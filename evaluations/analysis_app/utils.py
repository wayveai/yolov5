import streamlit as st
from hashlib import sha256
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import altair as alt
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

def add_tablet(s: str):
    tablet = ['🔴', '🟠', '🟡', '🟢', '🔵', '🟣', '🟤', '⚫', '🟥', '🟦', '🟧', '🟨', '🟩', '🟪', '🟫', '⬛']
    selected_tablet = tablet[int(sha256(s.encode('utf-8')).hexdigest(), 16) % len(tablet)]
    return selected_tablet + '  ' + s


def display_classification_report(report: str):
    first_indent = len(report.split('\n')[2].split()[0])
    st.write(f'```bash\n-{report}```')


def display_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true', labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm * 100, display_labels=labels)

    fig, ax = plt.subplots(dpi=120, figsize=(5, 5))
    disp.plot(xticks_rotation='vertical', values_format='.1f', ax=ax, colorbar=False)
    st.pyplot(fig)




def _draw_bbox_with_text(draw, xy, text, size):
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", size)
    bbox = draw.textbbox(xy, text, font=fnt, anchor='ls')
    draw.rectangle(bbox, fill="red")
    draw.text(xy, text, font=fnt, fill="black", anchor='ls')
    draw.rectangle(xy, outline='red', width=2)
        
def draw_annotations(image, annotations, size: int = 30):
    if is_float_dtype(annotations['x0']):
        width, height = image.size
        annotations[['x0', 'y0', 'x1', 'y1']] = annotations[['x0', 'y0', 'x1', 'y1']].astype(float)
        annotations[['x0', 'x1']] *= width
        annotations[['y0', 'y1']] *= height
        annotations[['x0', 'y0', 'x1', 'y1']] = annotations[['x0', 'y0', 'x1', 'y1']].astype(int)
    
    draw = ImageDraw.Draw(image)
    for _, (label, conf, *xy) in annotations[['name', 'confidence', 'x0', 'y0', 'x1', 'y1']].iterrows():
        handle = "".join([x[0] for x in label.split("_")])
        text = f'{handle}:{conf*100:.0f}'
        _draw_bbox_with_text(draw, xy, text, size)


def display_2d_predictions(file, annotations):
    remaining_cols = {'x0', 'y0', 'x1', 'y1', 'confidence', 'class', 'name'} - set(annotations.columns)
    assert len(remaining_cols) == 0, remaining_cols
    image = Image.open(file).resize((1920 // 2, 1200 // 2), Image.ANTIALIAS)
    draw_annotations(image, annotations.copy(True), size=25)
    st.image(image)


@st.cache(allow_output_mutation=True)
def _cached_barchart(classification_df):
    chart = alt.Chart(classification_df)
    return chart.mark_bar(opacity=0.7).encode(
        x=alt.X('confidence', bin=alt.BinParams(maxbins=80)),
        y=alt.Y('count()', stack=None),
        color='is_true'
    ).properties(width=550)


def display_histograms(classification_df: pd.DataFrame):
    barchart = _cached_barchart(classification_df)
    st.altair_chart(barchart.facet(facet='gt', columns=3).resolve_scale(y='independent').properties(title='Facet by Ground Truth label'))
    st.altair_chart(barchart.facet(facet='pred', columns=3).resolve_scale(y='independent').properties(title='Facet by Predicted label'))



