"""This Streamlit app only makes predictions. It never trains on the data uploaded.

To launch it, run:  streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from evaluation import evaluate

ROOT = Path(__file__).resolve().parent
METADATA = json.loads((ROOT / 'model' / 'metadata.json').read_text(encoding='utf-8'))


@st.cache_resource
def load_models():
    return {name: joblib.load(ROOT / 'model' / filename) for name, filename in METADATA['models'].items()}


def read_upload(uploaded):
    """Validate the uploaded CSV and return (df, X, y). Raises ValueError on bad input."""
    df = pd.read_csv(uploaded)
    if df.empty:
        raise ValueError('CSV contains no test rows.')
    features, target = METADATA['features'], METADATA['target']
    required = features + [target]
    missing = [c for c in required if c not in df.columns]
    extra = [c for c in df.columns if c not in required]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    if extra:
        raise ValueError('Unexpected columns: ' + ', '.join(extra))
    try:
        X = df.loc[:, features].apply(pd.to_numeric, errors='raise').astype(float)
        y = pd.to_numeric(df[target], errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError(f'All feature and target values must be numeric ({exc}).') from exc
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError('Missing or infinite values are not supported.')
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError('Target values must be 0 (benign) or 1 (malignant).')
    return df, X, y.astype(int)


def main():
    st.set_page_config(page_title='CellScope breast cancer classifier | Model Comparison', page_icon='🔬', layout='wide')
    st.title('CellScope breast cancer: five perspectives on classification')
    st.caption('Breast Cancer Wisconsin Diagnostic_Educational demonstration')
    st.info('Positive class: malignant (1). Negative class: benign (0). This application is not a clinical diagnostic tool.')

    models = load_models()
    with st.sidebar:
        st.header('Evaluate a saved model')
        selected = st.selectbox('Classifier', list(models))
        uploaded = st.file_uploader('Upload test CSV with all 30 features and target', type=['csv'])
        st.download_button('Download supplied test data', (ROOT / 'test_data.csv').read_bytes(), 'test_data.csv', 'text/csv')
        st.caption('Models were trained before deployment. Uploaded data is used only for evaluation.')
        st.write('Training-CV selection:', METADATA['selected_model'])

    if uploaded is None:
        st.write('Upload a test CSV to evaluate all five classifiers. You can download the supplied held-out dataset from the sidebar.')
        return
    try:
        df, X, y = read_upload(uploaded)
    except Exception as exc:  # show any parsing/validation problem
        st.error(f'Invalid CSV: {exc}')
        return

    if y.nunique() < 2:
        st.warning('AUC is undefined for a one-class upload; it is displayed as N/A.')
    st.caption(f'{len(df)} uploaded test rows · {len(METADATA["features"])} input features · '
               'zero-division precision/recall/F1 convention: 0')

    rows, selected_metrics, selected_predictions = [], None, None
    for name, model in models.items():
        measures, predictions = evaluate(model, X, y)
        rows.append({'Model': name, **measures})
        if name == selected:
            selected_metrics, selected_predictions = measures, predictions

    st.subheader(selected)
    for col, (metric, value) in zip(st.columns(6), selected_metrics.items()):
        col.metric(metric, 'N/A' if value is None else f'{value:.4f}')

    st.subheader('All classifiers on the same uploaded data')
    comparison = pd.DataFrame(rows).set_index('Model')
    st.dataframe(comparison.style.format('{:.4f}', na_rep='N/A'), width='stretch')
    st.download_button('Download comparison CSV', comparison.to_csv().encode(), 'uploaded_data_metrics.csv', 'text/csv')

    left, right = st.columns(2)
    with left:
        st.subheader('Confusion matrix')
        cm = confusion_matrix(y, selected_predictions, labels=[0, 1])
        st.dataframe(pd.DataFrame(cm, index=['Actual benign', 'Actual malignant'],
                                  columns=['Predicted benign', 'Predicted malignant']), width='stretch')
        st.caption('Rows: actual labels. Columns: predicted labels.')
    with right:
        st.subheader('Classification report')
        report = classification_report(y, selected_predictions, labels=[0, 1], target_names=['Benign', 'Malignant'],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), width='stretch')
    with st.expander('Uploaded test-data preview'):
        st.dataframe(df.head(10), width='stretch')
    st.caption('Results describe the uploaded data only. Different data can produce different rankings.')


main()
