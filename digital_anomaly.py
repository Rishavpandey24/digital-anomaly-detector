# Universal Digital Anomaly Detector with Incremental Learning
# Fixed CLI: --source is optional (avoids SystemExit: 2). Auto-detects files and falls
# back to a demo synthetic dataset. Adds self-tests and safer OCR handling.

"""
This script supports CSV/XLSX files, image files (OCR numeric extraction), and
streaming (simulated) input. It implements an incremental-like update strategy for
IsolationForest by retraining on a rolling buffer of recent data batches.

How to run:
  - Analyze a file: python digital_anomaly_detector_project.py --source data.csv
  - Run demo (no args): python digital_anomaly_detector_project.py
  - Stream mode: python digital_anomaly_detector_project.py --source data.csv --stream
  - Run internal tests: python digital_anomaly_detector_project.py --run-tests

Notes:
- The --source argument is now optional. If omitted the script searches for
  common filenames (data.csv, dataset.csv, input.csv, timeseries.csv, values.csv)
  and uses the first match. If none found, it runs a synthetic demo dataset.
- Image OCR requires pytesseract and Pillow. If these are not available the
  script will report a friendly error and skip image support.
"""

import os
import argparse
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend by default so headless environments won't crash when saving plots.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional libraries for OCR/image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# IsolationForest and utilities
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# Configuration & filenames
# -----------------------------
COMMON_FILENAMES = ['data.csv', 'dataset.csv', 'input.csv', 'timeseries.csv', 'values.csv']
MODEL_DIR = 'models'
OUT_DIR_DEFAULT = 'outputs'
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Helpers: find candidate file
# -----------------------------

def find_candidate_file(provided_path=None):
    """Return a usable file path or None.

    If provided_path is set and exists, return it. Otherwise search common filenames
    in cwd and return the first match. If none found, return None.
    """
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return str(p)
        # try relative/absolute
        if Path(provided_path).is_absolute() and Path(provided_path).exists():
            return str(Path(provided_path).resolve())
        print(f"Provided file path not found: {provided_path}")
        return None

    cwd = Path.cwd()
    for fn in COMMON_FILENAMES:
        p = cwd / fn
        if p.exists():
            print(f"Auto-detected dataset file: {p}")
            return str(p)
    return None

# -----------------------------
# Data loaders
# -----------------------------

def load_csv_or_xlsx(path):
    ext = Path(path).suffix.lower()
    try:
        if ext == '.csv':
            return pd.read_csv(path)
        elif ext in ('.xls', '.xlsx'):
            return pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read tabular file {path}: {e}")
    raise ValueError(f"Unsupported spreadsheet extension: {ext}")


def image_to_dataframe(path):
    """Extract numeric tokens from an image using OCR and return as single-column DataFrame.

    The function tries Pillow + pytesseract. It supports numbers with signs and decimals.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError('Pillow (PIL) not available. Install with `pip install pillow` to use image OCR.')
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError('pytesseract not available. Install with `pip install pytesseract` and ensure Tesseract is installed in your system.')

    try:
        img = Image.open(path).convert('L')
        text = pytesseract.image_to_string(img)
    except Exception as e:
        raise RuntimeError(f'OCR failed on image {path}: {e}')

    # Find floats/ints, allow negative numbers and decimals
    tokens = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except Exception:
            continue
    if len(nums) == 0:
        raise ValueError('No numeric tokens found in OCR result.')
    df = pd.DataFrame({'value': nums})
    print(f'Extracted {len(nums)} numeric values from image.')
    return df


def load_data_universal(source):
    """Load a dataset from various sources: CSV/XLSX/image or accept a DataFrame.

    If source is None, return None (caller will decide fallback behavior).
    """
    if source is None:
        return None
    if isinstance(source, pd.DataFrame):
        return source
    path = str(source)
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')

    ext = Path(path).suffix.lower()
    if ext in ('.csv', '.xls', '.xlsx'):
        return load_csv_or_xlsx(path)
    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
        return image_to_dataframe(path)
    raise ValueError(f'Unsupported file type: {ext}')

# -----------------------------
# Preprocessing & features
# -----------------------------

def preprocess_numeric(df, use_all_numeric=True, column=None):
    """Return (X, df_num, scaler).

    If use_all_numeric True, selects all numeric columns. If column specified, uses only that column.
    """
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset columns: {list(df.columns)}")
        df_num = df[[column]].copy()
    else:
        df_num = df.select_dtypes(include=[np.number]).copy()
        if df_num.shape[1] == 0:
            raise ValueError('No numeric columns found in dataset')

    # basic cleaning
    df_num = df_num.ffill().bfill()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_num.values)
    return X, df_num, scaler

# -----------------------------
# Incremental-like detection strategy
# -----------------------------

def _model_path(outdir, version=None):
    os.makedirs(outdir, exist_ok=True)
    if version is None:
        return os.path.join(outdir, 'iforest_model.pkl')
    return os.path.join(outdir, f'iforest_model_v{version}.pkl')


def save_model(model, outdir, version=None):
    path = _model_path(outdir, version)
    joblib.dump(model, path)
    print(f'Saved model to {path}')


def load_model(outdir):
    path = _model_path(outdir)
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            print(f'Loaded model from {path}')
            return model
        except Exception as e:
            print(f'Failed to load model from {path}: {e}')
    return None


def detect_anomalies_full(X, contamination=0.01, random_state=42):
    """Train a fresh IsolationForest and return results dict (scores,preds,model,threshold)."""
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    scores = -model.decision_function(X)
    # compute threshold using percentile consistent with contamination
    if contamination <= 0 or contamination >= 1:
        thresh = np.percentile(scores, 99)
    else:
        pct = 100.0 - contamination * 100.0
        pct = min(max(pct, 0.0), 100.0)
        thresh = np.percentile(scores, pct)
    preds = (scores > thresh).astype(int)
    return {'scores': scores, 'preds': preds, 'threshold': float(thresh), 'model': model}


def incremental_stream_process(df, outdir, contamination=0.01, chunk_size=200, keep_buffer=1000):
    """Simulate an incremental processing pipeline.

    Strategy: maintain a rolling buffer of recent rows (keep_buffer). For each incoming
    chunk, append to buffer, train a new IsolationForest on the buffer, detect anomalies
    in the chunk, and optionally save the model. This is not true partial_fit but a
    pragmatic incremental approach.
    """
    buffer_df = pd.DataFrame(columns=df.columns)
    results_rows = []
    model = load_model(outdir)
    version = 0

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start+chunk_size]
        buffer_df = pd.concat([buffer_df, chunk]).iloc[-keep_buffer:]
        try:
            Xbuf, dfbuf, _ = preprocess_numeric(buffer_df)
        except Exception as e:
            print('Preprocess failed for buffer:', e)
            continue
        # train on buffer
        res = detect_anomalies_full(Xbuf, contamination=contamination)
        model = res['model']
        # detect on current chunk (align indices)
        Xchunk, dfchunk, _ = preprocess_numeric(chunk)
        scores_chunk = -model.decision_function(Xchunk)
        # threshold from buffer
        thresh = res['threshold']
        preds_chunk = (scores_chunk > thresh).astype(int)
        # store
        chunk_out = chunk.copy()
        # flatten if multiple cols into first col in outputs for plotting convenience
        chunk_out['anomaly_score'] = scores_chunk
        chunk_out['is_anomaly'] = preds_chunk
        results_rows.append(chunk_out)
        version += 1
        save_model(model, outdir, version=None)  # overwrite latest
        # log progress
        print(f'Processed chunk {start}..{start+len(chunk)-1}: detected {int(preds_chunk.sum())} anomalies; model version saved')

    # concat results
    result_df = pd.concat(results_rows) if len(results_rows) > 0 else pd.DataFrame()
    return result_df, model

# -----------------------------
# Output & plotting
# -----------------------------

def save_report(df_out, outdir):
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, 'anomaly_report.csv')
    df_out.to_csv(out_csv, index=False)
    print(f'Saved anomaly report to {out_csv}')
    return out_csv


def plot_and_save(df_out, outdir, show=False):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, 'anomaly_plot.png')
    plt.figure(figsize=(12,5))
    # Pick first numeric column for plotting convenience
    if df_out.shape[1] == 0:
        print('No data to plot')
        return png
    # find first numeric column (not including anomaly cols)
    non_plot_cols = {'anomaly_score', 'is_anomaly'}
    plot_col = None
    for c in df_out.columns:
        if c not in non_plot_cols and np.issubdtype(df_out[c].dtype, np.number):
            plot_col = c
            break
    if plot_col is None:
        # fallback to first column
        plot_col = df_out.columns[0]

    plt.plot(df_out[plot_col].values, label=str(plot_col))
    if 'is_anomaly' in df_out.columns:
        idxs = np.where(df_out['is_anomaly'].values == 1)[0]
        if len(idxs) > 0:
            plt.scatter(idxs, df_out[plot_col].values[idxs], color='red', label='anomaly')
    plt.legend()
    plt.title('Anomaly Detection')
    plt.tight_layout()
    plt.savefig(png)
    plt.close()
    print(f'Saved plot to {png}')
    if show:
        try:
            # try to open with PIL (this may or may not work depending on env)
            if PIL_AVAILABLE:
                im = Image.open(png)
                im.show()
                print('Displayed plot using PIL.Image.show()')
            else:
                print('PIL not available; cannot display plot interactively in this environment.')
        except Exception as e:
            print('Unable to display plot interactively:', e)
    return png

# -----------------------------
# Self-tests
# -----------------------------

def _make_synthetic(length=500, anomaly_ratio=0.02, seed=0):
    np.random.seed(seed)
    t = np.arange(length)
    vals = np.sin(2 * math.pi * 0.01 * t) + 0.001 * t + np.random.normal(scale=0.1, size=length)
    n_anom = max(1, int(length * anomaly_ratio))
    idx = np.random.choice(np.arange(10, length-10), size=n_anom, replace=False)
    vals[idx] += np.random.choice([4.0, -4.0], size=n_anom)
    df = pd.DataFrame({'value': vals})
    return df, idx


def run_self_tests():
    print('
Running self-tests...')
    # Test 1: full batch
    df, injected = _make_synthetic(length=200, anomaly_ratio=0.03, seed=1)
    X, df_num, scaler = preprocess_numeric(df)
    res = detect_anomalies_full(X, contamination=0.03)
    assert 'scores' in res and 'preds' in res
    assert res['preds'].shape[0] == len(df)

    # Test 2: streaming
    df2, _ = _make_synthetic(length=500, anomaly_ratio=0.02, seed=2)
    outdir = 'test_outputs'
    res_df, model = incremental_stream_process(df2, outdir, contamination=0.02, chunk_size=100, keep_buffer=300)
    assert not res_df.empty
    assert 'is_anomaly' in res_df.columns

    # Test 3: image OCR not enforced in tests due to environment variability
    print('All self-tests passed.')

# -----------------------------
# CLI / Main
# -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='Universal Digital Anomaly Detector (robust CLI)')
    parser.add_argument('--source', '-s', help='Path to CSV/XLSX/image file (optional). If omitted, auto-detects or runs demo.')
    parser.add_argument('--outdir', '-o', default=OUT_DIR_DEFAULT, help='Directory to write outputs')
    parser.add_argument('--contamination', '-c', type=float, default=0.01, help='Expected anomaly fraction (0-1)')
    parser.add_argument('--stream', action='store_true', help='Enable streaming (incremental) processing')
    parser.add_argument('--column', help='If set, analyze only this single column (otherwise all numeric columns used)')
    parser.add_argument('--show', action='store_true', help='Attempt to display the saved plot (may not work in headless env)')
    parser.add_argument('--run-tests', action='store_true', help='Run internal tests and exit')

    args = parser.parse_args(argv)

    if args.run_tests:
        run_self_tests()
        return

    file_path = find_candidate_file(args.source)
    if file_path is None:
        print('No file provided or auto-detected. Running demo synthetic dataset.')
        df, _ = _make_synthetic(length=1000, anomaly_ratio=max(0.001, args.contamination), seed=42)
        # use full-batch detection for demo
        X, df_num, scaler = preprocess_numeric(df)
        res = detect_anomalies_full(X, contamination=args.contamination)
        df_out = df.copy()
        df_out['anomaly_score'] = res['scores']
        df_out['is_anomaly'] = res['preds']
        save_report(df_out, args.outdir)
        plot_and_save(df_out, args.outdir, show=args.show)
        return

    # load dataset
    try:
        df_loaded = load_data_universal(file_path)
    except Exception as e:
        print('Failed to load provided file:', e)
        traceback.print_exc()
        print('Falling back to demo synthetic dataset.')
        df_loaded, _ = _make_synthetic(length=1000, anomaly_ratio=max(0.001, args.contamination), seed=42)

    # if streaming, perform incremental processing
    if args.stream:
        print('Starting incremental/streaming processing...')
        result_df, model = incremental_stream_process(df_loaded, MODEL_DIR, contamination=args.contamination)
        if result_df.empty:
            print('Streaming produced no results (empty).')
            return
        save_report(result_df, args.outdir)
        plot_and_save(result_df, args.outdir, show=args.show)
        return

    # otherwise, full-batch detection
    try:
        X, df_num, scaler = preprocess_numeric(df_loaded, column=args.column)
    except Exception as e:
        print('Preprocessing failed:', e)
        traceback.print_exc()
        print('Falling back to demo synthetic dataset.')
        df_loaded, _ = _make_synthetic(length=1000, anomaly_ratio=max(0.001, args.contamination), seed=42)
        X, df_num, scaler = preprocess_numeric(df_loaded)

    res = detect_anomalies_full(X, contamination=args.contamination)
    df_out = df_num.copy()
    df_out['anomaly_score'] = res['scores']
    df_out['is_anomaly'] = res['preds']

    save_report(df_out, args.outdir)
    plot_and_save(df_out, args.outdir, show=args.show)


if __name__ == '__main__':
    main()
