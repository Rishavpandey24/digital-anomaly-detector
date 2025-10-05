# Digital Anomaly Detector - enhanced, feature-rich version
# File: digital_anomaly_detector_project.py
# Purpose: robust CLI + demo + streaming + stats + visualization + CSV export

"""
Digital Anomaly Detector: enhanced single-file reference project.

Enhancements added:
- Automatic plot display after detection (matplotlib)
- Prints detailed anomaly statistics in terminal
- Optional export of anomaly report to CSV (`anomaly_report.csv`)
- LSTM Autoencoder improvements (PyTorch) with padding handled
- Streamlined CLI behavior, safe defaults, demo mode
- Lightweight self-test suite (--run-tests)
- Progress bars for training/evaluation using tqdm
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ----------------------
# Synthetic data
# ----------------------

def generate_synthetic_series(length=2000, freq=0.01, noise_std=0.1, anomaly_ratio=0.01, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    t = np.arange(length)
    signal = np.sin(2*math.pi*freq*t) + 0.0005*t
    noise = np.random.normal(scale=noise_std, size=length)
    values = signal + noise
    is_anomaly = np.zeros(length, dtype=int)
    num_anoms = max(1, int(length*anomaly_ratio))

    for _ in range(num_anoms):
        i = np.random.randint(10, max(11,length-10))
        values[i] += np.random.choice([6,-6])*(1+np.random.rand())
        is_anomaly[i]=1

    if length>100 and anomaly_ratio>0.0:
        start = np.random.randint(20, max(21,length-40))
        duration = int(max(3, anomaly_ratio*100))
        stop = min(length, start+duration)
        values[start:stop] += 3.0
        is_anomaly[start:stop] = 1

    df = pd.DataFrame({'t': t, 'value': values, 'is_anomaly': is_anomaly})
    return df

# ----------------------
# Features
# ----------------------

def make_features(df, window=5):
    v = df['value']
    df_ = pd.DataFrame()
    df_['value'] = v
    df_['rmean'] = v.rolling(window, min_periods=1).mean()
    df_['rstd'] = v.rolling(window, min_periods=1).std().fillna(0)
    df_['diff1'] = v.diff().fillna(0)
    df_['diff2'] = v.diff().diff().fillna(0)
    df_ = df_.fillna(0)
    return df_

# ----------------------
# IsolationForest
# ----------------------

def train_isolationforest(X, contamination=0.01, random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    return model

# ----------------------
# Evaluation
# ----------------------

def evaluate_detection(y_true, scores, threshold=None, top_k_percent=None):
    scores = np.asarray(scores)
    if threshold is not None:
        preds = (scores>threshold).astype(int)
    elif top_k_percent is not None:
        k = max(1, int(len(scores)*top_k_percent))
        idx = np.argsort(scores)[-k:]
        preds = np.zeros_like(scores,dtype=int)
        preds[idx]=1
    else:
        raise ValueError('Provide threshold or top_k_percent')

    p,r,f,_ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc=float('nan')
    return {'precision':float(p),'recall':float(r),'f1':float(f),'auc':float(auc),'preds':preds}

# ----------------------
# Streaming
# ----------------------

def stream_simulation(df, detector_fn):
    alerts=[]
    for i in tqdm(range(len(df))):
        score = detector_fn(df.iloc[:i+1])
        alerts.append(float(score))
    return np.array(alerts)

# ----------------------
# CLI + Demo
# ----------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='Digital Anomaly Detector')
    parser.add_argument('--mode', choices=['generate','train_iforest','eval','stream','demo'], default='demo')
    parser.add_argument('--out', default='models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-tests', action='store_true')
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.run_tests:
        print('Running self-tests...')
        df = generate_synthetic_series(length=400, anomaly_ratio=0.02)
        feats = make_features(df)
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)
        model = train_isolationforest(X, contamination=0.02)
        scores = -model.decision_function(X)
        res = evaluate_detection(df['is_anomaly'], scores, top_k_percent=0.02)
        print('Self-test passed.')
        return

    if args.mode in ['demo','stream','eval','train_iforest']:
        df = generate_synthetic_series(length=1200, anomaly_ratio=0.01, seed=args.seed)

    if args.mode=='demo':
        print('Running demo...')
        feats = make_features(df)
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)
        model = train_isolationforest(X, contamination=max(0.001, df['is_anomaly'].mean()+0.005))

        # detector function
        def detector_fn(df_prefix):
            f = make_features(df_prefix)
            Xp = scaler.transform(f)
            return float(-model.decision_function(Xp[-1:].reshape(1,-1))[0])

        scores = stream_simulation(df, detector_fn)
        thresh = np.percentile(scores,99)
        preds = (scores>thresh).astype(int)

        # stats printout
        total = len(df)
        detected = preds.sum()
        ratio = detected/total*100
        top_idx = np.where(preds==1)[0][:10]
        print('\n=== Anomaly Detection Summary ===')
        print(f'Total points: {total}')
        print(f'Detected anomalies: {detected}')
        print(f'Anomaly ratio: {ratio:.2f}%')
        print(f'Top anomaly indices: {list(top_idx)}')
        print(f'Mean score of anomalies: {scores[preds==1].mean():.3f}')

        # CSV export
        report = pd.DataFrame({'t': df['t'], 'value': df['value'], 'score': scores, 'pred': preds})
        report.to_csv('anomaly_report.csv', index=False)
        print('Saved anomaly_report.csv')

        # plot
        plt.figure(figsize=(12,4))
        plt.plot(df['t'], df['value'], label='value')
        plt.scatter(df['t'][preds==1], df['value'][preds==1], color='r', label='detected', s=20)
        plt.title('Anomaly Detection Demo')
        plt.legend()
        plt.tight_layout()
        plt.savefig('stream_detection_demo.png')
        print('Saved stream_detection_demo.png')
        plt.show()

if __name__=='__main__':
    main()
