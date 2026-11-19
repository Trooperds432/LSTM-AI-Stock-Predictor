"""Walk-forward retraining for the Conv1D+LSTM model (design: research/walkforward_2026-11-19_preregistration.md).

One job = one (test year, seed). Each job trains only on data whose labels end before its
validation window, which itself ends before the test year (126-trading-day purge/embargo).

Usage:
  python walk_forward.py job --year 2016 --seed-index 0 [--smoke]
  python walk_forward.py run --years 2016-2023 --seeds 3 --workers 4 --threads 4 [--smoke]
Outputs (resumable, skipped if present):
  wf/forecasts/Y{year}_s{k}_{val|test}.csv.gz  and  wf/meta/Y{year}_s{k}.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "TrainingData/indicators_data/processed/stocksData"
SPY_PATH = ROOT / "TrainingData/indicators_data/processed/SPY-VIX/SPY_daily_processed.csv"

HORIZONS = ["1d", "1w", "1m", "6m"]
HORIZON_DAYS = [1, 5, 21, 126]
WINDOW = 60
TRAIN_STEP = 21
PURGE = 126          # trading days = longest label horizon
VAL_LEN = 252
LEVEL_COLS = ["YesterdayClose", "MA10", "MA20", "MA30", "MA50", "MA200",
              "EMA10", "EMA30", "BollingerUpper", "BollingerLower"]
EXCLUDED = {"date", "close", "OBV"} | {f"Target_{h}" for h in HORIZONS}


def is_non_common(symbol: str) -> bool:
    return len(symbol) >= 5 and symbol[-1] in "WURP"


# ----------------------------------------------------------------------------- data

def load_universe(max_tickers: int | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(DATA_DIR.glob("*.csv")):
        if f.stat().st_size < 200:
            continue
        ticker = f.stem  # e.g. AAL_daily_processed
        if is_non_common(ticker.split("_")[0]):
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if df.empty or "close" not in df:
            continue
        for c in LEVEL_COLS:
            if c in df.columns:
                df[c] = df[c] / df["close"] - 1.0
        for h, n in zip(HORIZONS, HORIZON_DAYS):
            df[f"Target_{h}"] = np.log(df["close"].shift(-n) / df["close"])
        df = df.replace([np.inf, -np.inf], np.nan)
        feat = [c for c in df.columns if c not in EXCLUDED]
        df = df.dropna(subset=feat).reset_index(drop=True)  # early indicator warm-up rows
        if len(df) > WINDOW + 1:
            out[ticker] = df
        if max_tickers and len(out) >= max_tickers:
            break
    return out


def trading_calendar() -> pd.DatetimeIndex:
    spy = pd.read_csv(SPY_PATH, parse_dates=["date"])
    return pd.DatetimeIndex(spy["date"].sort_values().unique())


def fold_dates(year: int, cal: pd.DatetimeIndex) -> dict[str, pd.Timestamp]:
    y0 = cal[cal >= pd.Timestamp(f"{year}-01-01")][0]
    i = cal.get_loc(y0)
    v1 = cal[i - PURGE]
    v0 = cal[i - PURGE - VAL_LEN]
    train_end = cal[i - PURGE - VAL_LEN - PURGE]
    return {"test_start": y0, "test_end": pd.Timestamp(f"{year}-12-31"), "val_start": v0, "val_end": v1,
            "train_end": train_end}


# ----------------------------------------------------------------------------- model

def build_model(n_features: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    class MCDropout(layers.Dropout):
        def call(self, inputs, training=None):
            return super().call(inputs, training=True)

    model = tf.keras.Sequential([
        layers.Input(shape=(WINDOW + 1, n_features)),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        MCDropout(0.3),
        layers.LSTM(64, return_sequences=False),
        MCDropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(4, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def windows(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.stack([arr[i - WINDOW:i + 1] for i in idx]).astype(np.float32) if len(idx) else \
        np.empty((0, WINDOW + 1, arr.shape[1]), np.float32)


def run_job(year: int, k: int, smoke: bool, threads: int, out_dir: Path) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    seed = 1000 * year + k
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    t0 = time.time()
    data = load_universe(max_tickers=15 if smoke else None)
    fd = fold_dates(year, trading_calendar())
    feat_all = sorted(set().union(*[set(c for c in df.columns if c not in EXCLUDED) for df in data.values()]))

    # Per-ticker label thresholds from training rows only (mean + 1 std, as in the notebook)
    thresholds, train_rows = {}, []
    for t, df in data.items():
        tr = df[(df["date"] < fd["train_end"])]
        if len(tr) >= 100:
            thresholds[t] = {h: float(tr[f"Target_{h}"].mean() + tr[f"Target_{h}"].std()) for h in HORIZONS}
            train_rows.append(tr.reindex(columns=feat_all, fill_value=0.0).to_numpy(float))
    Xs = np.nan_to_num(np.vstack(train_rows), nan=0.0, posinf=1e6, neginf=-1e6)
    keep = Xs.std(axis=0) > 1e-12
    feat = [c for c, kp in zip(feat_all, keep) if kp]
    scaler = StandardScaler().fit(Xs[:, keep])

    def prep(df):
        f = df.reindex(columns=feat, fill_value=0.0).to_numpy(float)
        return np.nan_to_num(scaler.transform(np.nan_to_num(f, nan=0.0)), nan=0.0, posinf=10, neginf=-10)

    def labels(df, t):
        th = thresholds.get(t)
        y = np.full((len(df), 4), np.nan)
        if th is None:
            return y
        for j, h in enumerate(HORIZONS):
            tg = df[f"Target_{h}"].to_numpy(float)
            y[:, j] = np.where(np.isfinite(tg), (tg > th[h]).astype(float), np.nan)
        return y

    Xtr, ytr, Xv, yv = [], [], [], []
    cache = {}
    for t, df in data.items():
        F, Y = prep(df), labels(df, t)
        d = df["date"].to_numpy()
        cache[t] = (F, Y, d)
        base = np.arange(WINDOW, len(df), TRAIN_STEP)
        lab_ok = np.all(np.isfinite(Y[base]), axis=1)
        tr_idx = base[(d[base] < np.datetime64(fd["train_end"])) & lab_ok]
        v_idx = base[(d[base] >= np.datetime64(fd["val_start"])) & (d[base] < np.datetime64(fd["val_end"])) & lab_ok]
        if len(tr_idx):
            Xtr.append(windows(F, tr_idx)); ytr.append(Y[tr_idx])
        if len(v_idx):
            Xv.append(windows(F, v_idx)); yv.append(Y[v_idx])
    Xtr, ytr, Xv, yv = map(np.concatenate, (Xtr, ytr, Xv, yv))

    class EarlyStoppingWithWarmup(tf.keras.callbacks.EarlyStopping):
        def __init__(self, start_epoch=3, **kw):
            super().__init__(**kw)
            self.start_epoch = start_epoch

        def on_epoch_end(self, epoch, logs=None):
            if epoch >= self.start_epoch:
                super().on_epoch_end(epoch, logs)

    model = build_model(len(feat))
    es = EarlyStoppingWithWarmup(start_epoch=1 if smoke else 3, patience=8, monitor="val_auc", mode="max",
                                 restore_best_weights=True)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-5)
    hist = model.fit(Xtr, ytr, validation_data=(Xv, yv), batch_size=128, epochs=2 if smoke else 100,
                     callbacks=[es, rlr], shuffle=True, verbose=2)
    if getattr(es, "best_weights", None) is not None:
        model.set_weights(es.best_weights)
    va = hist.history["val_auc"]
    best_epoch = int(np.argmax(va[(1 if smoke else 3):]) + (1 if smoke else 3)) if len(va) > 3 else int(np.argmax(va))

    # MC-dropout forecasts, daily, for validation and test windows
    n_mc = 3 if smoke else 25
    for split, lo, hi in [("val", fd["val_start"], fd["val_end"]), ("test", fd["test_start"], fd["test_end"] + pd.Timedelta(days=1))]:
        rows = []
        for t, (F, Y, d) in cache.items():
            idx = np.arange(WINDOW, len(d))
            idx = idx[(d[idx] >= np.datetime64(lo)) & (d[idx] < np.datetime64(hi))]
            if not len(idx):
                continue
            X = windows(F, idx)
            preds = np.stack([model(X, training=True).numpy() for _ in range(n_mc)])
            mean, std = preds.mean(0), preds.std(0)
            df = data[t]
            rec = {"Date": d[idx], "Ticker": t, "Close": df["close"].to_numpy()[idx]}
            for j, h in enumerate(HORIZONS):
                rec[f"Pred_Prob_{h}"] = mean[:, j]
                rec[f"Pred_Prob_Std_{h}"] = std[:, j]
                rec[f"Class_{h}"] = Y[idx, j]
            rows.append(pd.DataFrame(rec))
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(out_dir / "forecasts" / f"Y{year}_s{k}_{split}.csv.gz", index=False)
        if split == "test":
            test_auc = {}
            for h in HORIZONS:
                m = out[f"Class_{h}"].notna()
                yy = out.loc[m, f"Class_{h}"]
                test_auc[h] = float(roc_auc_score(yy, out.loc[m, f"Pred_Prob_{h}"])) if yy.nunique() == 2 else None

    meta = {"year": year, "seed_index": k, "seed": seed, "smoke": smoke,
            "dates": {a: str(b.date()) for a, b in fd.items()},
            "n_train": int(len(Xtr)), "n_val": int(len(Xv)), "train_pos_rate": ytr.mean(0).round(4).tolist(),
            "n_features": len(feat), "dropped_features": [c for c, kp in zip(feat_all, keep) if not kp],
            "tickers_trained": len(thresholds), "tickers_total": len(data),
            "epochs_run": len(va), "best_epoch": best_epoch, "best_val_auc": float(max(va[(1 if smoke else 3):] or va)),
            "test_auc": test_auc, "minutes": round((time.time() - t0) / 60, 2),
            "versions": {"tensorflow": tf.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    (out_dir / "meta" / f"Y{year}_s{k}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k2: meta[k2] for k2 in ["year", "seed_index", "n_train", "best_epoch", "best_val_auc", "test_auc", "minutes"]}))


def parse_years(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["job", "run"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--years", default="2016-2023")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_dir = ROOT / ("wf_smoke" if a.smoke else "wf")
    for sub in ["forecasts", "meta", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    if a.mode == "job":
        run_job(a.year, a.seed_index, a.smoke, a.threads, out_dir)
        return

    jobs = [(y, k) for y in parse_years(a.years) for k in range(a.seeds)
            if not (out_dir / "meta" / f"Y{y}_s{k}.json").exists()]
    print(f"{len(jobs)} jobs to run with {a.workers} workers x {a.threads} threads")
    running: list[tuple[subprocess.Popen, tuple[int, int], object]] = []
    failed = []
    while jobs or running:
        while jobs and len(running) < a.workers:
            y, k = jobs.pop(0)
            log = open(out_dir / "logs" / f"Y{y}_s{k}.log", "w")
            cmd = [sys.executable, str(Path(__file__)), "job", "--year", str(y), "--seed-index", str(k),
                   "--threads", str(a.threads)] + (["--smoke"] if a.smoke else [])
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), (y, k), log))
        time.sleep(5)
        for item in running[:]:
            p, yk, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
                if p.returncode != 0:
                    failed.append(yk)
                print(f"[{time.strftime('%H:%M:%S')}] job Y{yk[0]} s{yk[1]} {status}", flush=True)
    print("done; failed:", failed)


if __name__ == "__main__":
    main()
