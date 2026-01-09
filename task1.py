from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ObsConfig:
    window_seconds: int = 60
    cpu_bins: Tuple[float, float] = (50.0, 80.0)
    mem_bins: Tuple[float, float] = (60.0, 85.0)
    lat_bins: Tuple[float, float] = (200.0, 800.0)
    err_bins: Tuple[int, int] = (0, 5)
    auth_bins: Tuple[int, int] = (0, 3)

def _bin3(x: float, t1: float, t2: float, labels=("LOW", "MED", "HIGH")) -> str:
    return labels[0] if x <= t1 else (labels[1] if x <= t2 else labels[2])

def _binc(x: float, t0: int, t1: int, labels=("NONE", "LOW", "HIGH")) -> str:
    x = int(round(x))
    return labels[0] if x <= t0 else (labels[1] if x <= t1 else labels[2])

def build_observations(df: pd.DataFrame, cfg: ObsConfig = ObsConfig()
                       ) -> Tuple[List[int], Dict[str, int], pd.DataFrame]:
    """
    Task 1: window logs/metrics -> discretize -> observation IDs for HMM.
    Required cols: timestamp,cpu_pct,mem_pct,latency_ms,error_count,auth_fail_count
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").set_index("timestamp")
    w = f"{cfg.window_seconds}S"

    win = pd.DataFrame({
        "cpu": df["cpu_pct"].resample(w).mean(),
        "mem": df["mem_pct"].resample(w).mean(),
        "lat": df["latency_ms"].resample(w).quantile(0.95),
        "err": df["error_count"].resample(w).sum(),
        "auth": df["auth_fail_count"].resample(w).sum(),
    }).dropna().reset_index()

    win["cpuL"]  = win["cpu"].apply(lambda v: _bin3(v, *cfg.cpu_bins))
    win["memL"]  = win["mem"].apply(lambda v: _bin3(v, *cfg.mem_bins))
    win["latL"]  = win["lat"].apply(lambda v: _bin3(v, *cfg.lat_bins))
    win["errL"]  = win["err"].apply(lambda v: _binc(v, *cfg.err_bins))
    win["authL"] = win["auth"].apply(lambda v: _binc(v, *cfg.auth_bins))

    win["symbol"] = win.apply(
        lambda r: f"C{r.cpuL}|M{r.memL}|L{r.latL}|E{r.errL}|A{r.authL}", axis=1
    )

    vocab = {s: i for i, s in enumerate(sorted(win["symbol"].unique()))}
    obs_ids = [vocab[s] for s in win["symbol"]]
    return obs_ids, vocab, win

# Optional quick test (remove in submission if you want)
if __name__ == "__main__":
    n = 200
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="5S"),
        "cpu_pct": np.random.normal(55, 15, n).clip(0, 100),
        "mem_pct": np.random.normal(65, 10, n).clip(0, 100),
        "latency_ms": np.random.lognormal(np.log(150), 0.5, n).clip(10, 5000),
        "error_count": np.random.poisson(0.2, n),
        "auth_fail_count": np.random.poisson(0.05, n),
    })
    obs, vocab, table = build_observations(df)
    print(len(obs), "observations,", len(vocab), "symbols")
    print(table.head())
