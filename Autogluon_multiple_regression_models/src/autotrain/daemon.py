from __future__ import annotations
import os, time, subprocess
from pathlib import Path

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
WATCH_PATH = Path(os.getenv("AUTO_WATCH_PATH", "/incoming"))
A_GLOB = os.getenv("AUTO_A_GLOB", "*_A.csv")
B_GLOB = os.getenv("AUTO_B_GLOB", "*_B.csv")
TASK = os.getenv("AUTO_TASK", "churn")
TARGET = os.getenv("AUTO_TARGET", "churned")
JOIN_KEYS = os.getenv("AUTO_JOIN_KEYS", "customer_id")
PROMOTE = os.getenv("AUTO_PROMOTE", "1") in {"1","true","True","YES","yes"}
INTERVAL = int(os.getenv("AUTO_INTERVAL_SECONDS", "30"))

def newest(p: Path, pattern: str) -> Path | None:
    files = sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0] if files else None

def train(a_path: Path, b_path: Path):
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    cmd = [
        "python", "-m", "src.pipeline.train",
        "--task", TASK, "--data-csv-a", str(a_path), "--data-csv-b", str(b_path),
        "--target", TARGET, "--join-keys", JOIN_KEYS,
    ]
    if PROMOTE: cmd.append("--promote")
    print(f"[autotrain] start: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(f"[autotrain] returncode={res.returncode}")
    if res.stdout: print(res.stdout)
    if res.stderr: print(res.stderr)
    return res.returncode == 0

def main():
    WATCH_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[autotrain] watching {WATCH_PATH} | A:{A_GLOB} B:{B_GLOB} | interval={INTERVAL}s")
    last_combo: tuple[str,str] | None = None
    while True:
        try:
            a = newest(WATCH_PATH, A_GLOB)
            b = newest(WATCH_PATH, B_GLOB)
            if a and b:
                combo = (a.name, b.name)
                if combo != last_combo:
                    ok = train(a, b)
                    if ok: last_combo = combo
            time.sleep(INTERVAL)
        except Exception as e:
            print("[autotrain] error:", e)
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
