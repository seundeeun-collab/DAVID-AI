"""Run nanoGPT training and send live numbers to your website's dashboard.

export SITE_URL="https://your-site.lovable.app"
export INGEST_TOKEN="(the token from your website's Settings)"
python train_with_dashboard.py config/train_code.py [--any=train.py --options]

Training never stops because of the dashboard: if the site can't be reached,
numbers are skipped and training goes on.
"""
import json, os, re, subprocess, sys, threading, time, urllib.request

SITE = os.environ.get("SITE_URL", "").rstrip("/")
TOKEN = os.environ.get("INGEST_TOKEN", "")
RUN_ID = os.environ.get("RUN_ID") or time.strftime("run-%Y%m%d-%H%M%S")
ITER = re.compile(r"iter (\d+): loss ([\d.]+), time ([\d.]+)ms")
STEP = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")

def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "unknown"

def max_iters(args):
    for a in args:
        if a.startswith("--max_iters="): return int(a.split("=")[1])
    for a in args:
        if a.endswith(".py") and os.path.exists(a):
            m = re.search(r"^max_iters\s*=\s*(\d+)", open(a).read(), re.M)
            if m: return int(m.group(1))
    return None

def send(payload):
    if not SITE or not TOKEN: return
    def go():
        try:
            req = urllib.request.Request(f"{SITE}/api/public/training/ingest", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", "x-ingest-token": TOKEN})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            print(f"[dashboard] could not send update: {e}", flush=True)
    threading.Thread(target=go, daemon=True).start()

def main():
    args = sys.argv[1:]
    base = {"run_id": RUN_ID, "gpu": gpu_name(), "max_iters": max_iters(args)}
    if not SITE or not TOKEN:
        print("[dashboard] SITE_URL or INGEST_TOKEN not set: training without live dashboard.", flush=True)
    send(dict(base, event="status", status="running"))
    proc = subprocess.Popen([sys.executable, "-u", "train.py", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        if m := ITER.search(line):
            send(dict(base, event="iter", step=int(m[1]), loss=float(m[2]), ms_per_step=float(m[3])))
        elif m := STEP.search(line):
            send(dict(base, event="eval", step=int(m[1]), train_loss=float(m[2]), val_loss=float(m[3])))
    code = proc.wait()
    send(dict(base, event="status", status="finished" if code == 0 else "error", exit_code=code))
    time.sleep(3)
    sys.exit(code)

if __name__ == "__main__":
    main()
