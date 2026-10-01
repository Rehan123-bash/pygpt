"""
push_day1.py - Drive the whole Kaggle workflow from the command line (no browser).

One-time setup: kaggle.com -> Settings -> API -> Create New Token -> save as
~/.kaggle/kaggle.json. The account must be phone-verified (internet-on kernels).

Day 1:
  python kaggle/push_day1.py code-dataset            # upload the repo as dataset <user>/pygpt-code
  python kaggle/push_day1.py nb01                    # data prep, CPU, ~40-60 min
  python kaggle/push_day1.py nb02 --mode sanity      # ~4 min on the GPUs
  python kaggle/push_day1.py nb02 --mode bench       # ~8 min; read tok/s from the log
  python kaggle/push_day1.py nb02 --mode main --max-iters 3800   # the 10 h run
Day 2:
  python kaggle/push_day1.py nb03                    # eval + export + space bundle
Anytime:
  python kaggle/push_day1.py status nb02 | log nb02 | watch nb02 | quota

Each push creates a new version of the same kernel and runs it to completion
(the API equivalent of "Save & Run All"). --accelerator sets the machine shape
(e.g. T4 x2) if the server rejects the default; see kernel-metadata docs.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BUILD = os.path.join(REPO, "build")

NB = {  # name -> (ipynb, slug, needs_gpu)
    "nb01": ("01_prepare_data.ipynb", "pygpt-01-prepare-data", False),
    "nb02": ("02_train.ipynb", "pygpt-02-train", True),
    "nb03": ("03_eval_and_demo.ipynb", "pygpt-03-eval-demo", True),
}


def kaggle_cli():
    exe = shutil.which("kaggle") or os.path.expandvars(
        r"%APPDATA%\Python\Python312\Scripts\kaggle.exe")
    if not os.path.exists(exe):
        sys.exit("kaggle CLI not found - pip install kaggle")
    return exe


def run(args, **kw):
    print("+", " ".join(args))
    return subprocess.run(args, text=True, capture_output=True, **kw)


def username():
    path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(path):
        sys.exit("missing ~/.kaggle/kaggle.json - kaggle.com -> Settings -> API -> Create New Token")
    return json.load(open(path))["username"]


def code_dataset(user):
    """Stage the committed repo files into a dataset folder and create/version it."""
    stage = os.path.join(BUILD, "code_dataset")
    shutil.rmtree(stage, ignore_errors=True)
    dst = os.path.join(stage, "pygpt")
    files = subprocess.run(["git", "ls-files"], cwd=REPO, text=True,
                           capture_output=True, check=True).stdout.split()
    for f in files:
        os.makedirs(os.path.dirname(os.path.join(dst, f)) or dst, exist_ok=True)
        shutil.copy(os.path.join(REPO, f), os.path.join(dst, f))
    with open(os.path.join(stage, "dataset-metadata.json"), "w") as f:
        json.dump({"id": f"{user}/pygpt-code", "title": "pygpt-code",
                   "licenses": [{"name": "CC0-1.0"}]}, f)
    r = run([kaggle_cli(), "datasets", "create", "-p", stage, "--dir-mode", "zip"])
    out = r.stdout + r.stderr
    print(out.strip())
    if "already exists" in out or "409" in out:
        r = run([kaggle_cli(), "datasets", "version", "-p", stage, "--dir-mode", "zip",
                 "-m", "update " + time.strftime("%Y-%m-%d %H:%M")])
        print((r.stdout + r.stderr).strip())


def set_nb02_params(nb_json, args):
    """Rewrite the SET-THESE cell of notebook 02 in place."""
    subs = {"MODE": f'"{args.mode}"'}
    if args.micro_batch: subs["MICRO_BATCH"] = str(args.micro_batch)
    if args.grad_accum: subs["GRAD_ACCUM"] = str(args.grad_accum)
    if args.max_iters: subs["MAX_ITERS"] = str(args.max_iters)
    if args.time_limit: subs["TIME_LIMIT_H"] = str(args.time_limit)
    if args.resume_path: subs["RESUME_PATH"] = f'"{args.resume_path}"'
    for cell in nb_json["cells"]:
        if cell["cell_type"] == "code" and "SET THESE" in cell["source"]:
            src = cell["source"]
            for key, val in subs.items():
                src, n = re.subn(rf"^{key} = [^#\n]*", f"{key} = {val:<22}", src, flags=re.M)
                assert n == 1, f"could not set {key}"
            cell["source"] = src
            print("set:", ", ".join(f"{k}={v}" for k, v in subs.items()))
            return
    sys.exit("SET THESE cell not found in notebook 02")


def push(user, name, args):
    ipynb, slug, gpu = NB[name]
    stage = os.path.join(BUILD, "push_" + name)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    nb_json = json.load(open(os.path.join(REPO, "notebooks", ipynb)))
    if name == "nb02":
        set_nb02_params(nb_json, args)
    json.dump(nb_json, open(os.path.join(stage, ipynb), "w"), indent=1)

    kernel_sources = []
    if name == "nb02" and args.mode != "sanity":
        kernel_sources = [f"{user}/{NB['nb01'][1]}"]   # token data from notebook 01's output
    if name == "nb03":
        kernel_sources = [f"{user}/{NB['nb01'][1]}", f"{user}/{NB['nb02'][1]}"]

    meta = {
        "id": f"{user}/{slug}", "title": slug,
        "code_file": ipynb, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/pygpt-code"],
        "kernel_sources": kernel_sources, "competition_sources": [], "model_sources": [],
    }
    if args.accelerator:
        meta["machine_shape"] = args.accelerator
    json.dump(meta, open(os.path.join(stage, "kernel-metadata.json"), "w"), indent=1)

    cmd = [kaggle_cli(), "kernels", "push", "-p", stage]
    if args.timeout_s:
        cmd += ["-t", str(args.timeout_s)]
    r = run(cmd)
    print((r.stdout + r.stderr).strip())
    if r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower():
        print(f"\npushed & running: https://www.kaggle.com/code/{user}/{slug}")
        print(f"follow with: python kaggle/push_day1.py watch {name}")


def status(user, name, verbose=True):
    r = run([kaggle_cli(), "kernels", "status", f"{user}/{NB[name][1]}"])
    out = (r.stdout + r.stderr).strip()
    if verbose:
        print(out)
    return out


def watch(user, name, interval=90):
    """Poll until the latest version stops running, then print the tail of the log."""
    while True:
        out = status(user, name, verbose=False)
        print(time.strftime("%H:%M"), out)
        if any(s in out.lower() for s in ("complete", "error", "cancel")):
            break
        time.sleep(interval)
    log(user, name)


def log(user, name):
    r = run([kaggle_cli(), "kernels", "logs", f"{user}/{NB[name][1]}"])
    out = (r.stdout + r.stderr)
    print(out[-6000:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["code-dataset", "nb01", "nb02", "nb03",
                                    "status", "log", "watch", "quota"])
    ap.add_argument("which", nargs="?", choices=list(NB), help="notebook for status/log/watch")
    ap.add_argument("--mode", default="sanity", choices=["sanity", "bench", "main", "resume"])
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--max-iters", type=int)
    ap.add_argument("--time-limit", type=float)
    ap.add_argument("--resume-path")
    ap.add_argument("--accelerator", help='machine shape, if the account default is not T4 x2')
    ap.add_argument("--timeout-s", type=int, help="server-side max run seconds for this push")
    args = ap.parse_args()

    user = username()
    if args.cmd == "code-dataset":
        code_dataset(user)
    elif args.cmd in NB:
        push(user, args.cmd, args)
    elif args.cmd == "quota":
        r = run([kaggle_cli(), "quota"])
        print((r.stdout + r.stderr).strip())
    else:
        if not args.which:
            sys.exit(f"{args.cmd} needs a notebook name: nb01 | nb02 | nb03")
        {"status": status, "log": log, "watch": watch}[args.cmd](user, args.which)


if __name__ == "__main__":
    main()
