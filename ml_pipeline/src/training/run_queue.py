"""
Run several trainings one after another, committing each result as it lands.

For long unattended stretches -- the remaining baselines now, the ablation
matrix in Phase 3 -- so the GPU never idles between runs and every finished
result is in git, and pushed if a remote is set, seconds after it finishes.

The queue file holds one train.py argument line per run (forward slashes in
paths). Finished runs are skipped and interrupted ones resumed, so after a
crash or a restart, start the same queue again.

Beyond train.py's own safeguards, the queue:
  - waits for the charger before each run, since on battery the GPU runs at a
    third of its speed and empties the battery within the hour;
  - stops, without committing, if a paper run's test accuracy is below
    --min-acc or exactly 100% -- either deserves a look before more GPU time;
  - resumes a run that crashed part-way, up to twice, before giving up.

Its own log is checkpoints/queue.log; each run's stderr (where a crash's
traceback goes) is kept in checkpoints/<run>/stderr.log.

Usage:
    python ml_pipeline/src/training/run_queue.py ml_pipeline/configs/queues/baselines.txt
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data"))

import paths  # noqa: E402
from power import keep_awake, on_mains_power  # noqa: E402

TRAIN = os.path.join(HERE, "train.py")
CHECKPOINT_DIR = os.path.join(paths.REPO_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(paths.REPO_ROOT, "results")
LOG = os.path.join(CHECKPOINT_DIR, "queue.log")
RETRIES = 2


def log(msg):
    line = f"[queue {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def git(*args, **kwargs):
    return subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True,
                          **kwargs)


def run_name(argv):
    """The name train.py will give a run: --set run_name, else the config's
    run_name, else its model."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--set", nargs="+", default=[])
    args, _ = parser.parse_known_args(argv)
    sets = dict(item.partition("=")[::2] for item in args.set)
    cfg = {}
    if args.config:
        with open(os.path.join(paths.REPO_ROOT, args.config), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    return sets.get("run_name") or cfg.get("run_name") or sets.get("model") or cfg.get("model")


def wait_for_mains(allow_battery):
    if allow_battery or on_mains_power() is not False:
        return
    log("on battery: waiting for the charger before starting the next run")
    while on_mains_power() is False:
        time.sleep(30)
    log("charger connected")


def train(argv, run):
    """Run train.py with its output shown live. stderr is also kept in the
    run's folder, because train.log does not capture a crash's traceback."""
    run_dir = os.path.join(CHECKPOINT_DIR, run)
    os.makedirs(run_dir, exist_ok=True)
    proc = subprocess.Popen([sys.executable, TRAIN, *argv], cwd=paths.REPO_ROOT,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    with open(os.path.join(run_dir, "stderr.log"), "a", encoding="utf-8") as err:
        for line in proc.stderr:
            sys.stderr.write(line)
            err.write(line)
            err.flush()
    return proc.wait()


def accept(run, min_acc, trailer):
    """Commit a finished paper run's result, unless it needs a human look.
    Returns False when the queue should stop."""
    results_path = os.path.join(RESULTS_DIR, f"{run}.json")
    with open(results_path, encoding="utf-8") as fh:
        test = json.load(fh)["test"]
    if test["acc"] < min_acc or test["acc"] >= 1.0:
        log(f"{run}: test accuracy {test['acc']:.2%} needs a look before more GPU time is spent "
            f"-- stopping the queue; the result is NOT committed")
        return False

    rel = os.path.relpath(results_path, paths.REPO_ROOT).replace("\\", "/")
    if git("ls-files", "--error-unmatch", rel).returncode == 0:
        return True                                   # already committed
    msg = (f"Result: {run} -- test acc {test['acc']:.2%}, macro-F1 {test['macro_f1']:.2%}, "
           f"ECE {test['ece']:.2%}")
    if trailer:
        msg += "\n\n" + trailer
    done = git("add", rel)
    if done.returncode == 0:
        done = git("commit", "-m", msg, "--", rel)
    if done.returncode != 0:
        log(f"{run}: could not commit {rel}: {(done.stderr or done.stdout).strip()[:300]}")
        return True                                   # the result file is safe on disk
    log(f"{run}: committed {rel}")

    if git("remote", "get-url", "origin").returncode == 0:
        # never prompt: a sign-in window at 3 a.m. would wait for nobody
        pushed = git("push", env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"))
        log(f"{run}: pushed" if pushed.returncode == 0 else
            f"{run}: push failed (the commit is safe locally): {pushed.stderr.strip()[:300]}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run trainings one after another, committing each result")
    parser.add_argument("queue", help="file with one train.py argument line per run")
    parser.add_argument("--min-acc", type=float, default=0.98,
                        help="stop if a paper run's test accuracy is below this (default 0.98)")
    parser.add_argument("--allow-battery", action="store_true",
                        help="do not wait for the charger (short debug queues)")
    parser.add_argument("--commit-trailer", default="", help="appended to each result's commit message")
    args = parser.parse_args()

    with open(args.queue, encoding="utf-8") as fh:
        items = [shlex.split(line) for line in fh if line.strip() and not line.lstrip().startswith("#")]

    keep_awake()      # for the whole queue, including the gaps between runs
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("AgriSwin training queue -- closing this window stops it")
    log(f"{args.queue}: {len(items)} run(s), pid {os.getpid()}")

    for argv in items:
        run = run_name(argv)
        if os.path.isfile(os.path.join(CHECKPOINT_DIR, run, "FINISHED")):
            log(f"{run}: already finished -- skipped")
        else:
            for attempt in range(RETRIES + 1):
                wait_for_mains(args.allow_battery)
                resume = os.path.isfile(os.path.join(CHECKPOINT_DIR, run, "last.pt"))
                log(f"{run}: {'resuming' if resume else 'starting'}"
                    + (f" (retry {attempt} of {RETRIES})" if attempt else ""))
                code = train(argv + (["--resume"] if resume else []), run)
                if code == 0:
                    break
                if not os.path.isfile(os.path.join(CHECKPOINT_DIR, run, "last.pt")):
                    log(f"{run}: train.py exited with {code} before finishing an epoch -- stopping "
                        f"the queue (see checkpoints/{run}/stderr.log)")
                    return 1
                log(f"{run}: train.py exited with {code} part-way")
            else:
                log(f"{run}: still failing after {RETRIES} resumes -- stopping the queue")
                return 1

        if run.startswith("dev_"):
            log(f"{run}: done (a dev run, so not committed)")
        elif not accept(run, args.min_acc, args.commit_trailer):
            return 1

    log("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
