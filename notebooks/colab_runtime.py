"""
Everything a Colab runtime needs before it can train or evaluate, in one call.

The notebooks stay thin -- mount Drive, fetch the code, call setup() -- so that
nothing about training lives in a notebook cell, where it could drift from the
scripts the laptop runs. A Colab run and a laptop run go through the very same
train.py, recipe and frozen splits; only the machine differs.

What setup() does, and why each step is there:

  packages     installs the exact timm the results were made with. Colab
               already has torch, numpy, scipy and the rest.
  images       puts PlantVillage and PlantDoc on the runtime's local disk
               (reading 56,000 small files from Drive is far too slow): from a
               cache archive on Drive if there is one, else by downloading the
               pinned upstream commits and then writing that cache. Either way
               verify_data.py must then confirm every image is byte-for-byte
               the one the laptop trained on, or setup stops.
  checkpoints  links checkpoints/ to Drive, so weights, resume points and a
               copy of every result survive a disconnect.
  git          signs result commits, and pushes them if a GITHUB_TOKEN secret
               is set (see the notebook); without one, results stay on Drive.
  recovery     brings back results that a previous session finished but did
               not push, so nothing a disconnect interrupted is lost.

Colab only. On the laptop, nothing here is needed or used.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys

REPO = "/content/agriswin-moe"
DRIVE = "/content/drive/MyDrive/agriswin"
DATA = "/content/data"                       # the runtime's local disk
CACHE = f"{DRIVE}/data/agriswin_images.tar"
OUTPUTS = f"{DRIVE}/outputs"                  # where results go when they cannot be pushed
GIT_NAME = "amritaj312-max"
GIT_EMAIL = "amritaj312-max@users.noreply.github.com"
TOKEN_SECRET = "GITHUB_TOKEN"
IMAGE_DIRS = ["raw/plantvillage/raw/color", "raw/plantdoc/train", "raw/plantdoc/test"]


def sh(cmd, check=True):
    """Run a shell command in the repo, showing its output in the notebook.

    A plain subprocess writes to the runtime's own log, not to the cell, so
    the output is read back and printed line by line -- otherwise the
    image check's verdict, the evidence that matters, would never be seen.
    """
    proc = subprocess.Popen(cmd, shell=True, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    if check and code != 0:
        raise RuntimeError(f"command failed ({code}): {cmd}")
    return code


def git(*args):
    # never prompt: a notebook cell cannot answer, so git would just hang
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))


def gpu_report():
    """What this runtime has, and how many data-loading workers suit it."""
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True)
    if query.returncode != 0:
        raise RuntimeError("No GPU. In Colab: Runtime > Change runtime type > L4 GPU or A100 GPU, "
                           "then Runtime > Restart session and run the cells again.")
    name, memory = [s.strip() for s in query.stdout.strip().splitlines()[0].split(",")]
    cpus = os.cpu_count() or 2
    # The augmentation pipeline is CPU work. On the laptop 4 workers kept up
    # with an RTX 3050; an L4 or A100 eats images several times faster.
    workers = max(2, min(8, cpus - 1))
    with open("/proc/meminfo") as fh:
        ram_gb = int(re.search(r"MemTotal:\s+(\d+)", fh.read()).group(1)) / 1024 ** 2
    return {"gpu": name, "gpu_memory": memory, "cpus": cpus, "ram_gb": round(ram_gb),
            "num_workers": workers}


def install():
    """The exact timm the results were made with, read from requirements.txt
    so the two can never disagree."""
    with open(f"{REPO}/ml_pipeline/requirements.txt", encoding="utf-8") as fh:
        pin = re.search(r"^(timm==[\w.]+)", fh.read(), re.M).group(1)
    sh(f"{sys.executable} -m pip install -q {pin}")
    return pin


def images_present():
    return all(os.path.isdir(f"{DATA}/{d}") and os.listdir(f"{DATA}/{d}") for d in IMAGE_DIRS)


def prepare_data():
    """Images on local disk, verified byte for byte, from the Drive cache when
    there is one. Returns where they came from."""
    os.environ["AGRISWIN_DATA_ROOT"] = DATA      # read by paths.data_root(), in every subprocess
    source = "already on this runtime"
    if not images_present():
        if os.path.isfile(CACHE):
            source = "the cache on Drive"
            print(f"Copying images from {CACHE} ...")
            os.makedirs(DATA, exist_ok=True)
            sh(f'tar -xf "{CACHE}" -C "{DATA}"')
        else:
            source = "a fresh download (pinned upstream commits)"
            print("No image cache on Drive yet: downloading (a few minutes, once)...")
            sh(f"{sys.executable} ml_pipeline/src/data/download.py --dataset all")

    print("\nChecking every image against the reference the results were made on...")
    if sh(f"{sys.executable} ml_pipeline/src/data/verify_data.py", check=False) != 0:
        raise RuntimeError("These images differ from the ones the results were made on. Do not "
                           "train here until this is fixed; see the output above.")

    if not os.path.isfile(CACHE):
        print(f"\nWriting the image cache to Drive for next time ({CACHE}) ...")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        partial = CACHE + ".partial"            # a disconnect mid-write leaves no half archive
        sh(f'tar -cf "{partial}" -C "{DATA}" ' + " ".join(IMAGE_DIRS))
        os.replace(partial, CACHE)
    return source


def link_checkpoints():
    """checkpoints/ -> Drive. It is git-ignored, so the link leaves the clone
    clean, which train.py insists on before a paper run."""
    target, link = f"{DRIVE}/checkpoints", f"{REPO}/checkpoints"
    os.makedirs(target, exist_ok=True)
    if os.path.islink(link):
        return target
    if os.path.isdir(link):                     # made before the link existed: keep its contents
        for item in os.listdir(link):
            shutil.move(os.path.join(link, item), os.path.join(target, item))
        os.rmdir(link)
    os.symlink(target, link)
    return target


def configure_git():
    """Sign commits; enable pushing if a GITHUB_TOKEN secret is set.

    Returns "ok", "no secret" or "cannot push". The token never touches disk
    or output: it is read from Colab's secret store into this process's
    environment, and a credential helper hands it to git from there.
    """
    git("config", "--global", "user.name", GIT_NAME)
    git("config", "--global", "user.email", GIT_EMAIL)
    try:
        from google.colab import userdata
        token = userdata.get(TOKEN_SECRET)
    except Exception:                           # noqa: BLE001 -- no secret, or no notebook access
        token = None
    if not token:
        return "no secret"
    os.environ[TOKEN_SECRET] = token
    helper = ('!f() { test "$1" = get && echo username=x-access-token && '
              f'echo "password=${TOKEN_SECRET}"; }}; f')
    git("config", "--global", "credential.helper", helper)
    # The repository is public, so reading it proves nothing; a dry-run push
    # authenticates for writing without changing anything.
    check = git("push", "--dry-run")
    if check.returncode != 0:
        print(f"\nThe {TOKEN_SECRET} secret was found but GitHub refused it for pushing:\n"
              f"  {check.stderr.strip()[:300]}\n"
              f"A 403 'Permission ... denied' means the token works but may not write here. On "
              f"GitHub, edit the token (Settings > Developer settings > Fine-grained tokens): "
              f"Repository access must be 'Only select repositories' with agriswin-moe chosen "
              f"(the default, 'Public repositories', is read-only), and Repository permissions "
              f"> Contents must be 'Read and write'. Editing keeps the same token, so the secret "
              f"needs no change; run this cell again afterwards.\n")
        return "cannot push"
    return "ok"


def push_pending():
    """Push commits made here that never reached GitHub -- results a queue
    committed while pushing was not yet possible. Returns how many."""
    ahead = git("rev-list", "--count", "@{upstream}..HEAD")
    count = int(ahead.stdout.strip() or 0) if ahead.returncode == 0 else 0
    if count:
        git("pull", "--rebase")                 # onto anything pushed meanwhile
        pushed = git("push")
        if pushed.returncode != 0:
            print(f"Could not push {count} pending commit(s): {pushed.stderr.strip()[:200]}")
            return 0
    return count


def acceptable(result_path):
    """The queue's own gate: a result outside 98-100% is not committed
    unattended, here either."""
    with open(result_path, encoding="utf-8") as fh:
        acc = json.load(fh)["test"]["acc"]
    return 0.98 <= acc < 1.0


def recover_results():
    """Results a previous session finished, found in their Drive copies
    (checkpoints/<run>/result.json) and missing from this fresh clone."""
    recovered = []
    for marker in sorted(glob.glob(f"{REPO}/checkpoints/*/FINISHED")):
        run = os.path.basename(os.path.dirname(marker))
        src, dst = f"{REPO}/checkpoints/{run}/result.json", f"{REPO}/results/{run}.json"
        if not run.startswith("dev_") and os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            recovered.append(run)
    return recovered


def commit_and_push(paths, message, can_push):
    """Commit `paths` and push them; without push access, copy them to Drive."""
    paths = [p for p in paths if os.path.exists(f"{REPO}/{p}")]
    if not paths:
        return
    git("add", "--", *paths)
    if git("diff", "--cached", "--quiet").returncode == 0:
        print("Nothing new to commit.")
        return
    git("commit", "-m", message)
    if can_push:
        pushed = git("push")
        if pushed.returncode == 0:
            print(f"Committed and pushed: {', '.join(paths)}")
            return
        print(f"Push failed ({pushed.stderr.strip()[:200]}).")
    os.makedirs(OUTPUTS, exist_ok=True)
    for p in paths:
        src, dst = f"{REPO}/{p}", f"{OUTPUTS}/{p}"
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print(f"Not pushed: copied to {OUTPUTS} instead. Set the {TOKEN_SECRET} secret to push.")


def runs_needing_evaluation():
    """Finished runs with weights on this runtime and no evaluation yet."""
    runs = []
    for path in sorted(glob.glob(f"{REPO}/results/*.json")):
        run = os.path.basename(path)[:-len(".json")]
        if (not run.startswith("dev_") and os.path.isfile(f"{REPO}/checkpoints/{run}/best.pt")
                and not os.path.isfile(f"{REPO}/results/eval/{run}.json")):
            runs.append(run)
    return runs


def setup():
    """Prepare this runtime. Returns what the next cells need."""
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError("Google Drive is not mounted: run the drive.mount cell first.")
    report = gpu_report()
    print(f"GPU {report['gpu']} ({report['gpu_memory']}), {report['cpus']} CPUs, "
          f"{report['ram_gb']} GB RAM -> {report['num_workers']} data-loading workers\n")

    pin = install()
    source = prepare_data()
    checkpoints = link_checkpoints()
    token = configure_git()
    can_push = token == "ok"
    pending = push_pending() if can_push else 0
    if pending:
        print(f"Pushed {pending} result commit(s) an earlier queue could not push.")

    recovered = recover_results()
    good = [r for r in recovered if acceptable(f"{REPO}/results/{r}.json")]
    for run in sorted(set(recovered) - set(good)):
        print(f"WARNING: {run} finished outside 98-100% test accuracy; left uncommitted for a look")
    if good:
        commit_and_push([f"results/{r}.json" for r in good],
                        "Results recovered from an interrupted Colab session: " + ", ".join(good),
                        can_push)

    print("\n" + "=" * 70)
    print(f"  code         {git('log', '--oneline', '-1').stdout.strip()}")
    print(f"  packages     {pin}")
    print(f"  images       verified, from {source}")
    print(f"  checkpoints  {checkpoints}")
    print(f"  results      " + {
        "ok": "committed and pushed to GitHub as each run finishes",
        "no secret": f"kept on Drive (no {TOKEN_SECRET} secret, so not pushed)",
        "cannot push": f"kept on Drive ({TOKEN_SECRET} cannot push yet: see the note above)",
    }[token])
    if recovered:
        print(f"  recovered    {', '.join(recovered)}")
    print("=" * 70)
    return {**report, "can_push": can_push}
