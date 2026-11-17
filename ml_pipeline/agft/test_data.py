"""The parts the synthetic tests cannot reach: real HuggingFace loading.

Uses PlantDoc only (2,922 images, a small download), because the code path is
the same one PlantVillage takes.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

root = tempfile.mkdtemp(prefix="agft_data_")
cfg = P.Config(root=root, workers=0, img_size=64)
P.prepare_dirs(cfg)
ok = True

def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")

from pipeline import PLANTDOC_TO_PV
from datasets import load_dataset_builder

# The real class names of the pinned PlantVillage release, from its metadata
# (no 2 GB download), minus the dropped class -- exactly what load_plantvillage
# hands to load_plantdoc. Checking the bridge against its own targets instead
# is how four wrong names once passed this test.
print("0. The bridge targets classes the real PlantVillage has")
names = load_dataset_builder(cfg.pv_dataset, revision=cfg.pv_revision).info.features["label"].names
pv_classes = [n for n in names if n not in cfg.drop_classes]
absent = sorted(v for v in PLANTDOC_TO_PV.values() if v not in pv_classes)
check("every one of the 29 targets is a real PlantVillage class", not absent,
      f"{len(pv_classes)} classes" if not absent else f"absent: {absent}")
print("\n1. PlantDoc loads and relabels into the PlantVillage space")
try:
    pd_data, reachable = P.load_plantdoc(pv_classes, cfg, log=lambda m: print("   ", m.strip()))
    check("PlantDoc downloaded and mapped", len(pd_data) > 2000, f"{len(pd_data)} images")
    labels = pd_data["label"]
    check("every label is a valid class index",
          min(labels) >= 0 and max(labels) < len(pv_classes))
    targets = {pv_classes.index(v) for v in set(PLANTDOC_TO_PV.values()) if v in pv_classes}
    check("reachable classes are a subset of the mapped targets",
          set(reachable) <= targets,
          f"{len(reachable)} reachable of {len(targets)} mapped "
          f"(rare categories vanish once ambiguous images are dropped)")
    row = pd_data[0]
    check("rows carry a decodable image", hasattr(row["image"], "convert"),
          type(row["image"]).__name__)
except Exception as e:
    check("PlantDoc loads", False, f"{type(e).__name__}: {e}")
    raise SystemExit(1)

print("\n2. Hashing and splitting on real photographs")
subset = pd_data.select(range(400))
hashes = [P.dhash(subset[i]["image"]) for i in range(len(subset))]
check("a hash per image", len(hashes) == 400)
check("hashes look like 16 hex characters",
      all(len(h) == 16 for h in hashes), hashes[0])
dups = len(hashes) - len(set(hashes))
check("duplicate detection runs on real data", dups >= 0, f"{dups} near-duplicates found")
splits = P.grouped_split([int(subset[i]["label"]) for i in range(400)], hashes, cfg,
                         log=lambda m: print("   ", m.strip()))
check("no leakage between the three splits",
      P.check_no_leakage(splits, hashes, log=lambda m: print("   ", m.strip())))

print("\n3. A real batch flows through a real model")
import torch
loader = P.loader_for(subset, splits["val"][:16], cfg, batch_size=8)
images, y = next(iter(loader))
check("batch shape is right", tuple(images.shape[1:]) == (3, cfg.img_size, cfg.img_size),
      str(tuple(images.shape)))
model = P.build_model("agft:gated", len(pv_classes), cfg, pretrained=False)
with torch.no_grad():
    logits, g = model(images, return_gate=True)
check("the gated model consumes a real batch",
      logits.shape == (len(y), len(pv_classes)), str(tuple(logits.shape)))
check("the gate is per image", g.shape[0] == len(y), str(tuple(g.shape)))

import shutil; shutil.rmtree(root, ignore_errors=True)
print(f"\n{'all data checks passed' if ok else 'SOME DATA CHECKS FAILED'}")
raise SystemExit(0 if ok else 1)
