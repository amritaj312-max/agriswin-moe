"""The one path never run on real data: PlantVillage loading and cleaning."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

cfg = P.Config(root=os.path.join(os.environ.get("TMP", "."), "agft_pv"), workers=0)
P.prepare_dirs(cfg)
pv, classes = P.load_plantvillage(cfg, log=lambda m: print("   ", m.strip()))
print(f"  classes: {len(classes)}")
print(f"  images:  {len(pv)}")
labels = pv["label"]
print(f"  label range: {min(labels)}..{max(labels)} (want 0..{len(classes)-1})")
assert min(labels) == 0 and max(labels) == len(classes) - 1, "labels not remapped contiguously"
assert "Background_without_leaves" not in classes, "background class not dropped"
row = pv[0]
print(f"  row keys: {list(row)}")
print(f"  image type: {type(row['image']).__name__}, size {row['image'].size}")
h = P.dhash(row["image"])
print(f"  dhash works: {h}")
print("\nPlantVillage path OK")
