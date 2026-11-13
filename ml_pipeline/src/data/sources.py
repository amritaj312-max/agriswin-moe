"""Who photographed each PlantVillage image, recovered from its filename.

PlantVillage names encode the photo session that produced the image:

    Tomato___Bacterial_spot/GCREC_Bact.Sp 3434.JPG   -> source GCREC
    Potato___Early_blight/RS_Early.B 7777.JPG        -> source RS

The leading code is a site or photographer -- GCREC is Florida's Gulf Coast
Research and Education Center, PSU is Penn State, Crnl is Cornell. The rest of
the tag repeats the disease, which is already the folder name, so only the
leading code is kept.

Why this matters more than it looks
-----------------------------------
Class and photo session are badly confounded here: most classes were shot in
a single session, so "what this disease looks like" and "what that afternoon's
camera, bench and lighting looked like" are the same signal in the training
data. That is the shortcut the field collapse exposes -- background pixels
alone already predict the class 45% of the time (chance 2.6%).

Unusually, the confounder is *recorded*: robustness benchmarks normally have
to manufacture one. Nothing published appears to use these codes, so the table
this builds is released with the paper.

It is also why the obvious fixes do not apply. GroupDRO, DANN, IRM and DFR all
need most class-by-source cells to hold images; here most are empty, so
telling a model to ignore the source would tell it to ignore the class. Only
the handful of classes shot by two or more sessions can separate the two, and
that handful is what the method has to work from.

Reading the summary
-------------------
    python ml_pipeline/src/data/sources.py           # the confound, in numbers
    python ml_pipeline/src/data/sources.py --build   # refreeze the table
"""

import argparse
import collections
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402

SOURCE_FILE = os.path.join(paths.SPLIT_DIR, "plantvillage_sources.csv")
SPLITS = ("train", "val", "test")

# One session, written more than one way. Each merge below is supported by the
# classes the two spellings appear in, printed by the summary:
#   R.S    only Corn healthy, while RS covers 14 classes but not that one --
#          complementary coverage under the same initials. This is the one
#          judgement call here; drop the line to keep them apart.
#   Rut.   only Peach bacterial spot, which Rutg. also shot.
#   MD     only Squash powdery mildew, which UMD also shot (U. of Maryland).
#   GHLB2  the same greenhouse late-blight series as GHLB.
ALIASES = {"R.S": "RS", "Rut": "Rutgers", "Rutg": "Rutgers",
           "MD": "UMD", "GHLB2": "GHLB"}
MIN_IMAGES = 30          # a session smaller than this cannot teach its signature


def source_of(path):
    """The photo session's code, from an image's filename.

    A few names carry no '___', so the whole stem is the tag; the code is then
    whatever precedes the first separator, with spelling variants merged.
    """
    stem = os.path.basename(path).rsplit(".", 1)[0]
    tag = stem.split("___", 1)[1] if "___" in stem else stem
    head = re.split(r"[_\s]", tag.strip())[0] if tag.strip() else ""
    head = head.rstrip(".")
    return ALIASES.get(head, head) or "unknown"


def read_rows():
    """Every split's rows, with the split name and the source attached."""
    rows = []
    for split in SPLITS:
        with open(os.path.join(paths.SPLIT_DIR, f"plantvillage_{split}.csv"), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append({"split": split, "path": row["path"], "class_name": row["class_name"],
                             "class_idx": row["class_idx"], "crop": row["crop"],
                             "source": source_of(row["path"])})
    return rows


def structure(rows):
    """The confound, as the numbers the method has to live with.

    Returns the per-class source counts, and the classes a session can be
    learned from: those holding two or more sessions of at least MIN_IMAGES,
    where the same disease was photographed twice and so source and class
    finally vary independently.
    """
    per_class = collections.defaultdict(collections.Counter)
    for row in rows:
        per_class[row["class_name"]][row["source"]] += 1
    usable = {cls: counts for cls, counts in per_class.items()
              if sum(1 for n in counts.values() if n >= MIN_IMAGES) >= 2}
    return per_class, usable


def build():
    rows = read_rows()
    with open(SOURCE_FILE, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "class_name", "class_idx", "crop",
                                                "split", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"wrote {os.path.relpath(SOURCE_FILE, paths.REPO_ROOT)}: {len(rows)} images")


def summarise():
    rows = read_rows()
    per_class, usable = structure(rows)
    sources = collections.Counter(row["source"] for row in rows)
    per_source = collections.defaultdict(set)
    for row in rows:
        per_source[row["source"]].add(row["class_name"])

    single = [c for c, counts in per_class.items()
              if sum(1 for n in counts.values() if n >= MIN_IMAGES) <= 1]
    print(f"{len(rows)} images, {len(sources)} photo sessions, {len(per_class)} classes\n")
    print(f"classes shot by ONE session only     {len(single)} of {len(per_class)}")
    print(f"classes a session can be learned in  {len(usable)}  "
          f"(two or more sessions of {MIN_IMAGES}+ images)")
    crops = {cls.split('___')[0] for cls in usable}
    print(f"   spanning {len(crops)} crops: {', '.join(sorted(crops))}")
    for cls, counts in sorted(usable.items()):
        kept = [(s, n) for s, n in counts.most_common() if n >= MIN_IMAGES]
        print(f"   {cls.replace('___', ' / '):52s} " + ", ".join(f"{s} {n}" for s, n in kept))

    print(f"\nsessions covering several classes (where erasing the session can be "
          f"checked\nagainst the class signal it must not take with it):")
    for source, classes in sorted(per_source.items(), key=lambda kv: -len(kv[1])):
        if len(classes) < 2:
            continue
        crops = {c.split("___")[0] for c in classes}
        print(f"   {source:10s} {len(classes):2d} classes across {len(crops)} crops "
              f"({sources[source]} images)")
    print(f"\nThe method estimates each session's signature in the {len(usable)} classes above "
          f"and\nremoves it from all {len(per_class)}; the sessions listed last are the check "
          f"that it\ntook the session and left the class.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--build", action="store_true", help="refreeze the table, then stop")
    args = parser.parse_args()
    if args.build:
        build()
    else:
        summarise()


if __name__ == "__main__":
    main()
