"""
PlantVillage class definitions and crop-family expert assignment.

This is the single source of truth for the 38-class label space and for which
MoE expert owns which class. Import from here rather than re-deriving the lists,
so the teacher, the student and the evaluation scripts can never drift apart.

NOTE: the crop counts here are the real PlantVillage counts. The original design
doc (docs/project_breakdown_v3_final.md) listed Potato=5, Apple=8, Others=7,
which does not match the dataset. Correct counts are Potato=3, Apple=4,
Others=13. See docs/ACTION_PLAN.md section 3.1.
"""

# The 38 PlantVillage directory names, in sorted order. class_idx is the index
# into this list, so sorting must stay stable -- do not reorder.
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(PLANTVILLAGE_CLASSES)

CLASS_TO_IDX = {name: i for i, name in enumerate(PLANTVILLAGE_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(PLANTVILLAGE_CLASSES)}


def crop_of(class_name):
    """Return the crop part of a PlantVillage directory name ('Apple___healthy' -> 'Apple')."""
    return class_name.split("___")[0]


def disease_of(class_name):
    """Return the disease part ('Apple___healthy' -> 'healthy')."""
    return class_name.split("___")[1]


# --- MoE expert assignment (6 experts) -------------------------------------
#
# Five single-crop experts plus one catch-all. The catch-all is deliberately
# heterogeneous (13 classes, 9 crops, 8 of them 'healthy'), which is the main
# weakness of a hand-assigned routing scheme and is worth ablating against a
# learned router. See docs/ACTION_PLAN.md section 3.2.

EXPERT_NAMES = ["Tomato", "Corn", "Grape", "Apple", "Potato", "Other"]
NUM_EXPERTS = len(EXPERT_NAMES)

_DEDICATED_CROPS = {
    "Tomato": 0,
    "Corn_(maize)": 1,
    "Grape": 2,
    "Apple": 3,
    "Potato": 4,
}
_OTHER_EXPERT = 5


def expert_of(class_name):
    """Return the expert index (0-5) that owns this class."""
    return _DEDICATED_CROPS.get(crop_of(class_name), _OTHER_EXPERT)


# expert_idx for every class, aligned with PLANTVILLAGE_CLASSES
EXPERT_ASSIGNMENT = [expert_of(c) for c in PLANTVILLAGE_CLASSES]


def expert_class_counts():
    """Return {expert_name: number_of_classes} -- the corrected 10/4/4/4/3/13 split."""
    counts = dict.fromkeys(EXPERT_NAMES, 0)
    for class_name in PLANTVILLAGE_CLASSES:
        counts[EXPERT_NAMES[expert_of(class_name)]] += 1
    return counts


def crop_class_counts():
    """Return {crop: number_of_classes} across all 14 PlantVillage crops."""
    counts = {}
    for class_name in PLANTVILLAGE_CLASSES:
        counts[crop_of(class_name)] = counts.get(crop_of(class_name), 0) + 1
    return counts


if __name__ == "__main__":
    print(f"Classes: {NUM_CLASSES}")
    print()
    print("Per-crop class counts:")
    for crop, n in sorted(crop_class_counts().items(), key=lambda kv: -kv[1]):
        print(f"  {crop:<28} {n:>2}")
    print()
    print(f"Per-expert class counts ({NUM_EXPERTS} experts):")
    for name, n in expert_class_counts().items():
        print(f"  {name:<28} {n:>2}")
    assert sum(expert_class_counts().values()) == NUM_CLASSES
    print("\nOK: expert assignment covers all 38 classes exactly once.")
