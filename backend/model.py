# ═══════════════════════════════════════════
# FloraVision — Model Loading & Inference
# Includes smart preprocessing for real-world photos
# ═══════════════════════════════════════════

import io
import os
import gc

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torchvision import models, transforms

from flower_metadata import FLOWER_METADATA

torch.set_grad_enabled(False)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ── Config ──
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "resnet50_flower_model.pth")
NUM_CLASSES = 102

# ── Standard inference transforms ──
# Matches training test_transforms exactly — used AFTER our smart crop
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Hard-coded index → Oxford class ID mapping ──
# Derived from checkpoint class_to_idx (lexicographic ImageFolder order).
# Verified directly from resnet50_flower_model.pth
IDX_TO_OXFORD_CLASS = {
    0: 1,   1: 10,  2: 100, 3: 101, 4: 102, 5: 11,  6: 12,  7: 13,
    8: 14,  9: 15,  10: 16, 11: 17, 12: 18, 13: 19, 14: 2,  15: 20,
    16: 21, 17: 22, 18: 23, 19: 24, 20: 25, 21: 26, 22: 27, 23: 28,
    24: 29, 25: 3,  26: 30, 27: 31, 28: 32, 29: 33, 30: 34, 31: 35,
    32: 36, 33: 37, 34: 38, 35: 39, 36: 4,  37: 40, 38: 41, 39: 42,
    40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 5,
    48: 50, 49: 51, 50: 52, 51: 53, 52: 54, 53: 55, 54: 56, 55: 57,
    56: 58, 57: 59, 58: 6,  59: 60, 60: 61, 61: 62, 62: 63, 63: 64,
    64: 65, 65: 66, 66: 67, 67: 68, 68: 69, 69: 7,  70: 70, 71: 71,
    72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79,
    80: 8,  81: 80, 82: 81, 83: 82, 84: 83, 85: 84, 86: 85, 87: 86,
    88: 87, 89: 88, 90: 89, 91: 9,  92: 90, 93: 91, 94: 92, 95: 93,
    96: 94, 97: 95, 98: 96, 99: 97, 100: 98, 101: 99,
}

# Canonical Oxford-102 class names in label-id order (1..102).
OXFORD_102_NAMES = {
    1: "Pink Primrose", 2: "Hard-leaved Pocket Orchid", 3: "Canterbury Bells", 4: "Sweet Pea",
    5: "English Marigold", 6: "Tiger Lily", 7: "Moon Orchid", 8: "Bird of Paradise",
    9: "Monkshood", 10: "Globe Thistle", 11: "Snapdragon", 12: "Colt's Foot", 13: "King Protea",
    14: "Spear Thistle", 15: "Yellow Iris", 16: "Globe-flower", 17: "Purple Coneflower",
    18: "Peruvian Lily", 19: "Balloon Flower", 20: "Giant White Arum Lily", 21: "Fire Lily",
    22: "Pincushion Flower", 23: "Fritillary", 24: "Red Ginger", 25: "Grape Hyacinth",
    26: "Corn Poppy", 27: "Prince of Wales Feathers", 28: "Stemless Gentian", 29: "Artichoke",
    30: "Sweet William", 31: "Carnation", 32: "Garden Phlox", 33: "Love in the Mist",
    34: "Mexican Aster", 35: "Alpine Sea Holly", 36: "Ruby-lipped Cattleya", 37: "Cape Flower",
    38: "Great Masterwort", 39: "Siam Tulip", 40: "Lenten Rose", 41: "Barbeton Daisy",
    42: "Daffodil", 43: "Sword Lily", 44: "Poinsettia", 45: "Bolero Deep Blue", 46: "Wallflower",
    47: "Marigold", 48: "Buttercup", 49: "Oxeye Daisy", 50: "Common Dandelion", 51: "Petunia",
    52: "Wild Pansy", 53: "Primula", 54: "Sunflower", 55: "Pelargonium",
    56: "Bishop of Llandaff", 57: "Gaura", 58: "Geranium", 59: "Orange Dahlia",
    60: "Pink-yellow Dahlia", 61: "Cautleya Spicata", 62: "Japanese Anemone",
    63: "Black-eyed Susan", 64: "Silverbush", 65: "Californian Poppy", 66: "Osteospermum",
    67: "Spring Crocus", 68: "Bearded Iris", 69: "Windflower", 70: "Tree Poppy", 71: "Gazania",
    72: "Azalea", 73: "Water Lily", 74: "Rose", 75: "Thorn Apple", 76: "Morning Glory",
    77: "Passion Flower", 78: "Lotus", 79: "Toad Lily", 80: "Anthurium", 81: "Frangipani",
    82: "Clematis", 83: "Hibiscus", 84: "Columbine", 85: "Desert-rose", 86: "Tree Mallow",
    87: "Magnolia", 88: "Cyclamen", 89: "Watercress", 90: "Canna Lily", 91: "Hippeastrum",
    92: "Bee Balm", 93: "Ball Moss", 94: "Foxglove", 95: "Bougainvillea", 96: "Camellia",
    97: "Mallow", 98: "Mexican Petunia", 99: "Bromelia", 100: "Blanket Flower",
    101: "Trumpet Creeper", 102: "Blackberry Lily",
}


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in (name or "") if ch.isalnum())


def _build_metadata_lookup_by_name() -> dict:
    lookup = {}
    for meta in FLOWER_METADATA.values():
        key = _normalize_name(meta.get("name", ""))
        if key and key not in lookup:
            lookup[key] = meta
    return lookup


METADATA_BY_NAME = _build_metadata_lookup_by_name()


def get_metadata_for_oxford_id(oxford_id: int) -> dict:
    canonical_name = OXFORD_102_NAMES.get(oxford_id, f"Unknown (Oxford ID {oxford_id})")
    matched_meta = METADATA_BY_NAME.get(_normalize_name(canonical_name), {})

    return {
        "name": canonical_name,
        "scientific_name": matched_meta.get("scientific_name"),
        "sun": matched_meta.get("sun"),
        "water": matched_meta.get("water"),
        "soil": matched_meta.get("soil"),
        "bloom_months": matched_meta.get("bloom_months", []),
        "description": matched_meta.get(
            "description",
            f"{canonical_name} is one of the 102 flower classes in the Oxford-102 dataset."
        ),
        "fun_fact": matched_meta.get("fun_fact"),
        "meaning": matched_meta.get("meaning"),
        "wikipedia_url": matched_meta.get("wikipedia_url"),
    }


# ══════════════════════════════════════
# SMART PREPROCESSING
# Bridges the gap between real-world photos and Oxford-style images.
# Oxford images: close-up, centered, clean background.
# Real-world photos: wide shots, busy backgrounds, off-center flowers.
# ══════════════════════════════════════

def find_flower_region(image: Image.Image) -> tuple:
    """
    Find the most likely flower region using color saturation analysis.
    Flowers are typically the most saturated region in a photo.

    Returns:
        (left, upper, right, lower) crop box, or None to use full image
    """
    # Work on a small thumbnail for speed
    thumb = image.copy()
    thumb.thumbnail((128, 128), Image.LANCZOS)
    w, h = thumb.size

    # Convert to HSV-like analysis via numpy
    img_array = np.array(thumb.convert("RGB")).astype(float)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

    # Saturation proxy: max(R,G,B) - min(R,G,B)
    saturation = np.max(img_array, axis=2) - np.min(img_array, axis=2)

    # Avoid pure white/grey backgrounds — weight by non-grey pixels
    grey_mask = saturation < 30  # near-grey pixels
    saturation[grey_mask] = 0

    # Find the region with highest average saturation
    # Divide into a 3x3 grid and find the hottest cell
    cell_h, cell_w = h // 3, w // 3
    best_score = -1
    best_cell  = (1, 1)  # default: centre cell

    for row in range(3):
        for col in range(3):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell_sat = saturation[y1:y2, x1:x2].mean()
            if cell_sat > best_score:
                best_score = cell_sat
                best_cell  = (row, col)

    # If the centre cell is close to the best, just use centre (more stable)
    centre_score = saturation[cell_h:2*cell_h, cell_w:2*cell_w].mean()
    if centre_score >= best_score * 0.75:
        best_cell = (1, 1)

    # Map best cell back to full image coordinates
    orig_w, orig_h = image.size
    scale_x = orig_w / w
    scale_y = orig_h / h
    row, col = best_cell

    cx = int((col * cell_w + (col + 1) * cell_w) / 2 * scale_x)
    cy = int((row * cell_h + (row + 1) * cell_h) / 2 * scale_y)

    # Crop a square around the best centre point
    # Use 80% of the shorter dimension as crop size
    crop_size = int(min(orig_w, orig_h) * 0.80)
    half = crop_size // 2

    left  = max(0, cx - half)
    upper = max(0, cy - half)
    right  = min(orig_w, left + crop_size)
    lower  = min(orig_h, upper + crop_size)

    # Adjust if we hit the edges
    if right == orig_w:
        left = orig_w - crop_size
    if lower == orig_h:
        upper = orig_h - crop_size
    left  = max(0, left)
    upper = max(0, upper)

    return (left, upper, right, lower)


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Apply subtle enhancements to make real-world photos look more
    like the clean Oxford dataset images:
    - Slight contrast boost
    - Slight colour/saturation boost
    - Gentle sharpening
    """
    # Contrast boost
    image = ImageEnhance.Contrast(image).enhance(1.15)
    # Colour/saturation boost
    image = ImageEnhance.Color(image).enhance(1.20)
    # Sharpness boost
    image = ImageEnhance.Sharpness(image).enhance(1.10)
    return image


def smart_preprocess(image: Image.Image) -> Image.Image:
    """
    Full smart preprocessing pipeline for real-world photos.

    Steps:
    1. Find the most saturated (flower-likely) region
    2. Crop to that region
    3. Enhance contrast, colour, and sharpness
    4. Return ready for INFERENCE_TRANSFORMS
    """
    # Step 1 — Find flower region
    crop_box = find_flower_region(image)
    if crop_box is not None:
        image = image.crop(crop_box)

    # Step 2 — Enhance
    image = enhance_image(image)

    return image


# ══════════════════════════════════════
# MODEL
# ══════════════════════════════════════

def build_resnet50(num_classes: int) -> nn.Module:
    """
    Recreate the exact architecture from training.
    Must match build_model(arch='resnet50') in the notebook exactly.
    """
    model = models.resnet50(weights=None)
    input_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(input_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


def load_model() -> nn.Module:
    """
    Load trained ResNet50 from checkpoint.
    Returns model in eval mode on CPU.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please place resnet50_flower_model.pth in the backend/ folder."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model = build_resnet50(NUM_CLASSES)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    class_to_idx = checkpoint.get("class_to_idx") if isinstance(checkpoint, dict) else None
    if class_to_idx:
        idx_to_oxford_class = {}
        for class_id_str, idx in class_to_idx.items():
            try:
                idx_to_oxford_class[int(idx)] = int(class_id_str)
            except (ValueError, TypeError):
                continue
        if len(idx_to_oxford_class) == NUM_CLASSES:
            model.idx_to_oxford_class = idx_to_oxford_class
        else:
            model.idx_to_oxford_class = IDX_TO_OXFORD_CLASS
    else:
        model.idx_to_oxford_class = IDX_TO_OXFORD_CLASS

    model.eval()
    del checkpoint
    gc.collect()
    return model


# ══════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════

def predict(image_bytes: bytes, model: nn.Module) -> dict:
    """
    Run inference on raw image bytes with smart preprocessing.

    Args:
        image_bytes : raw bytes of the uploaded image
        model       : loaded ResNet50 in eval mode

    Returns:
        dict matching all fields expected by identify.js renderResult()
    """
    # ── Load image ──
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── Smart preprocessing for real-world photos ──
    image = smart_preprocess(image)

    # ── Standard inference transforms ──
    tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0)  # (1, 3, 224, 224)

    # ── Multi-crop ensemble for robustness ──
    # Run 3 crops (smart-cropped, full-image, center-only) and average
    # This reduces sensitivity to exactly where the flower is framed
    original   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    enhanced   = enhance_image(original)

    tensor_full   = INFERENCE_TRANSFORMS(enhanced).unsqueeze(0)
    tensor_smart  = tensor  # already smart-cropped above

    # Centre-biased crop: tighter than standard, looser than smart
    orig_w, orig_h = original.size
    tight_size = int(min(orig_w, orig_h) * 0.65)
    cx, cy = orig_w // 2, orig_h // 2
    half = tight_size // 2
    tight_crop = original.crop((
        max(0, cx - half), max(0, cy - half),
        min(orig_w, cx + half), min(orig_h, cy + half)
    ))
    tensor_tight = INFERENCE_TRANSFORMS(enhance_image(tight_crop)).unsqueeze(0)

    # ── Forward pass — average the three logits ──
    with torch.inference_mode():
        logits_smart = model(tensor_smart)
        logits_full  = model(tensor_full)
        logits_tight = model(tensor_tight)

        # Weighted average: smart crop gets most weight
        logits_avg = (logits_smart * 0.50 +
                      logits_full  * 0.25 +
                      logits_tight * 0.25)

        probs = torch.softmax(logits_avg, dim=1)

    # ── Top-3 predictions ──
    top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)
    top3_probs   = top3_probs[0].tolist()
    top3_indices = top3_indices[0].tolist()

    # ── Map indices → Oxford IDs → metadata ──
    idx_to_oxford_class = getattr(model, "idx_to_oxford_class", IDX_TO_OXFORD_CLASS)

    top3_results = []
    for prob, idx in zip(top3_probs, top3_indices):
        oxford_id = idx_to_oxford_class.get(idx)
        meta      = get_metadata_for_oxford_id(oxford_id)
        top3_results.append({
            "name":       meta.get("name", f"Unknown (Oxford ID {oxford_id})"),
            "oxford_id":  oxford_id,
            "confidence": round(prob, 4)
        })

    # ── Best prediction ──
    best_oxford_id  = top3_results[0]["oxford_id"]
    best_meta       = get_metadata_for_oxford_id(best_oxford_id)
    best_confidence = top3_results[0]["confidence"]

    # Out-of-distribution guard:
    # The model always predicts one of 102 classes, so low confidence/margin means uncertain fit.
    second_confidence = top3_results[1]["confidence"] if len(top3_results) > 1 else 0.0
    confidence_gap = round(best_confidence - second_confidence, 4)
    is_low_confidence = best_confidence < 0.45 or confidence_gap < 0.12

    # ── Build response ──
    response = {
        "name":            best_meta.get("name",            f"Unknown (Oxford ID {best_oxford_id})"),
        "scientific_name": best_meta.get("scientific_name", None),
        "confidence":      best_confidence,
        "confidence_gap":  confidence_gap,
        "is_low_confidence": is_low_confidence,
        "warning": (
            "Low-confidence match. This image may be outside FloraVision's Oxford-102 training set "
            "or the flower may be hard to isolate."
            if is_low_confidence else None
        ),
        "sun":             best_meta.get("sun",             None),
        "water":           best_meta.get("water",           None),
        "soil":            best_meta.get("soil",            None),
        "bloom_months":    best_meta.get("bloom_months",    []),
        "description":     best_meta.get("description",     "No description available."),
        "fun_fact":        best_meta.get("fun_fact",        None),
        "meaning":         best_meta.get("meaning",         None),
        "wikipedia_url":   best_meta.get("wikipedia_url",   None),
        "top3":            top3_results,
    }

    return response
