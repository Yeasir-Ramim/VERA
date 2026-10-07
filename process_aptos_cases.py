"""
Process 5 representative APTOS 2019 clinical cases across ICDR grades 0-4 through the full VERA pipeline:
1. Preprocessing (Circular Crop + Ben Graham + CLAHE)
2. Vessel segmentation (Frangi + Top-Hat)
3. Production ResNet-50 4-Channel inference & Grad-CAM++ generation
4. Clinical Biomarker analysis (ETDRS 4-quadrant, FAZ, OD, Lesion counts)
5. Save real assets to sample_data/vera_outputs/ and sample_data/images/
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from src.preprocessing import preprocess_fundus, crop_fundus_circle, apply_ben_graham, apply_clahe
from src.vessel_segmentation import VesselSegmenter
from src.models import build_model, load_checkpoint_safe
from src.explainability import GradCAMPlusPlus, overlay_cam_on_image, compute_vessel_attention_overlap
from src.clinical_biomarkers import analyze_clinical_biomarkers
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, VESSEL_MEAN, VESSEL_STD

# 1. Candidate selection from train.csv
df = pd.read_csv("data/raw/aptos2019/train.csv")
img_dir = Path("data/raw/aptos2019/train_images")

# Candidate IDs
candidates = {
    0: ["002c21358ce6", "00cc2b75cddd", "00f6c1be5a33", "014508ccb9cb"],
    1: ["0024cdab0c1e", "01b3aed3ed4c", "0124dffecf29", "0369f3efe69b"],
    2: ["000c1434d8d7", "00a8624548a9", "012a242ac6ff", "0161338f53cc"],
    3: ["0104b032c141", "03c85870824c", "042470a92154", "069f43616fab"],
    4: ["001639a390f0", "0243404e8a00", "02685f13cefd", "0083ee8054ee"]
}

# Select best candidate for each grade that has good aspect ratio and resolution
selected_cases = {}
for g in range(5):
    for cid in candidates[g]:
        p = img_dir / f"{cid}.png"
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None and img.shape[0] > 400 and img.shape[1] > 400:
                selected_cases[g] = (cid, p)
                print(f"Selected Grade {g}: {cid} (shape: {img.shape})")
                break

# Setup output dirs
vera_out_dir = Path("sample_data/vera_outputs")
vera_out_dir.mkdir(parents=True, exist_ok=True)
images_out_dir = Path("sample_data/images")
images_out_dir.mkdir(parents=True, exist_ok=True)

# Load Production ResNet-50 4-channel model
ckpt_path = Path("checkpoints/production_model/best_model.pth")
model = build_model(model_type="vessel_aware", backbone="resnet50", num_classes=5, pretrained=False)
if ckpt_path.exists():
    state_dict, meta = load_checkpoint_safe(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    print("Loaded production ResNet-50 4-channel weights successfully!")
model.eval()

segmenter = VesselSegmenter(cache_dir="outputs/vessel_cache")
cam_engine = GradCAMPlusPlus(model)

case_reports = {}

for g in range(5):
    cid, path = selected_cases[g]
    raw_bgr = cv2.imread(str(path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to standard 512x512 for paper figures
    raw_512 = cv2.resize(raw_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Save standard sample image
    sample_save_path = images_out_dir / f"fundus_{g}_000.png"
    cv2.imwrite(str(sample_save_path), cv2.cvtColor(raw_512, cv2.COLOR_RGB2BGR))
    
    # Preprocessing (Stream A)
    preproc_rgb = preprocess_fundus(
        raw_512,
        target_size=(512, 512),
        apply_crop=True,
        apply_enhancement=True,
        enhancement_method="combined"
    )
    cv2.imwrite(str(vera_out_dir / f"{g}_enhanced.png"), cv2.cvtColor(preproc_rgb, cv2.COLOR_RGB2BGR))
    
    # Vessel extraction (Stream B)
    vessel_map = segmenter.predict(preproc_rgb, target_size=(512, 512))
    vessel_u8 = (np.clip(vessel_map, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(vera_out_dir / f"{g}_vessel.png"), vessel_u8)
    
    # Prepare 4-channel tensor for model inference (224x224 for backbone)
    p_224 = cv2.resize(preproc_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    v_224 = cv2.resize(vessel_map, (224, 224), interpolation=cv2.INTER_AREA)
    
    rgb_norm = p_224.astype(np.float32) / 255.0
    for c in range(3):
        rgb_norm[:, :, c] = (rgb_norm[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    rgb_t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
    
    v_norm = (v_224.astype(np.float32) - VESSEL_MEAN[0]) / VESSEL_STD[0]
    v_t = torch.from_numpy(v_norm).unsqueeze(0).float()
    
    x_4ch = torch.cat([rgb_t, v_t], dim=0).unsqueeze(0)
    
    # Generate Grad-CAM++
    cam_map, pred_class, conf, all_probs = cam_engine.generate_cam(x_4ch, target_class=g)
    # Resize cam_map to 512x512
    cam_512 = cv2.resize(cam_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    np.save(str(vera_out_dir / f"{g}_gradcam.npy"), cam_512)
    
    # Compute overlap score
    ov = compute_vessel_attention_overlap(cam_512, vessel_map)
    
    # Extract clinical biomarkers
    report = analyze_clinical_biomarkers(
        img_rgb=preproc_rgb,
        vessel_prob_map=vessel_map,
        pred_grade=g,
        confidence=float(conf)
    )
    case_reports[g] = {
        "id": cid,
        "pred_class": pred_class,
        "confidence": conf,
        "overlap": ov["overlap_score"],
        "report": report
    }
    print(f"Grade {g} ({cid}): Pred={pred_class}, Conf={conf:.3f}, Overlap={ov['overlap_score']:.3f}, HM={report.hemorrhage_count}, MA={report.microaneurysm_count}")

print("\nAll 5 real cases successfully processed and saved to sample_data/vera_outputs/ and sample_data/images/")
