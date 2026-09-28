"""
Utilities module:
1. Synthetic / Sample Fundus Data Generator for testing and standalone demonstration.
2. Multi-panel Visualization Helper for the Explainability MVP Demo.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ICDR_CLASS_NAMES = [
    "0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"
]


def generate_synthetic_fundus(
    label: int,
    size: Tuple[int, int] = (512, 512),
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a realistic synthetic retinal fundus image with anatomical structures
    (retinal disc, vascular tree, background pigmentation) and DR lesions corresponding
    to the given severity label (0: None, 1: Microaneurysms, 2: Hemorrhages/Exudates,
    3: Severe hemorrhages/IRMA, 4: Neovascularization).
    """
    if seed is not None:
        np.random.seed(seed)
        
    h, w = size
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.45)
    
    # 1. Base retinal background (reddish-orange with natural illumination gradient)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create circular mask
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    fundus_mask = dist_from_center <= radius
    
    # Retinal background gradient
    r_base = np.clip(180 - 40 * (dist_from_center / radius), 120, 220)
    g_base = np.clip(80 - 30 * (dist_from_center / radius), 40, 110)
    b_base = np.clip(20 - 10 * (dist_from_center / radius), 10, 35)
    
    # Texture noise
    noise = np.random.normal(0, 4, (h, w))
    img[:, :, 0] = np.clip((r_base + noise) * fundus_mask, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip((g_base + noise) * fundus_mask, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip((b_base + noise) * fundus_mask, 0, 255).astype(np.uint8)
    
    # 2. Optic Disc (bright yellowish oval)
    disc_x = center[0] - int(radius * 0.45)
    disc_y = center[1] + int(radius * 0.05)
    cv2.ellipse(
        img,
        (disc_x, disc_y),
        (int(radius * 0.15), int(radius * 0.18)),
        0, 0, 360,
        (240, 210, 140),
        -1
    )
    
    # 3. Vascular Tree (branching dark red/brown vessels from optic disc)
    num_main_branches = 6
    for b in range(num_main_branches):
        angle = np.random.uniform(-np.pi * 0.8, np.pi * 0.8)
        cur_x, cur_y = float(disc_x), float(disc_y)
        thickness = np.random.randint(3, 6)
        
        steps = int(radius * 0.75)
        for s in range(steps):
            angle += np.random.normal(0, 0.08)
            next_x = cur_x + np.cos(angle) * 1.5
            next_y = cur_y + np.sin(angle) * 1.5
            
            d = np.sqrt((next_x - center[0])**2 + (next_y - center[1])**2)
            if d < radius * 0.95:
                cv2.line(
                    img,
                    (int(cur_x), int(cur_y)),
                    (int(next_x), int(next_y)),
                    (90, 20, 10),
                    max(1, int(thickness * (1.0 - s / steps)))
                )
                
                # Random side branches
                if s % 25 == 0 and s > 15:
                    sub_angle = angle + np.random.choice([-0.6, 0.6])
                    sub_x, sub_y = cur_x, cur_y
                    for sub_s in range(int(radius * 0.25)):
                        sub_angle += np.random.normal(0, 0.06)
                        sub_nx = sub_x + np.cos(sub_angle) * 1.2
                        sub_ny = sub_y + np.sin(sub_angle) * 1.2
                        if np.sqrt((sub_nx - center[0])**2 + (sub_ny - center[1])**2) < radius * 0.95:
                            cv2.line(
                                img,
                                (int(sub_x), int(sub_y)),
                                (int(sub_nx), int(sub_ny)),
                                (100, 25, 12),
                                max(1, int(thickness * 0.5))
                            )
                        sub_x, sub_y = sub_nx, sub_ny
                        
            cur_x, cur_y = next_x, next_y
            
    # 4. Inject DR Pathological Lesions according to ICDR Grade
    if label >= 1:
        # Grade 1 (Mild): Small red microaneurysms (dots)
        num_ma = np.random.randint(5, 15) if label == 1 else np.random.randint(15, 40)
        for _ in range(num_ma):
            rand_r = np.random.uniform(radius * 0.2, radius * 0.8)
            rand_theta = np.random.uniform(0, 2 * np.pi)
            lx = int(center[0] + rand_r * np.cos(rand_theta))
            ly = int(center[1] + rand_r * np.sin(rand_theta))
            cv2.circle(img, (lx, ly), np.random.randint(1, 3), (80, 10, 5), -1)
            
    if label >= 2:
        # Grade 2 (Moderate): Dot-and-blot hemorrhages + Hard exudates (bright yellow wax)
        num_hem = np.random.randint(8, 20)
        for _ in range(num_hem):
            rand_r = np.random.uniform(radius * 0.25, radius * 0.75)
            rand_theta = np.random.uniform(0, 2 * np.pi)
            lx = int(center[0] + rand_r * np.cos(rand_theta))
            ly = int(center[1] + rand_r * np.sin(rand_theta))
            cv2.ellipse(
                img,
                (lx, ly),
                (np.random.randint(3, 8), np.random.randint(2, 6)),
                np.random.randint(0, 180), 0, 360,
                (70, 8, 5), -1
            )
            
        num_exudates = np.random.randint(5, 15)
        for _ in range(num_exudates):
            rand_r = np.random.uniform(radius * 0.2, radius * 0.7)
            rand_theta = np.random.uniform(0, 2 * np.pi)
            lx = int(center[0] + rand_r * np.cos(rand_theta))
            ly = int(center[1] + rand_r * np.sin(rand_theta))
            cv2.circle(img, (lx, ly), np.random.randint(2, 6), (245, 235, 120), -1)
            
    if label >= 3:
        # Grade 3 (Severe): Extensive flame hemorrhages, venous beading, cotton wool spots
        num_large_hem = np.random.randint(15, 30)
        for _ in range(num_large_hem):
            rand_r = np.random.uniform(radius * 0.15, radius * 0.85)
            rand_theta = np.random.uniform(0, 2 * np.pi)
            lx = int(center[0] + rand_r * np.cos(rand_theta))
            ly = int(center[1] + rand_r * np.sin(rand_theta))
            cv2.ellipse(
                img,
                (lx, ly),
                (np.random.randint(6, 16), np.random.randint(4, 10)),
                np.random.randint(0, 180), 0, 360,
                (60, 5, 5), -1
            )
        # Cotton wool spots (soft white patches)
        for _ in range(np.random.randint(3, 8)):
            rand_r = np.random.uniform(radius * 0.2, radius * 0.6)
            rand_theta = np.random.uniform(0, 2 * np.pi)
            lx = int(center[0] + rand_r * np.cos(rand_theta))
            ly = int(center[1] + rand_r * np.sin(rand_theta))
            cv2.circle(img, (lx, ly), np.random.randint(5, 12), (230, 230, 210), -1)
            
    if label == 4:
        # Grade 4 (Proliferative): Fronds of fragile neovascular vessels & preretinal hemorrhage
        num_neo = np.random.randint(4, 8)
        for _ in range(num_neo):
            nx, ny = center[0] + np.random.randint(-50, 50), center[1] + np.random.randint(-50, 50)
            for _ in range(12):
                end_x = nx + np.random.randint(-25, 25)
                end_y = ny + np.random.randint(-25, 25)
                cv2.line(img, (nx, ny), (end_x, end_y), (110, 15, 10), np.random.randint(1, 3))
                
        # Large preretinal hemorrhage boat-shaped
        cv2.ellipse(
            img,
            (center[0] + int(radius * 0.3), center[1] - int(radius * 0.2)),
            (int(radius * 0.2), int(radius * 0.08)),
            -20, 0, 360,
            (50, 2, 2), -1
        )
        
    # Re-apply mask to clean outer boundary
    img[~fundus_mask] = 0
    return img


def setup_sample_dataset(
    output_dir: Union[str, Path] = "sample_data",
    num_samples_per_class: int = 20
) -> Tuple[Path, pd.DataFrame]:
    """
    Creates a sample dataset directory structured as APTOS 2019 format:
    sample_data/
      images/
        <id_code>.png
      train.csv (id_code, diagnosis)
    """
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    idx = 0
    for grade in range(5):
        for s in range(num_samples_per_class):
            id_code = f"fundus_{grade}_{s:03d}"
            img = generate_synthetic_fundus(label=grade, seed=grade * 1000 + s)
            
            img_file = images_dir / f"{id_code}.png"
            cv2.imwrite(str(img_file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            data.append({
                "id_code": id_code,
                "diagnosis": grade
            })
            idx += 1
            
    df = pd.DataFrame(data)
    csv_file = out_path / "train.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"Generated sample dataset: {len(df)} images at {out_path} with 5 balanced classes.")
    return out_path, df


generate_synthetic_fundus_dataset = setup_sample_dataset



def plot_prediction_explanation(
    original_rgb: np.ndarray,
    vessel_map: np.ndarray,
    gradcam_overlay: np.ndarray,
    probs: np.ndarray,
    true_grade: Optional[int] = None,
    pred_grade: Optional[int] = None,
    title: str = "Explainable DR Severity Prediction",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (18, 4.5)
) -> plt.Figure:
    """
    Generates the MVP 4-panel deliverable:
    [Original Fundus] -> [Vessel Probability Map] -> [Grad-CAM Overlay] -> [Class Probabilities Bar Chart]
    """
    class_names = ["0: No DR", "1: Mild", "2: Moderate", "3: Severe", "4: Proliferative"]
    if pred_grade is None:
        pred_grade = int(np.argmax(probs))
        
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Panel 1: Original Preprocessed Fundus
    axes[0].imshow(original_rgb)
    axes[0].set_title("1. Preprocessed Fundus (RGB)", fontsize=11, weight="bold")
    axes[0].axis("off")
    
    # Panel 2: Retinal Vasculature Map
    im2 = axes[1].imshow(vessel_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("2. Vessel Probability Map", fontsize=11, weight="bold")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Panel 3: Grad-CAM Overlay
    axes[2].imshow(gradcam_overlay)
    axes[2].set_title("3. Grad-CAM Saliency Overlay", fontsize=11, weight="bold")
    axes[2].axis("off")
    
    # Panel 4: Class Probabilities Bar Chart
    colors = ["#3498db" if i != pred_grade else "#e74c3c" for i in range(len(class_names))]
    bars = axes[3].barh(class_names, probs, color=colors, height=0.6)
    axes[3].set_xlim(0, 1.0)
    axes[3].set_xlabel("Confidence Score", fontsize=10)
    
    # Add probability percentage labels on bars
    for bar in bars:
        w = bar.get_width()
        axes[3].text(
            w + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{w*100:.1f}%",
            va="center",
            fontsize=9,
            weight="bold" if bar.get_facecolor()[:3] == (0.9058823529411765, 0.2980392156862745, 0.23529411764705882) else "normal"
        )
        
    truth_text = f" | True: Grade {true_grade}" if true_grade is not None else ""
    axes[3].set_title(f"4. Prediction: Grade {pred_grade} ({probs[pred_grade]*100:.1f}%){truth_text}", fontsize=11, weight="bold")
    axes[3].grid(axis="x", linestyle="--", alpha=0.5)
    axes[3].invert_yaxis()
    
    plt.suptitle(title, fontsize=13, weight="bold", y=1.03)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    return fig
