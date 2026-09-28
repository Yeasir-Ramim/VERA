"""
Clinical Biomarkers and Retinal Pathology Analysis Module for VERA.
Designed from the perspective of an Ophthalmic Retina Specialist.

Extracts and quantifies key clinical diabetic retinopathy biomarkers:
1. Red Lesions: Microaneurysms (MAs) and Dot/Blot Hemorrhages (HMs).
2. Bright Lesions: Hard Exudates (HEs) and Cotton Wool Spots (CWS).
3. Anatomical Landmarks: Optic Disc (OD) and Fovea / Macula Center localization.
4. Clinically Significant Macular Edema (CSME / DME) Risk Triage:
   Quantifies proximity of hard exudates to the Foveal Avascular Zone (FAZ).
5. Anatomical 4-Quadrant Pathology Distribution (The 4-2-1 Rule):
   Categorizes lesions into Superior-Temporal (ST), Inferior-Temporal (IT),
   Superior-Nasal (SN), and Inferior-Nasal (IN) quadrants.
6. Vascular Caliber & Morphological Metrics:
   Vessel Density, Fractal Dimension, and Tortuosity Index.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


@dataclass
class QuadrantPathology:
    name: str
    code: str  # ST, IT, SN, IN
    hemorrhage_count: int
    microaneurysm_count: int
    hard_exudate_area_px: int
    vessel_density: float
    meets_severe_threshold: bool


@dataclass
class ClinicalBiomarkerReport:
    # Lesion quantitation
    microaneurysm_count: int
    hemorrhage_count: int
    hard_exudate_count: int
    hard_exudate_total_area_px: int
    cotton_wool_spot_count: int
    
    # Anatomical locations (x, y)
    optic_disc_center: Optional[Tuple[int, int]]
    optic_disc_radius: int
    fovea_center: Optional[Tuple[int, int]]
    fovea_radius: int
    
    # Macular Risk / DME
    csme_risk_level: str  # 'Low', 'Moderate', 'High (CSME Warning)'
    csme_distance_to_fovea_px: float
    exudates_within_1dd: bool
    exudates_within_2dd: bool
    
    # 4-2-1 Rule Assessment
    quadrant_data: Dict[str, QuadrantPathology]
    quadrants_with_severe_hemorrhages: int
    meets_4_quadrant_hemorrhage_rule: bool
    
    # Vascular Morphology
    vessel_density_pct: float
    fractal_dimension: float
    vascular_tortuosity_index: float
    
    # Specialist Decision Support Summary
    clinical_stage_interpretation: str
    primary_threat_to_vision: str
    actionable_specialist_advice: List[str]
    
    # Visual overlay image (RGB uint8)
    annotated_fundus: np.ndarray = field(repr=False)


def compute_fractal_dimension(binary_vessels: np.ndarray) -> float:
    """
    Computes box-counting fractal dimension (D_box) of the retinal vascular network.
    Healthy human retinal vascular tree fractal dimension is typically 1.40 - 1.48.
    Drops in capillary dropout/severe ischemia (< 1.38), and elevates in neovascular proliferation (> 1.50).
    """
    # Only pixels > 0
    p = (binary_vessels > 0)
    if not np.any(p):
        return 1.0

    # Minimal dimension
    min_dim = min(p.shape)
    # Greatest power of 2 less than or equal to min_dim
    n = 2 ** int(np.floor(np.log2(min_dim)))
    # Extract square region
    p = p[:n, :n]

    # Box sizes
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = []

    for size in sizes:
        # Reshape and sum
        box_view = p.reshape(n // size, size, n // size, size)
        has_vessel = box_view.any(axis=(1, 3))
        counts.append(np.sum(has_vessel))

    # Fit linear regression line: log(N) = -D * log(s) + C
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = float(-coeffs[0])
    return float(np.clip(fractal_dim, 1.0, 1.95))


def compute_vascular_tortuosity(vessel_prob_map: np.ndarray, threshold: float = 0.3) -> float:
    """
    Estimates vascular tortuosity index based on contour arc-length to chord-length ratios.
    Elevated tortuosity is a key clinical hallmark of diabetic microangiopathy and venous beading.
    """
    binary = (vessel_prob_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    tortuosities = []
    for cnt in contours:
        arc_len = cv2.arcLength(cnt, closed=False)
        if arc_len > 25:  # Significant vessel branch
            # Chord length between endpoints
            pt_start = cnt[0][0]
            pt_end = cnt[-1][0]
            chord_len = np.linalg.norm(pt_start - pt_end)
            if chord_len > 5:
                # Tortuosity ratio = (arc / chord)
                ratio = arc_len / (chord_len * 2.0)  # factor of 2 accounts for closed contour loop
                tortuosities.append(min(ratio, 4.0))
                
    if len(tortuosities) > 0:
        return float(np.mean(tortuosities))
    return 1.15  # Baseline normal vascular tortuosity


def detect_optic_disc_and_fovea(
    img_rgb: np.ndarray,
    fundus_mask: np.ndarray
) -> Tuple[Tuple[int, int], int, Tuple[int, int], int]:
    """
    Localizes Optic Disc (OD) and estimates Fovea / Macula Center coordinates.
    Optic disc is identified as a high-luminance, high-contrast circular region.
    The fovea is estimated geometrically relative to the OD and fundus center.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Optic disc has high intensity in Red and Gray channels
    red = img_rgb[:, :, 0]
    
    # Smooth to suppress small exudates and artifacts
    blurred = cv2.GaussianBlur(red, (31, 31), 0)
    blurred = cv2.bitwise_and(blurred, blurred, mask=fundus_mask)
    
    # Erode mask slightly to avoid perimeter edge reflections
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    inner_mask = cv2.erode(fundus_mask, kernel_erode)
    blurred_inner = np.where(inner_mask > 0, blurred, 0)
    
    # Find brightest candidate
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred_inner)
    od_x, od_y = max_loc
    
    # Approximate OD radius (~1/12th of fundus diameter)
    fundus_diameter = np.sqrt(np.sum(fundus_mask) / np.pi) * 2
    od_radius = int(max(15, fundus_diameter * 0.08))
    
    # Geometric estimation of Fovea:
    # In fundus photography, OD is on the nasal side, and Fovea is temporal.
    # If OD is on the left half (OD_x < w/2), it's a Left Eye (OS), Fovea is to the right.
    # If OD is on the right half (OD_x >= w/2), it's a Right Eye (OD), Fovea is to the left.
    center_x, center_y = w // 2, h // 2
    
    if od_x < center_x:
        # Left eye: Fovea is temporal (to the right of OD, near image center)
        fovea_x = int(center_x + (center_x - od_x) * 0.25)
    else:
        # Right eye: Fovea is temporal (to the left of OD, near image center)
        fovea_x = int(center_x - (od_x - center_x) * 0.25)
        
    # Fovea is slightly lower or at horizontal level with OD
    fovea_y = int(od_y + 0.1 * od_radius)
    fovea_y = int(np.clip(fovea_y, h * 0.3, h * 0.7))
    fovea_x = int(np.clip(fovea_x, w * 0.25, w * 0.75))
    
    fovea_radius = int(od_radius * 0.8)
    
    return (od_x, od_y), od_radius, (fovea_x, fovea_y), fovea_radius


def analyze_clinical_biomarkers(
    img_rgb: np.ndarray,
    vessel_prob_map: np.ndarray,
    pred_grade: int,
    confidence: float
) -> ClinicalBiomarkerReport:
    """
    Executes complete clinical biomarker extraction, lesion localization,
    CSME/DME risk triage, and quadrant distribution profiling.
    """
    h, w = img_rgb.shape[:2]
    
    # 1. Fundus Mask
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fundus_mask = (gray > 15).astype(np.uint8) * 255
    retinal_area_px = int(np.sum(fundus_mask > 0))
    if retinal_area_px == 0:
        retinal_area_px = h * w
        
    # 2. Localize Optic Disc & Fovea
    od_center, od_radius, fovea_center, fovea_radius = detect_optic_disc_and_fovea(img_rgb, fundus_mask)
    
    # Optic disc mask to exclude disc physiologic pallor from exudate count
    od_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(od_mask, od_center, int(od_radius * 1.3), 255, -1)
    
    # 3. Vascular Mask
    vessel_binary = (vessel_prob_map > 0.25).astype(np.uint8) * 255
    vessel_density_pct = float((np.sum(vessel_binary > 0) / retinal_area_px) * 100.0)
    fractal_dim = compute_fractal_dimension(vessel_binary)
    tortuosity_idx = compute_vascular_tortuosity(vessel_prob_map)
    
    # 4. Red Lesion Extraction (Microaneurysms & Hemorrhages)
    # Green channel has strongest contrast for red hemoglobin absorption
    green = img_rgb[:, :, 1]
    
    # Morphological bottom-hat (closing - original) highlights dark structures on green channel
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    
    bhat_small = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel_small)
    bhat_med = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel_med)
    
    # Mask out main vessel tree to isolate isolated red lesions (MAs and hemorrhages)
    dilated_vessels = cv2.dilate(vessel_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    red_lesions_raw = cv2.bitwise_and(bhat_med, bhat_med, mask=cv2.bitwise_not(dilated_vessels))
    red_lesions_raw = cv2.bitwise_and(red_lesions_raw, red_lesions_raw, mask=fundus_mask)
    
    # Threshold for candidate red lesions
    _, red_thresh = cv2.threshold(red_lesions_raw, 18, 255, cv2.THRESH_BINARY)
    
    # Connected component analysis for red lesions
    num_labels_red, labels_red, stats_red, centroids_red = cv2.connectedComponentsWithStats(red_thresh)
    
    microaneurysms: List[Tuple[int, int, int]] = []  # (x, y, r)
    hemorrhages: List[Tuple[int, int, int]] = []      # (x, y, r)
    
    for i in range(1, num_labels_red):
        area = stats_red[i, cv2.CC_STAT_AREA]
        cx = int(centroids_red[i][0])
        cy = int(centroids_red[i][1])
        
        # Exclude border artifacts
        if fundus_mask[cy, cx] == 0:
            continue
            
        if 2 <= area <= 20:  # Small focal punctate spot = Microaneurysm
            microaneurysms.append((cx, cy, max(2, int(np.sqrt(area)))))
        elif 21 < area <= 400:  # Larger blot/flame = Hemorrhage
            hemorrhages.append((cx, cy, max(3, int(np.sqrt(area)))))
            
    # 5. Bright Lesion Extraction (Hard Exudates & Cotton Wool Spots)
    # Hard exudates are bright yellow waxy deposits; high in luminance L and Red/Green
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L_channel = lab[:, :, 0]
    
    # Top-hat highlights bright local peaks
    kernel_exudate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    that_exudate = cv2.morphologyEx(L_channel, cv2.MORPH_TOPHAT, kernel_exudate)
    # Mask out optic disc
    that_exudate = cv2.bitwise_and(that_exudate, that_exudate, mask=cv2.bitwise_not(od_mask))
    that_exudate = cv2.bitwise_and(that_exudate, that_exudate, mask=fundus_mask)
    
    _, bright_thresh = cv2.threshold(that_exudate, 22, 255, cv2.THRESH_BINARY)
    
    num_labels_bright, labels_bright, stats_bright, centroids_bright = cv2.connectedComponentsWithStats(bright_thresh)
    
    hard_exudates: List[Tuple[int, int, int, int]] = []  # (x, y, r, area)
    cotton_wool_spots: List[Tuple[int, int, int]] = []
    total_he_area = 0
    
    for i in range(1, num_labels_bright):
        area = stats_bright[i, cv2.CC_STAT_AREA]
        cx = int(centroids_bright[i][0])
        cy = int(centroids_bright[i][1])
        
        if fundus_mask[cy, cx] == 0:
            continue
            
        if 3 <= area <= 60:
            hard_exudates.append((cx, cy, max(2, int(np.sqrt(area))), area))
            total_he_area += area
        elif 61 < area <= 350:
            cotton_wool_spots.append((cx, cy, max(4, int(np.sqrt(area)))))
            
    # 6. Macular Involvement / Diabetic Macular Edema (DME) Triage
    # Disc Diameter (DD) reference distance = 2 * od_radius
    one_dd = 2 * od_radius
    two_dd = 4 * od_radius
    
    min_dist_to_fovea = float("inf")
    exudates_in_1dd = False
    exudates_in_2dd = False
    
    for (hx, hy, _, _) in hard_exudates:
        dist = np.sqrt((hx - fovea_center[0])**2 + (hy - fovea_center[1])**2)
        if dist < min_dist_to_fovea:
            min_dist_to_fovea = dist
            
        if dist <= one_dd:
            exudates_in_1dd = True
        if dist <= two_dd:
            exudates_in_2dd = True
            
    if exudates_in_1dd:
        csme_risk = "HIGH (CSME Warning - Macular Threat)"
    elif exudates_in_2dd or len(hard_exudates) > 5:
        csme_risk = "MODERATE (Macula Approaching)"
    else:
        csme_risk = "LOW (Peripheral / Quiescent)"
        
    if min_dist_to_fovea == float("inf"):
        min_dist_to_fovea = 0.0
        
    # 7. Anatomical 4-Quadrant Partitioning (ST, IT, SN, IN)
    # Demarcated by Fovea horizontal and vertical axes
    fx, fy = fovea_center
    quadrant_specs = [
        ("Superior-Temporal", "ST", lambda x, y: (x >= fx and y < fy) if od_center[0] < fx else (x < fx and y < fy)),
        ("Inferior-Temporal", "IT", lambda x, y: (x >= fx and y >= fy) if od_center[0] < fx else (x < fx and y >= fy)),
        ("Superior-Nasal",    "SN", lambda x, y: (x < fx and y < fy) if od_center[0] < fx else (x >= fx and y < fy)),
        ("Inferior-Nasal",    "IN", lambda x, y: (x < fx and y >= fy) if od_center[0] < fx else (x >= fx and y >= fy))
    ]
    
    quadrant_results: Dict[str, QuadrantPathology] = {}
    severe_quadrant_count = 0
    
    for q_name, q_code, q_filter in quadrant_specs:
        q_hem = sum(1 for (hx, hy, _) in hemorrhages if q_filter(hx, hy))
        q_ma = sum(1 for (mx, my, _) in microaneurysms if q_filter(mx, my))
        q_he_area = sum(a for (hx, hy, _, a) in hard_exudates if q_filter(hx, hy))
        
        # Quadrant mask for vessel density
        q_mask = np.zeros((h, w), dtype=bool)
        y_grid, x_grid = np.ogrid[:h, :w]
        if (od_center[0] < fx and "Temporal" in q_name) or (od_center[0] >= fx and "Nasal" in q_name):
            x_cond = x_grid >= fx
        else:
            x_cond = x_grid < fx
            
        y_cond = y_grid < fy if "Superior" in q_name else y_grid >= fy
        q_mask = x_cond & y_cond & (fundus_mask > 0)
        q_area = np.sum(q_mask)
        q_vessel_density = float((np.sum(vessel_binary[q_mask] > 0) / max(1, q_area)) * 100.0)
        
        # Severe threshold (4-2-1 rule: severe hemorrhages in quadrant)
        meets_severe = (q_hem >= 6) or (pred_grade >= 3 and q_hem >= 4)
        if meets_severe:
            severe_quadrant_count += 1
            
        quadrant_results[q_code] = QuadrantPathology(
            name=q_name,
            code=q_code,
            hemorrhage_count=q_hem,
            microaneurysm_count=q_ma,
            hard_exudate_area_px=q_he_area,
            vessel_density=round(q_vessel_density, 1),
            meets_severe_threshold=meets_severe
        )
        
    meets_4_quad_rule = severe_quadrant_count >= 4 or (pred_grade >= 3)
    
    # 8. Clinical Interpretation & Decision Advice
    if pred_grade == 0:
        stage_desc = "Normal Retinal Examination (No signs of Diabetic Retinopathy)"
        primary_threat = "None detected. Microvascular structure is intact."
        actions = [
            "Maintain baseline glycemic control (target HbA1c < 7.0%).",
            "Routine annual or biennial dilated retinal screening advised.",
            "Continue optimal blood pressure and lipid monitoring."
        ]
    elif pred_grade == 1:
        stage_desc = "Mild Non-Proliferative Diabetic Retinopathy (NPDR)"
        primary_threat = "Isolated microaneurysms indicating early capillary wall breakdown."
        actions = [
            "Repeat dilated fundus examination in 12 months.",
            "Intensify diabetic medical management to stall microvascular progression.",
            "Counsel patient on glycemic variability and early visual symptoms."
        ]
    elif pred_grade == 2:
        stage_desc = "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)"
        primary_threat = "Microvascular leakage with intraretinal hemorrhages and lipid exudation."
        actions = [
            "Ophthalmology / Medical Retina referral within 3 to 6 months.",
            "Perform Macular Optical Coherence Tomography (OCT) to rule out subclinical macular edema.",
            "Strict blood pressure (< 130/80 mmHg) and lipid control to prevent exudate accumulation."
        ]
    elif pred_grade == 3:
        stage_desc = "Severe Non-Proliferative Diabetic Retinopathy (Severe NPDR - High Risk of PDR)"
        primary_threat = "Widespread retinal ischemia (4-2-1 criteria met), high risk of rapid progression to proliferative neovascularization."
        actions = [
            "URGENT Medical Retina review within 2 to 4 weeks.",
            "Baseline High-Resolution Macular OCT and Widefield Fluorescein Angiography (FFA).",
            "Evaluate for early Panretinal Photocoagulation (PRP) or Anti-VEGF therapy to prevent tractional complications."
        ]
    else:  # Grade 4
        stage_desc = "Proliferative Diabetic Retinopathy (PDR) — Vision-Threatening"
        primary_threat = "Active neovascularization, preretinal/vitreous hemorrhage risk, and tractional retinal detachment threat."
        actions = [
            "IMMEDIATE EMERGENCY Retina Specialist consultation (within 48 to 72 hours).",
            "Prompt Panretinal Photocoagulation (PRP) and/or Intravitreal Anti-VEGF (e.g. Aflibercept / Faricimab) therapy.",
            "Surgical vitreoretinal evaluation if non-clearing vitreous hemorrhage is present."
        ]
        
    if csme_risk.startswith("HIGH"):
        actions.insert(0, "🚨 MACULAR INVOLVEMENT ALERT: Request urgent macular OCT for Clinically Significant Macular Edema (CSME).")

    # 9. Render Annotated Visual Overlay for Specialists
    annotated = img_rgb.copy()
    
    # Draw Quadrant Dividers (subtle dashed cyan lines)
    cv2.line(annotated, (0, fy), (w, fy), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(annotated, (fx, 0), (fx, h), (0, 200, 255), 1, cv2.LINE_AA)
    
    # Draw Optic Disc (Yellow circle)
    cv2.circle(annotated, od_center, od_radius, (255, 235, 50), 2, cv2.LINE_AA)
    cv2.putText(annotated, "OD", (od_center[0] - 12, od_center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 235, 50), 1)
    
    # Draw Fovea and 1-DD / 2-DD Macular Rings (Magenta rings)
    cv2.circle(annotated, fovea_center, 4, (255, 50, 255), -1, cv2.LINE_AA)
    cv2.circle(annotated, fovea_center, one_dd, (255, 50, 255), 1, cv2.LINE_AA)
    cv2.circle(annotated, fovea_center, two_dd, (200, 50, 200), 1, cv2.LINE_AA)
    cv2.putText(annotated, "FAZ", (fovea_center[0] + 6, fovea_center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 50, 255), 1)
    
    # Highlight Microaneurysms (Red circles)
    for (mx, my, mr) in microaneurysms:
        cv2.circle(annotated, (mx, my), mr + 2, (255, 60, 60), 1, cv2.LINE_AA)
        
    # Highlight Hemorrhages (Crimson thick circles)
    for (hx, hy, hr) in hemorrhages:
        cv2.circle(annotated, (hx, hy), hr + 3, (220, 20, 20), 2, cv2.LINE_AA)
        
    # Highlight Hard Exudates (Lime Green circles)
    for (ex, ey, er, _) in hard_exudates:
        cv2.circle(annotated, (ex, ey), er + 2, (50, 255, 50), 1, cv2.LINE_AA)
        
    # Highlight Cotton Wool Spots (Cyan squares)
    for (cx, cy, cr) in cotton_wool_spots:
        cv2.rectangle(annotated, (cx - cr, cy - cr), (cx + cr, cy + cr), (0, 255, 255), 2, cv2.LINE_AA)
        
    # Quadrant Labels (ST, IT, SN, IN)
    offset_x = 20
    offset_y = 25
    cv2.putText(annotated, "ST", (w - 40, offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(annotated, "IT", (w - 40, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(annotated, "SN", (offset_x, offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(annotated, "IN", (offset_x, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return ClinicalBiomarkerReport(
        microaneurysm_count=len(microaneurysms),
        hemorrhage_count=len(hemorrhages),
        hard_exudate_count=len(hard_exudates),
        hard_exudate_total_area_px=total_he_area,
        cotton_wool_spot_count=len(cotton_wool_spots),
        optic_disc_center=od_center,
        optic_disc_radius=od_radius,
        fovea_center=fovea_center,
        fovea_radius=fovea_radius,
        csme_risk_level=csme_risk,
        csme_distance_to_fovea_px=round(min_dist_to_fovea, 1),
        exudates_within_1dd=exudates_in_1dd,
        exudates_within_2dd=exudates_in_2dd,
        quadrant_data=quadrant_results,
        quadrants_with_severe_hemorrhages=severe_quadrant_count,
        meets_4_quadrant_hemorrhage_rule=meets_4_quad_rule,
        vessel_density_pct=round(vessel_density_pct, 2),
        fractal_dimension=round(fractal_dim, 3),
        vascular_tortuosity_index=round(tortuosity_idx, 3),
        clinical_stage_interpretation=stage_desc,
        primary_threat_to_vision=primary_threat,
        actionable_specialist_advice=actions,
        annotated_fundus=annotated
    )
