"""
Unit Tests for VERA Phase 2 Components:
1. Ben Graham preprocessing & enhancement modes.
2. Multi-dataset DataFrame standardization.
3. Dual-Branch & Attention-Gated fusion classifiers.
4. Differentiable Quadratic Weighted Kappa (QWK) & Focal Loss functions.
5. Grad-CAM++ & Quantitative Vessel-Attention Overlap Metric.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from src.preprocessing import apply_ben_graham, preprocess_fundus
from src.dataset import standardize_dataset_df
from src.models import build_model, DualBranchDRClassifier, SpatialAttentionGatedDRClassifier
from src.train import QuadraticWeightedKappaLoss, FocalLoss, HybridCEQWKLoss
from src.explainability import GradCAMPlusPlus, compute_vessel_attention_overlap


class TestPhase2Pipeline(unittest.TestCase):
    def setUp(self):
        # Create synthetic test fundus image (H, W, 3)
        self.test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add illuminated circle
        cv2_circle = np.zeros((224, 224), dtype=np.uint8)
        import cv2
        cv2.circle(cv2_circle, (112, 112), 90, 255, -1)
        self.test_img[cv2_circle > 0] = [180, 100, 50]
        
    def test_ben_graham_preprocessing(self):
        bg_img = apply_ben_graham(self.test_img, sigma=10.0)
        self.assertEqual(bg_img.shape, (224, 224, 3))
        self.assertEqual(bg_img.dtype, np.uint8)
        
        # Test integrated switchboard in preprocess_fundus
        processed_clahe = preprocess_fundus(self.test_img, enhancement_method="clahe")
        processed_bg = preprocess_fundus(self.test_img, enhancement_method="ben_graham")
        processed_comb = preprocess_fundus(self.test_img, enhancement_method="combined")
        self.assertEqual(processed_clahe.shape, (224, 224, 3))
        self.assertEqual(processed_bg.shape, (224, 224, 3))
        self.assertEqual(processed_comb.shape, (224, 224, 3))

    def test_dataset_standardization(self):
        # Test APTOS format
        df_aptos = pd.DataFrame({"id_code": ["img1", "img2"], "diagnosis": [0, 3]})
        std_aptos = standardize_dataset_df(df_aptos)
        self.assertIn("image_id", std_aptos.columns)
        self.assertIn("diagnosis", std_aptos.columns)
        self.assertIn("referable", std_aptos.columns)
        self.assertEqual(std_aptos["referable"].tolist(), [0, 1])

        # Test EyePACS format
        df_eyepacs = pd.DataFrame({"image": ["ep1", "ep2"], "level": [1, 4]})
        std_ep = standardize_dataset_df(df_eyepacs)
        self.assertEqual(std_ep["referable"].tolist(), [0, 1])

    def test_dual_branch_forward_pass(self):
        model = build_model(model_type="dual_branch", backbone="resnet18", pretrained=False)
        self.assertIsInstance(model, DualBranchDRClassifier)
        dummy_input = torch.randn(2, 4, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 5))

    def test_attention_gated_forward_pass(self):
        model = build_model(model_type="attention_gated", backbone="resnet18", pretrained=False)
        self.assertIsInstance(model, SpatialAttentionGatedDRClassifier)
        dummy_input = torch.randn(2, 4, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 5))

    def test_qwk_and_focal_loss_backward(self):
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.tensor([0, 2, 4, 1])

        # QWK Loss backward
        qwk_loss_fn = QuadraticWeightedKappaLoss(num_classes=5)
        loss_qwk = qwk_loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(loss_qwk))
        loss_qwk.backward(retain_graph=True)
        self.assertIsNotNone(logits.grad)

        # Focal Loss backward
        logits.grad.zero_()
        focal_loss_fn = FocalLoss(gamma=2.0)
        loss_focal = focal_loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(loss_focal))
        loss_focal.backward()
        self.assertIsNotNone(logits.grad)

    def test_gradcam_plus_plus_and_overlap_metric(self):
        model = build_model(model_type="vessel_aware", backbone="resnet18", pretrained=False)
        cam_engine = GradCAMPlusPlus(model)
        dummy_input = torch.randn(1, 4, 224, 224)
        
        cam_map, pred_class, conf, probs = cam_engine.generate_cam(dummy_input)
        self.assertEqual(cam_map.shape, (224, 224))
        self.assertTrue(0.0 <= cam_map.min() and cam_map.max() <= 1.0)
        self.assertTrue(0 <= pred_class <= 4)

        # Overlap metric
        dummy_vessel = np.random.uniform(0.0, 1.0, (224, 224)).astype(np.float32)
        overlap = compute_vessel_attention_overlap(cam_map, dummy_vessel)
        self.assertIn("overlap_score", overlap)
        self.assertTrue(0.0 <= overlap["overlap_score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
