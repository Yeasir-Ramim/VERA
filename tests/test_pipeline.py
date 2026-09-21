"""
Sanity and unit tests for the DR Detection Phase 1 MVP pipeline.
Tests all 5 chunks:
1. Preprocessing & Data Pipeline
2. Baseline CNN (3-channel)
3. Vessel Segmentation Extraction & Caching
4. Vessel-Aware CNN (4-channel conv1 adaptation)
5. Grad-CAM Explainability & Visualization
"""

import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import crop_fundus_circle, apply_clahe, preprocess_fundus
from src.vessel_segmentation import extract_vessel_map_multiscale, VesselSegmenter, UNetVesselSegmenter
from src.dataset import APTOSDataset, compute_class_weights
from src.models import build_model, adapt_first_conv_to_4channels
from src.explainability import GradCAM, overlay_cam_on_image
from src.utils import generate_synthetic_fundus, setup_sample_dataset


class TestDRPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate a test fundus image
        cls.test_img = generate_synthetic_fundus(label=2, size=(256, 256), seed=42)
        
    def test_1_preprocessing(self):
        """Verify Crop, CLAHE, and full preprocessing pipeline."""
        # 1. Test crop
        cropped = crop_fundus_circle(self.test_img)
        self.assertEqual(cropped.ndim, 3)
        self.assertGreater(cropped.shape[0], 0)
        self.assertGreater(cropped.shape[1], 0)
        
        # 2. Test CLAHE
        clahe_img = apply_clahe(self.test_img)
        self.assertEqual(clahe_img.shape, self.test_img.shape)
        self.assertEqual(clahe_img.dtype, np.uint8)
        
        # 3. Test combined preprocessing
        target_size = (224, 224)
        prep = preprocess_fundus(self.test_img, target_size=target_size)
        self.assertEqual(prep.shape, (224, 224, 3))
        self.assertEqual(prep.dtype, np.uint8)

    def test_2_vessel_segmentation(self):
        """Verify vessel extraction and caching."""
        prep = preprocess_fundus(self.test_img, target_size=(224, 224))
        vessel_map = extract_vessel_map_multiscale(prep, target_size=(224, 224))
        
        self.assertEqual(vessel_map.shape, (224, 224))
        self.assertGreaterEqual(vessel_map.min(), 0.0)
        self.assertLessEqual(vessel_map.max(), 1.0)
        self.assertEqual(vessel_map.dtype, np.float32)
        
        # Test Caching
        cache_dir = Path("/tmp/test_vessel_cache")
        segmenter = VesselSegmenter(cache_dir=cache_dir)
        cached_map = segmenter.get_vessel_map(prep, image_id="test_001", target_size=(224, 224))
        self.assertTrue((cache_dir / "test_001_vessel.png").exists())
        self.assertEqual(cached_map.shape, (224, 224))

    def test_3_model_adaptation(self):
        """Verify 4-channel conv1 weight adaptation and weight preservation."""
        orig_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        adapted_conv = adapt_first_conv_to_4channels(orig_conv)
        
        self.assertEqual(adapted_conv.in_channels, 4)
        self.assertEqual(adapted_conv.out_channels, 64)
        
        # Check RGB weight preservation
        self.assertTrue(torch.allclose(adapted_conv.weight[:, 0:3, :, :], orig_conv.weight.data))
        
        # Check 4th channel is the mean of RGB channels
        expected_vessel_weight = torch.mean(orig_conv.weight.data, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(adapted_conv.weight[:, 3:4, :, :], expected_vessel_weight))

    def test_4_models_forward(self):
        """Verify forward pass on both 3-channel baseline and 4-channel vessel-aware models."""
        batch_size = 2
        
        # 1. Baseline 3-channel
        model_3ch = build_model(model_type="baseline", backbone="resnet18", num_classes=5, pretrained=False)
        x_3ch = torch.randn(batch_size, 3, 224, 224)
        out_3ch = model_3ch(x_3ch)
        self.assertEqual(out_3ch.shape, (batch_size, 5))
        
        # 2. Vessel-Aware 4-channel
        model_4ch = build_model(model_type="vessel_aware", backbone="resnet18", num_classes=5, pretrained=False)
        x_4ch = torch.randn(batch_size, 4, 224, 224)
        out_4ch = model_4ch(x_4ch)
        self.assertEqual(out_4ch.shape, (batch_size, 5))

    def test_5_gradcam_explainability(self):
        """Verify Grad-CAM activation mapping and overlay blending."""
        model_4ch = build_model(model_type="vessel_aware", backbone="resnet18", num_classes=5, pretrained=False)
        gradcam = GradCAM(model_4ch)
        
        input_tensor = torch.randn(1, 4, 224, 224)
        cam_map, pred_class, confidence, probs = gradcam.generate_cam(input_tensor)
        
        self.assertEqual(cam_map.shape, (224, 224))
        self.assertGreaterEqual(cam_map.min(), 0.0)
        self.assertLessEqual(cam_map.max(), 1.0)
        self.assertIn(pred_class, list(range(5)))
        self.assertEqual(len(probs), 5)
        
        # Test overlay blending
        rgb_dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        overlay = overlay_cam_on_image(rgb_dummy, cam_map)
        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
