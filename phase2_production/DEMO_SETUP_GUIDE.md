# 🚀 VERA Demo Setup Guide

**Goal:** Run VERA on your local machine to demonstrate DR detection with uploaded images.

This guide provides **two options**:
1. **Quick Demo** - Use ImageNet pre-trained weights (works immediately, lower accuracy)
2. **Full Demo** - Train on Google Colab first (better accuracy, requires setup)

---

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection (for downloading dependencies)
- (Optional) NVIDIA GPU for faster inference

---

## Option 1: Quick Demo (5 Minutes) ⚡

Use this if you want to see the system working immediately without training.

### Step 1: Install Dependencies

```bash
cd VERA/phase2_production

# Install required packages
pip install torch torchvision
pip install streamlit
pip install opencv-python
pip install segmentation-models-pytorch
pip install albumentations
pip install timm
pip install scikit-learn scikit-image
pip install pyyaml
pip install matplotlib seaborn
pip install pillow numpy pandas
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 2: Run Demo App

```bash
streamlit run demo_app.py
```

### Step 3: Upload Test Images

1. Browser will open automatically (usually http://localhost:8501)
2. Click "Browse files" to upload a fundus image
3. Click "🔍 Analyze Image"
4. View results!

**Note:** This uses ImageNet pre-trained weights without DR-specific training, so accuracy will be limited (~60-70%). For production-quality results, use Option 2.

---

## Option 2: Full Demo with Trained Models (4-6 Hours) 🎯

This option trains models on Google Colab's free GPU for better accuracy.

### Step 1: Upload to Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/Train_VERA_on_Colab.ipynb`
3. Or directly: File → Open notebook → Upload → select the file

### Step 2: Configure Colab Runtime

1. Runtime → Change runtime type
2. Select **GPU** (T4 recommended)
3. Click **Save**

### Step 3: Get Kaggle API Credentials

To download datasets, you need Kaggle API access:

1. Go to [kaggle.com](https://www.kaggle.com/)
2. Sign in (create account if needed - it's free)
3. Go to Account → API → Create New API Token
4. Download `kaggle.json` file
5. Keep it ready for Step 4

### Step 4: Run Training Notebook

Follow the notebook cells in order:

1. **Cell 1-2**: Check GPU and clone repository
2. **Cell 3**: Install dependencies (~2 minutes)
3. **Cell 4-5**: Upload kaggle.json and download APTOS dataset (~15 minutes)
4. **Cell 6**: Train vessel segmenter (~30 minutes)
5. **Cell 7**: Cache vessel maps (~15 minutes)
6. **Cell 8**: Train DR classifier (~3-4 hours) ⏰
7. **Cell 9**: Evaluate model
8. **Cell 10-11**: Package and download models

**Total time:** ~4-6 hours (most is automated, you can leave it running)

### Step 5: Download Trained Models

After training completes:

1. Run the final cell to download `trained_models.zip`
2. Extract the zip file on your local machine
3. Copy files to your project:

```bash
# On your local machine
cd VERA/phase2_production

# Create checkpoints directory if it doesn't exist
mkdir -p models/checkpoints

# Copy trained models (adjust paths as needed)
cp path/to/trained_models/best_model.pth models/checkpoints/
cp path/to/trained_models/vessel_segmenter.pth models/checkpoints/
```

### Step 6: Run Full Demo

```bash
# Make sure you're in the right directory
cd VERA/phase2_production

# Run the demo app
streamlit run demo_app.py
```

The app will automatically detect trained models and use them for predictions!

---

## 🧪 Testing the Demo

### Sample Images

You can test with:

1. **Your own fundus images** - Any retinal photographs
2. **Public datasets:**
   - [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
   - [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
   - [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

### Expected Results

With **trained models**, you should see:
- **Accuracy:** ~80-85%
- **QWK:** ~0.85-0.90
- **Processing time:** 2-5 seconds per image

With **ImageNet weights only**:
- **Accuracy:** ~60-70%
- **QWK:** ~0.60-0.70
- **Processing time:** 2-5 seconds per image

---

## 📊 Demo Features

The demo app includes:

### 1. Image Upload & Preprocessing
- Upload PNG, JPG, JPEG files
- Automatic preprocessing (Ben Graham + CLAHE)
- Circular crop for field of view

### 2. Vessel Segmentation
- Real-time U-Net inference
- Visual display of detected vessels
- Used for vessel-aware prediction

### 3. DR Grading
- 5-class severity prediction (0-4)
- Confidence scores
- Probability distribution chart

### 4. Clinical Recommendations
- Grade-specific guidance
- Referral urgency
- Follow-up timeline

### 5. Results Display
- Original vs processed images
- Vessel segmentation overlay
- Probability bar chart
- Color-coded severity levels

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory" (if using GPU)

**Solution:**
```python
# Edit demo_app.py, change batch size or use CPU
device = 'cpu'  # Force CPU usage
```

### Issue: "Models not loading"

**Check:**
1. Models exist in `models/checkpoints/`
2. Files are named correctly:
   - `best_model.pth`
   - `vessel_segmenter.pth`
3. Files are not corrupted (re-download if needed)

### Issue: "Streamlit not opening browser"

**Solution:**
```bash
# Manually open browser to:
http://localhost:8501
```

### Issue: "Low accuracy with ImageNet weights"

**Expected behavior** - Train models on Colab for better results.

---

## 💡 Tips for Demonstration

### 1. Prepare Test Images
- Have 5-10 sample images ready
- Include different severity grades (0-4)
- Use high-quality fundus photographs

### 2. Explain the Process
- Show preprocessing (Ben Graham, CLAHE)
- Highlight vessel segmentation
- Explain probability distribution

### 3. Discuss Limitations
- "This is a demo/research tool"
- "Always consult ophthalmologist"
- "Trained on specific datasets"

### 4. Showcase Features
- Fast inference (2-5 seconds)
- Visual explanations
- Clinical recommendations
- Vessel-aware architecture

---

## 🎯 Demo Script Example

**For presenting to someone:**

1. **Introduction (1 min)**
   - "VERA is an AI system for detecting diabetic retinopathy"
   - "It analyzes fundus images and grades severity from 0-4"
   - "Uses vessel-aware neural networks for better accuracy"

2. **Upload Image (30 sec)**
   - "Let me upload a fundus photograph..."
   - [Upload image]

3. **Show Processing (1 min)**
   - "The system preprocesses the image with Ben Graham normalization"
   - "It segments blood vessels using U-Net"
   - "Then analyzes both together for DR grading"

4. **Explain Results (2 min)**
   - "The model predicts Grade X with Y% confidence"
   - "Here's the probability distribution across all grades"
   - "Notice how it focuses on the vessels"
   - "Clinical recommendation: [show recommendation]"

5. **Q&A (variable)**
   - Be ready to explain architecture, training data, accuracy

**Total: ~5 minutes per image**

---

## 📱 Alternative: Web-Based Demo

If you want to deploy for remote access:

### Option A: Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository
4. Deploy `demo_app.py`
5. Share public URL

### Option B: ngrok (Temporary)

```bash
# Install ngrok
pip install pyngrok

# Run Streamlit
streamlit run demo_app.py

# In another terminal, expose port
ngrok http 8501

# Share the ngrok URL (valid for 2 hours on free tier)
```

### Option C: Local Network

```bash
# Run Streamlit with network access
streamlit run demo_app.py --server.address 0.0.0.0

# Access from other devices on same network
http://YOUR_IP:8501
```

---

## 📊 Performance Optimization

### For Faster Inference

1. **Use GPU if available:**
   ```python
   # In demo_app.py, check:
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. **Reduce image size (if needed):**
   ```python
   # In configs/config.yaml
   image_size: 384  # Instead of 512
   ```

3. **Use lighter backbone:**
   ```python
   # In configs/config.yaml
   backbone: 'resnet34'  # Instead of resnet50
   ```

### For Better Accuracy

1. **Train on more data:**
   - Add EyePACS dataset to training
   - Use data augmentation

2. **Train longer:**
   - Increase epochs to 100
   - Use cosine annealing scheduler

3. **Use better backbone:**
   - Try EfficientNet-B4
   - Use attention-gated fusion

---

## 🎓 Next Steps

After successful demo:

1. **Collect feedback** - What features are most useful?
2. **Test with real data** - Partner with clinics for validation
3. **Improve model** - Add more training data
4. **Add features:**
   - Multi-disease detection (AMD, glaucoma)
   - Image quality assessment
   - Lesion segmentation
   - Report generation (PDF export)
5. **Deploy properly** - Consider FastAPI + Docker for production

---

## 📧 Support

If you encounter issues:

1. **Check logs:**
   ```bash
   streamlit run demo_app.py --logger.level=debug
   ```

2. **Review documentation:**
   - `API_REFERENCE.md` for code details
   - `USAGE_GUIDE.md` for comprehensive guide

3. **Common issues:**
   - Dependencies: Re-run `pip install -r requirements.txt`
   - Models: Verify files in `models/checkpoints/`
   - Images: Ensure proper format (PNG/JPG)

---

## ✅ Checklist Before Demo

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models downloaded (trained or ImageNet)
- [ ] Test images prepared (5-10 samples)
- [ ] Demo app runs successfully (`streamlit run demo_app.py`)
- [ ] Browser opens to http://localhost:8501
- [ ] Test 1-2 images to verify functionality
- [ ] Prepared talking points for presentation

---

## 🎉 You're Ready!

Your VERA demo is now set up and ready to demonstrate!

**Quick start command:**
```bash
cd VERA/phase2_production
streamlit run demo_app.py
```

**Have fun demonstrating diabetic retinopathy detection!** 🩺✨

---

*For questions or issues, refer to USAGE_GUIDE.md or API_REFERENCE.md*
