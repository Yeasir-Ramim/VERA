@echo off
REM VERA Production Training Script - Automatic Training on Real Datasets
REM This script trains multiple models with different configurations

echo ================================================================================
echo VERA Production Training Pipeline
echo ================================================================================
echo.

REM Check if data exists
if not exist "data\raw\aptos2019\train.csv" (
    echo ERROR: APTOS dataset not found!
    echo.
    echo Please download the dataset first:
    echo 1. See DOWNLOAD_DATASETS.md for instructions
    echo 2. Or run: kaggle competitions download -c aptos2019-blindness-detection
    echo.
    pause
    exit /b 1
)

echo Dataset found! Starting training...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo ================================================================================
echo Training 1/3: Baseline Model (ResNet-50, Early Fusion)
echo Expected time: 2 hours on GPU
echo ================================================================================
python train.py ^
    --dataset aptos ^
    --data_dir data/raw ^
    --epochs 30 ^
    --batch_size 32 ^
    --backbone resnet50 ^
    --fusion early ^
    --lr 0.0001 ^
    --checkpoint_dir checkpoints/baseline_resnet50 ^
    --mixed_precision

if %ERRORLEVEL% NEQ 0 (
    echo Training 1 failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Training 2/3: Advanced Model (EfficientNet-B3, Attention Fusion)
echo Expected time: 4 hours on GPU
echo ================================================================================
python train.py ^
    --dataset aptos ^
    --data_dir data/raw ^
    --epochs 50 ^
    --batch_size 32 ^
    --backbone efficientnet_b3 ^
    --fusion attention_gated ^
    --lr 0.0001 ^
    --checkpoint_dir checkpoints/advanced_efficientnet_attention ^
    --mixed_precision

if %ERRORLEVEL% NEQ 0 (
    echo Training 2 failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Training 3/3: Production Model (EfficientNet-B3, Attention, QWK Loss)
echo Expected time: 4 hours on GPU
echo ================================================================================
python train.py ^
    --dataset aptos ^
    --data_dir data/raw ^
    --epochs 50 ^
    --batch_size 32 ^
    --backbone efficientnet_b3 ^
    --fusion attention_gated ^
    --loss qwk ^
    --lr 0.0001 ^
    --checkpoint_dir checkpoints/production_qwk ^
    --mixed_precision

if %ERRORLEVEL% NEQ 0 (
    echo Training 3 failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo All Training Complete!
echo ================================================================================
echo.
echo Models saved to:
echo - checkpoints/baseline_resnet50/
echo - checkpoints/advanced_efficientnet_attention/
echo - checkpoints/production_qwk/
echo.
echo Next steps:
echo 1. Evaluate models: python evaluate.py --checkpoint checkpoints/production_qwk/best_model.pth
echo 2. Launch web app: streamlit run app.py
echo.
pause
