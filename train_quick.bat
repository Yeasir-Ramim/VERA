@echo off
REM Quick Training Script - Single Best Model

echo ================================================================================
echo VERA Quick Training - Production Model on APTOS
echo ================================================================================
echo.

REM Check if data exists
if not exist "data\raw\aptos2019\train.csv" (
    echo ERROR: APTOS dataset not found!
    echo.
    echo Please download first. Choose one method:
    echo.
    echo METHOD 1 - Kaggle API (Fastest):
    echo   pip install kaggle
    echo   kaggle competitions download -c aptos2019-blindness-detection -p data/raw/aptos2019/
    echo   cd data/raw/aptos2019
    echo   tar -xf train_images.zip
    echo   cd ../../..
    echo.
    echo METHOD 2 - Manual:
    echo   1. Visit: https://www.kaggle.com/c/aptos2019-blindness-detection/data
    echo   2. Download and extract to data/raw/aptos2019/
    echo.
    echo See DOWNLOAD_DATASETS.md for detailed instructions
    echo.
    pause
    exit /b 1
)

echo Dataset found! Starting training...
echo.
echo Configuration:
echo - Dataset: APTOS 2019 (3,662 images)
echo - Model: EfficientNet-B3 with Attention-Gated Fusion
echo - Loss: Quadratic Weighted Kappa (QWK)
echo - Training time: ~4 hours on GPU, ~16 hours on CPU
echo.
echo Press any key to start training or Ctrl+C to cancel...
pause >nul

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python train.py ^
    --dataset aptos ^
    --data_dir data/raw ^
    --epochs 50 ^
    --batch_size 32 ^
    --backbone efficientnet_b3 ^
    --fusion attention_gated ^
    --loss qwk ^
    --optimizer adamw ^
    --scheduler cosine ^
    --mixed_precision ^
    --class_weights ^
    --checkpoint_dir checkpoints/production_model

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo Training Complete!
    echo ================================================================================
    echo.
    echo Best model saved to: checkpoints/production_model/best_model.pth
    echo.
    echo Next steps:
    echo 1. Evaluate: python evaluate.py --checkpoint checkpoints/production_model/best_model.pth --dataset aptos
    echo 2. Web app: streamlit run app.py
    echo.
) else (
    echo.
    echo Training failed! Check the error messages above.
    echo.
)

pause
