@echo off
REM Suppress TensorFlow messages
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

REM SimpleNet Training Script for Windows - FULL TRAINING
REM CPU-only version (no GPU)
REM This will train on ALL MVTec AD classes

REM ===== CONFIGURATION =====
set DATASET_PATH=datasets\mvtec_anomaly_detection
set BATCH_SIZE=2
set META_EPOCHS=40
REM =========================

echo ========================================
echo SimpleNet - Full Dataset Training
echo ========================================
echo Dataset Path: %DATASET_PATH%
echo Epochs: %META_EPOCHS% (full training)
echo Batch Size: %BATCH_SIZE%
echo Device: CPU (no GPU)
echo Training on ALL classes...
echo ========================================
echo.
echo WARNING: This will take several hours on CPU!
echo Press Ctrl+C to cancel, or wait 5 seconds to continue...
timeout /t 5 /nobreak >nul
echo.

python main.py --seed 0 --log_group simplenet_full --log_project Full_Training --results_path results --run_name full_run net -b wideresnet50 -le layer2 -le layer3 --pretrain_embed_dimension 1536 --target_embed_dimension 1536 --patchsize 3 --meta_epochs %META_EPOCHS% --embedding_size 256 --gan_epochs 4 --noise_std 0.015 --dsc_hidden 1024 --dsc_layers 2 --dsc_margin .5 --pre_proj 1 dataset --batch_size %BATCH_SIZE% --resize 329 --imagesize 288 -d bottle -d cable -d capsule -d carpet -d grid -d hazelnut -d leather -d metal_nut -d pill -d screw -d tile -d toothbrush -d transistor -d wood -d zipper mvtec %DATASET_PATH%

echo.
echo ========================================
echo Full training completed!
echo Check results in: results\Full_Training\simplenet_full\full_run\
echo ========================================
echo.
echo Generating evaluation visualizations...
python generate_evaluations.py --results_path results --project Full_Training --group simplenet_full --run full_run
echo.
pause

