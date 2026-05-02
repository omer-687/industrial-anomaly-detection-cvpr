@echo off
REM Suppress TensorFlow messages
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

REM SimpleNet Training Script for Windows
REM CPU-only version (no GPU)

REM ===== CONFIGURATION - EDIT THESE =====
set DATASET_PATH=datasets\mvtec_anomaly_detection
set DATASET_NAME=bottle
set BATCH_SIZE=2
set META_EPOCHS=5
REM ======================================

echo Starting SimpleNet Training...
echo Dataset: %DATASET_NAME%
echo Dataset Path: %DATASET_PATH%
echo Device: CPU (no GPU)
echo.

python main.py --seed 0 --log_group simplenet_mvtec --log_project MVTecAD_Results --results_path results --run_name %DATASET_NAME%_run net -b wideresnet50 -le layer2 -le layer3 --pretrain_embed_dimension 1536 --target_embed_dimension 1536 --patchsize 3 --meta_epochs %META_EPOCHS% --embedding_size 256 --gan_epochs 4 --noise_std 0.015 --dsc_hidden 1024 --dsc_layers 2 --dsc_margin .5 --pre_proj 1 dataset --batch_size %BATCH_SIZE% --resize 329 --imagesize 288 -d %DATASET_NAME% mvtec %DATASET_PATH%

echo.
echo Training completed!
pause

