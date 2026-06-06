@echo off
REM ============================================
REM Setup pt2tflite env for YOLO TFLite export
REM ============================================
echo === Creating fresh pt2tflite env ===
call D:\Anaconda3\Scripts\conda.exe create -y -n pt2tflite python=3.10

echo === Installing PyTorch (CUDA 11.8) ===
call D:\Anaconda3\envs\pt2tflite\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --default-timeout=120

echo === Installing TensorFlow 2.16 + onnx ===
call D:\Anaconda3\envs\pt2tflite\python.exe -m pip install tensorflow==2.16.1 tf-keras==2.16.1 --default-timeout=120

echo === Installing ultralytics + onnx2tf + onnxruntime ===
call D:\Anaconda3\envs\pt2tflite\python.exe -m pip install ultralytics onnx2tf onnxruntime --default-timeout=120

echo === Verifying ===
call D:\Anaconda3\envs\pt2tflite\python.exe -c "import tensorflow as tf; print('TF', tf.__version__); import torch; print('Torch', torch.__version__); from ultralytics import YOLO; import onnx2tf; print('All OK')"

REM ============================================
REM Fix: if env is read-only (admin install),
REM run this as admin:
REM   icacls D:\Anaconda3\envs\pt2tflite /grant "%USERNAME%:(OI)(CI)F" /T /Q
REM ============================================
pause
