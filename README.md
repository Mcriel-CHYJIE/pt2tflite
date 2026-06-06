<div align="center">

# pt2tflite

**YOLO PyTorch → TFLite Converter** · *YOLO 模型转换工具链*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![ONNX](https://img.shields.io/badge/ONNX-✓-005CED?logo=onnx)](https://onnx.ai)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-✓-00A8E8)](https://ultralytics.com)

将 Ultralytics YOLO 模型（`.pt`）一键转换为 Android 可部署的 `.tflite` 格式。  
*One-click pipeline from Ultralytics YOLO (`.pt`) to Android-ready `.tflite`.*

</div>

---

## 📋 Table of Contents · 目录

- [Overview · 项目概览](#overview--项目概览)
- [Pipeline · 转换管线](#pipeline--转换管线)
- [Requirements · 环境要求](#requirements--环境要求)
- [Quick Start · 快速开始](#quick-start--快速开始)
- [Usage · 使用指南](#usage--使用指南)
  - [Basic · 基本用法](#basic--基本用法)
  - [Options · 完整参数](#options--完整参数)
  - [Step-by-Step · 分步执行](#step-by-step--分步执行)
  - [WSL · 从 WSL 调用](#wsl--从-wsl-调用)
- [Artifacts · 输出产物](#artifacts--输出产物)
- [Android Integration · Android 集成](#android-integration--android-集成)
- [Troubleshooting · 常见问题](#troubleshooting--常见问题)

---

## Overview · 项目概览

This project provides a **reliable, repeatable pipeline** to convert YOLO detection models trained with Ultralytics into formats suitable for mobile deployment.

本项目提供一套**可靠可复现**的转换管线，将 Ultralytics 训练的 YOLO 检测模型转换为移动端可部署的格式。

**Key Features · 核心特性：**

- ✅ **End-to-end** — PT → ONNX → TF SavedModel → TFLite, single command
- ✅ **Chinese Windows friendly** — patches around GBK encoding issues in ultralytics auto-updates
- ✅ **Auto-detection** — reads `imgsz` and model name from `.pt` metadata automatically
- ✅ **Flex ops support** — preserves `FlexSplitV` etc. via SELECT_TF_OPS
- ✅ **Quantization** — FP16 / INT8 / FP32 via `--quantize`
- ✅ **Step control** — `--step onnx|tf|tflite` to stop at intermediate formats

---

## Pipeline · 转换管线

```
                       ┌──────────────┐
                       │   best.pt    │  (PyTorch / Ultralytics YOLO)
                       └──────┬───────┘
                              │
                     ┌────────▼────────┐
                     │  ultralytics    │  opset 19, simplify
                     │  export onnx    │
                     └────────┬────────┘
                              │
                       ┌──────▼───────┐
                       │  model.onnx  │  (ONNX, ~11 MB for YOLOv8n)
                       └──────┬───────┘
                              │
                     ┌────────▼────────┐
                     │    onnx2tf     │  direct import, numpy monkey-patch
                     │    convert     │
                     └────────┬────────┘
                              │
                    ┌─────────▼──────────┐
                    │  saved_model.pb    │  (TensorFlow SavedModel)
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  TFLiteConverter   │  SELECT_TF_OPS enabled
                    └─────────┬──────────┘
                              │
                       ┌──────▼───────┐
                       │  model.tflite│  (TFLite, 5-10 MB)
                       └──────────────┘
```

Each step produces an independently verifiable artifact.  
*每步产物均可独立验证。*

---

## Requirements · 环境要求

| Dependency · 依赖 | Version · 版本 | Purpose · 用途 |
|---|---|---|
| Python | 3.10+ | Runtime |
| PyTorch | ≥2.0 | Weight loading |
| TensorFlow | **2.16.x** | TFLite conversion |
| ultralytics | ≥8.4 | PT → ONNX |
| onnx2tf | ≥1.26 | ONNX → TF |
| onnxruntime | ≥1.18 | ONNX verification |
| onnx | ≥1.15 | ONNX tooling |

### One-click Setup · 一键安装

```batch
setup_env.bat
```

Creates a `pt2tflite` conda environment with all dependencies.  
*创建 pt2tflite conda 环境并安装全部依赖。*

### Manual Setup · 手动安装

```batch
conda create -n pt2tflite python=3.10
conda activate pt2tflite

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow==2.16.1 tf-keras==2.16.1
pip install ultralytics onnx2tf onnxruntime onnx
```

> **⚠️ Read-only env fix** · *环境只读修复*  
> If Anaconda was installed as Administrator, env directories may be read-only.
> Run as admin: `icacls D:\Anaconda3\envs\pt2tflite /grant "%USERNAME%:(OI)(CI)F" /T /Q`

---

## Quick Start · 快速开始

```batch
conda activate pt2tflite

python scripts/convert_fall.py --source D:\path\to\your-model.pt
```

That's it. Output appears in `output/`.  
*一行完成，产物在 `output/` 目录下。*

---

## Usage · 使用指南

### Basic · 基本用法

```batch
python scripts/convert_fall.py --source best.pt
```

The script auto-detects:
- **`imgsz`** — reads training image size from `.pt` metadata
- **`name`** — derives from the run directory name (e.g., `0604_run_xxx`)

### Options · 完整参数

| Argument · 参数 | Default · 默认 | Description · 说明 |
|---|---|---|
| `--source` / `-s` | *(required)* | Path to `.pt` weights |
| `--name` / `-n` | auto | Output filename prefix |
| `--imgsz` / `-i` | auto-detect | Input image size |
| `--step` | `tflite` | Stop point: `onnx` / `tf` / `tflite` |
| `--quantize` / `-q` | `fp16` | Quantization: `none` / `fp16` / `int8` |

```batch
# Custom name, size, quantization
python scripts/convert_fall.py --source best.pt --name fall_v8n --imgsz 640 --quantize fp16

# FP32 only (no quantization)
python scripts/convert_fall.py --source best.pt --quantize none
```

### Step-by-Step · 分步执行

```batch
# ONNX only · 只转 ONNX
python scripts/convert_fall.py --source best.pt --step onnx

# TF SavedModel only · 只做到 TF
python scripts/convert_fall.py --source best.pt --step tf

# Full pipeline · 完整流程
python scripts/convert_fall.py --source best.pt --step tflite
```

### WSL · 从 WSL 调用

When calling Windows Python from WSL, always `cd /d C:\` first to avoid UNC path issues:  
*WSL 中调用 Windows Python 时需先切到 C:\ 根目录避免 UNC 路径问题：*

```bash
cmd.exe /c "cd /d C:\ && <python_path> <project_root>\scripts\convert_fall.py \
  --source <path_to_pt>"
```

---

## Artifacts · 输出产物

### Naming Convention · 命名规则

```
{run_name}_{quantize_suffix}.{ext}
```

| Variable · 变量 | Source · 来源 | Example · 示例 |
|---|---|---|
| `run_name` | Training run directory | `0604_run_xxx` |
| `quantize_suffix` | `none` → *(empty)*, `fp16` → `_fp16` | `_fp16` |
| `ext` | `onnx` / `tflite` | |

### Current Inventory · 当前产物

| File · 文件 | Source · 来源 | Size · 大小 | IO Shapes · 输入/输出形状 |
|---|---|---|---|
| `xxx.onnx` | YOLOv8n (640) | ~11 MB | `[1,3,640,640]` → `[1,8,8400]` |
| `xxx_fp16.tflite` | YOLOv8n **FP16** | ~5 MB | `[1,640,640,3]` → `[1,8,8400]` |
| `xxx.tflite` | YOLO11n (736) | ~10 MB | `[1,736,736,3]` → `[1,8,11109]` |
| `xxx.onnx` | YOLO11n-pose (640) | ~11 MB | — |
| `xxx.tflite` | YOLO11n-pose (640) | ~11 MB | — |

### Important Notes · 重要说明

- **Flex ops** — All TFLite models contain `Select TF ops` (e.g. `FlexSplitV`).  
  Android deployment requires `tensorflow-lite-select-tf-ops`.  
  *所有 TFLite 模型包含 Flex 算子，Android 需添加额外依赖。*
- **Intermediate artifacts** — TF SavedModels are cached in `cache/` for reuse.  
  *中间产物保存在 `cache/` 目录，可复用。*
- **ONNX visualization** — Load `.onnx` files into [Netron](https://netron.app) to inspect.  
  *ONNX 文件可通过 Netron 可视化。*

---

## Android Integration · Android 集成

### Gradle Dependencies

```gradle
dependencies {
    // Core runtime
    implementation 'org.tensorflow:tensorflow-lite:2.16.1'
    // Flex ops support (required for models with Select TF ops)
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1'
    // GPU delegate (optional, for hardware acceleration)
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.16.1'
}
```

### Interpreter Initialization · 初始化

```kotlin
val options = Interpreter.Options().apply {
    addDelegate(GpuDelegate())       // GPU acceleration
    addDelegate(NnApiDelegate())      // NNAPI fallback
}
val interpreter = Interpreter(tfliteModel, options)
```

### Output Decoding · 输出解析

The models output YOLO **raw grid format** `[1, C, N]`:

| Dim | Meaning | YOLOv8n (640) | YOLO11n (736) |
|---|---|---|---|
| B | Batch size | 1 | 1 |
| C | Channels: 4(bbox) + 1(obj) + 1(cls) + 2(extra) | 8 | 8 |
| N | Total grid cells | 8400 | 11109 |

You need to implement **grid-based decode + NMS** to extract bounding boxes:

```kotlin
// Pseudo-code: per grid cell
val cx = (sigmoid(box[0]) * 2 - 0.5 + gridX) * stride
val cy = (sigmoid(box[1]) * 2 - 0.5 + gridY) * stride
val w  = (sigmoid(box[2]) * 2) ^ 2 * anchorW
val h  = (sigmoid(box[3]) * 2) ^ 2 * anchorH
```

> **MIRO project reference:** GPU → NNAPI → CPU triple fallback with 90° CCW coordinate rotation for portrait mode. Final output `[1,300,6]` = `[x1,y1,x2,y2,conf,cls]`.  
> *MIRO 项目参考：三级回退、竖屏坐标旋转、后处理输出格式。*

---

## Troubleshooting · 常见问题

### ❌ FlexSplitV failed to prepare

**Missing `tensorflow-lite-select-tf-ops` dependency.** Add it to `app/build.gradle`.  
*缺少 Flex 算子依赖，添加到 Gradle 即可。*

### ❌ 'gbk' codec can't decode byte

**Chinese Windows encoding issue** with ultralytics auto-update.  
This project bypasses it by using manual onnx2tf conversion. If still occurring, run `chcp 65001` first.  
*中文 Windows 编码问题。本项目已绕过 ultralytics 内置导出，若仍有问题先切换代码页。*

### ❌ onnx2tf is slow or hangs

**Normal for 1-3 minutes.** If >10 min, reduce `--imgsz` or free up RAM.  
*1-3 分钟正常。超过 10 分钟降低 imgsz 或释放内存。*

### ❌ TFLite is 200+ MB

**Abnormal onnx2tf output.** Delete `cache/` and retry.  
*异常输出，删除 cache/ 目录重试。*

---

## License · 许可证

```
MIT License — feel free to use, modify, and distribute.
```

---

<div align="center">

*For detailed step-by-step instructions, see [执行.md](./执行.md).*  
*详细分步操作指南请参阅 [执行.md](./执行.md)。*

</div>
