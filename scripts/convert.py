#!/usr/bin/env python
"""General PT → ONNX → TFLite converter.

Supports two modes:
  1. YOLO (ultralytics) — auto-handles the full pipeline
  2. Generic PyTorch — torch.onnx.export → onnx2tf → TFLiteConverter

Usage:
    # YOLO mode (recommended)
    python scripts/convert.py --yolo path/to/best.pt --imgsz 640 --quantize fp16

    # Generic mode (any torch model)
    python scripts/convert.py --model model.pt --input-shape 1 3 224 224

Requires: torch, onnx, onnx2tf, tensorflow
For YOLO mode: ultralytics
"""
import sys, shutil, subprocess, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output'
CACHE = ROOT / 'cache'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


# ── YOLO mode ──────────────────────────────────────────────────────

def convert_yolo(pt_path: str, name: str = None, imgsz: int = 640,
                 step: str = 'tflite', quantize: str = 'fp16') -> bool:
    """Delegate to convert_fall.py for ultralytics models."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from convert_fall import convert
    return convert(pt_path, name, imgsz, step, quantize)


# ── Generic mode ───────────────────────────────────────────────────

def convert_generic(pt_path: str, name: str, input_shape, opset=17,
                    step='tflite', quantize='fp16') -> bool:
    import torch, onnx, tensorflow as tf

    SRC = Path(pt_path)
    onnx_dst = OUT / f'{name}.onnx'
    sm_dir = str(CACHE / f'{name}_saved_model')
    tflite_dst = OUT / f'{name}.tflite'

    # Step 1: PT → ONNX
    print('[1/3] PT → ONNX ...', flush=True)
    model = torch.load(str(SRC), map_location='cpu')
    model.eval()
    dummy = torch.randn(*input_shape)

    torch.onnx.export(model, dummy, str(onnx_dst),
                      input_names=['input'], output_names=['output'],
                      opset_version=opset, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(onnx_dst)))
    print(f'  [OK] {onnx_dst.name}  ({onnx_dst.stat().st_size / 1024:.0f} KB)', flush=True)
    if step == 'onnx':
        return True

    # Step 2: ONNX → TF SavedModel
    print('[2/3] ONNX → TF SavedModel ...', flush=True)
    if Path(sm_dir).exists():
        shutil.rmtree(sm_dir)
    r = subprocess.run([sys.executable, '-m', 'onnx2tf',
                        '-i', str(onnx_dst), '-o', sm_dir, '-osd'],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f'  [ERR] onnx2tf failed:\n{r.stderr[-300:]}', flush=True)
        return False
    print('  [OK] SavedModel ready', flush=True)
    if step == 'tf':
        return True

    # Step 3: TF → TFLite
    print(f'[3/3] TF → TFLite  (quantize={quantize}) ...', flush=True)
    sm_path = sm_dir
    for p in Path(sm_dir).rglob('saved_model.pb'):
        sm_path = str(p.parent)
        break
    converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    if quantize == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(str(tflite_dst), 'wb') as f:
        f.write(tflite_model)
    print(f'  [OK] {tflite_dst.name}  ({tflite_dst.stat().st_size / 1024:.0f} KB)', flush=True)
    return True


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PT → ONNX → TFLite converter')
    mg = parser.add_mutually_exclusive_group(required=True)
    mg.add_argument('--yolo', type=str, default=None,
                    help='YOLO .pt weights (ultralytics pipeline)')
    mg.add_argument('--model', type=str, default=None,
                    help='Generic PyTorch .pt/.pth model')

    parser.add_argument('--name', type=str, default=None, help='Output name prefix')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (YOLO mode)')
    parser.add_argument('--input-shape', type=int, nargs='+',
                        default=[1, 3, 640, 640], help='Input shape (generic mode)')
    parser.add_argument('--step', default='tflite',
                        choices=['onnx', 'tf', 'tflite'])
    parser.add_argument('--quantize', default='fp16',
                        choices=['none', 'fp16', 'int8'])
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset (generic mode)')
    args = parser.parse_args()

    if args.yolo:
        ok = convert_yolo(args.yolo, args.name, args.imgsz,
                          args.step, args.quantize)
    else:
        name = args.name or Path(args.model).stem
        ok = convert_generic(args.model, name, args.input_shape,
                             args.opset, args.step, args.quantize)

    sys.exit(0 if ok else 1)
