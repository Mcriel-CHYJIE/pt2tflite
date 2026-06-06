"""YOLO fall detection model converter: PT → ONNX → TFLite

Usage:
    conda activate pt2tflite

    # Full pipeline
    python scripts/convert_fall.py --source path/to/best.pt

    # Stop at ONNX only
    python scripts/convert_fall.py --source path/to/best.pt --step onnx

    # Custom name + imgsz + fp16 quantization
    python scripts/convert_fall.py --source best.pt --name fall_v8n --imgsz 640 --quantize fp16

Requires: ultralytics, onnx, onnx2tf, tensorflow, onnxruntime
"""
import sys, shutil, os, argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output'
CACHE = ROOT / 'cache'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def detect_imgsz(pt_path: Path) -> int:
    """Try to read imgsz from the training metadata embedded in a YOLO .pt file."""
    try:
        import torch
        ckpt = torch.load(str(pt_path), map_location='cpu', weights_only=False)
        # YOLO stores args in ckpt['train_args'] or model.args
        for key in ('train_args', 'args'):
            val = ckpt.get(key, {})
            if isinstance(val, dict):
                sz = val.get('imgsz', None)
                if sz:
                    return sz[0] if isinstance(sz, (list, tuple)) else int(sz)
        # Fallback: check model.yaml stride for common sizes
        model = ckpt.get('model', None)
        if model and hasattr(model, 'yaml'):
            yaml = model.yaml if isinstance(model.yaml, dict) else {}
            sz = yaml.get('imgsz', None)
            if sz:
                return sz[0] if isinstance(sz, (list, tuple)) else int(sz)
    except Exception:
        pass
    return 640  # safe default


def convert(source_pt: str, name: str = None, imgsz: int = None,
            step: str = 'tflite', quantize: str = 'fp16'):
    SRC = Path(source_pt)
    if not SRC.exists():
        print(f'[ERR] Source not found: {SRC}')
        return False

    # Auto-name from parent run dir
    if name is None:
        parts = SRC.relative_to(ROOT.anchor) if SRC.is_absolute() else SRC
        name = SRC.parent.parent.stem if SRC.parent.parent.stem != 'weights' else SRC.stem

    # Auto-detect imgsz
    if imgsz is None:
        imgsz = detect_imgsz(SRC)
        print(f'  [i] Auto-detected imgsz={imgsz}', flush=True)

    print(f'\n{"="*55}')
    print(f'  Convert: {SRC.name}')
    print(f'  Output:  {name} (imgsz={imgsz})')
    print(f'  Target:  {step.upper()}')
    print(f'{"="*55}\n')

    # ── Step 1: PT → ONNX ──────────────────────────────────────────
    print('[1/3] PT → ONNX ...', flush=True)
    from ultralytics import YOLO
    model = YOLO(str(SRC))

    onnx_kwargs = dict(format='onnx', imgsz=imgsz, opset=19, simplify=True,
                       project=str(SRC.parent), name='.')
    model.export(**onnx_kwargs)

    onnx_src = _pick_latest(SRC.parent.glob('*.onnx'))
    if onnx_src is None:
        print('  [ERR] ONNX export produced no output file')
        return False

    onnx_dst = OUT / f'{name}.onnx'
    shutil.copy2(str(onnx_src), str(onnx_dst))

    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_dst)))
    print(f'  [OK] {onnx_dst.name}  ({onnx_dst.stat().st_size / 1024:.0f} KB)', flush=True)

    if step == 'onnx':
        print('\n[OK] Stopped after ONNX (--step onnx)')
        return True

    # ── Step 2: ONNX → TF SavedModel ───────────────────────────────
    print('[2/3] ONNX → TF SavedModel (via onnx2tf) ...', flush=True)
    sm_dir = str(CACHE / f'{name}_saved_model')
    if Path(sm_dir).exists():
        shutil.rmtree(sm_dir)

    ok = _onnx2tf_convert(str(onnx_dst), sm_dir)
    if not ok:
        print('  [ERR] onnx2tf conversion failed')
        return False
    print('  [OK] SavedModel ready', flush=True)

    # Locate saved_model.pb
    sm_path = _find_saved_model(sm_dir)
    if sm_path is None:
        print('  [ERR] saved_model.pb not found in onnx2tf output')
        return False

    if step == 'tf':
        print('\n[OK] Stopped after SavedModel (--step tf)')
        return True

    # ── Step 3: TF SavedModel → TFLite ────────────────────────────
    print(f'[3/3] SavedModel → TFLite  (quantize={quantize}) ...', flush=True)
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)

    # Quantization
    if quantize == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # INT8 needs a representative dataset — warn but still try
        print('  [i] INT8 without representative dataset → fallback to FP16')
        converter.target_spec.supported_types = [tf.float16]
    # 'none' → default (FP32)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    suffix = '' if quantize == 'none' else f'_{quantize}'
    tflite_path = OUT / f'{name}{suffix}.tflite'
    with open(str(tflite_path), 'wb') as f:
        f.write(tflite_model)
    print(f'  [OK] {tflite_path.name}  ({tflite_path.stat().st_size / 1024:.0f} KB)', flush=True)

    # ── Verify ──────────────────────────────────────────────────────
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f'  Input:  {inp["shape"]}  {inp["dtype"]}')
    print(f'  Output: {out["shape"]}  {out["dtype"]}')
    print(f'  Flex ops: {"SELECT_TF_OPS" in str(converter.target_spec.supported_ops)}')

    print(f'\n{"="*55}')
    print(f'  [OK] Done!  {tflite_path}')
    print(f'{"="*55}')
    return True


# ── Helpers ────────────────────────────────────────────────────────

def _onnx2tf_convert(onnx_path: str, output_dir: str) -> bool:
    """Convert ONNX → TF SavedModel using onnx2tf (direct import, with numpy + download monkey-patches)."""
    import numpy as np
    # Patch 1: np.load with allow_pickle (numpy >= 1.16.3 defaults False)
    old_load = np.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault('allow_pickle', True)
        return old_load(*args, **kwargs)
    np.load = _patched_load

    # Patch 2: replace download_test_image_data with dummy data
    # (avoids network download + pickle corruption issues on Chinese Windows)
    import onnx2tf.onnx2tf as _onnx2tf_mod
    old_download = _onnx2tf_mod.download_test_image_data
    _onnx2tf_mod.download_test_image_data = lambda: np.zeros(
        (20, 128, 128, 3), dtype=np.uint8) + 128

    try:
        import logging
        logging.getLogger('onnx2tf').setLevel(logging.WARNING)

        _onnx2tf_mod.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            non_verbose=True,
            output_signaturedefs=True,
        )
        return True
    except Exception as e:
        print(f'  [onnx2tf] {type(e).__name__}: {e}', flush=True)
        return False
    finally:
        np.load = old_load
        _onnx2tf_mod.download_test_image_data = old_download


def _pick_latest(files):
    """Return the most recently modified file from an iterable, or None."""
    entries = list(files)
    return max(entries, key=lambda f: f.stat().st_mtime) if entries else None


def _find_saved_model(base: str):
    """Walk a directory tree looking for saved_model.pb, return its parent."""
    for p in Path(base).rglob('saved_model.pb'):
        return str(p.parent)
    return base if Path(base).joinpath('saved_model.pb').exists() else None


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLO PT → ONNX → TFLite converter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Path to .pt weights file')
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='Output name prefix (default: auto from run dir)')
    parser.add_argument('--imgsz', '-i', type=int, default=None,
                        help='Input image size (default: auto-detect)')
    parser.add_argument('--step', type=str, default='tflite',
                        choices=['onnx', 'tf', 'tflite'],
                        help='Stop point: onnx / tf / tflite (default: tflite)')
    parser.add_argument('--quantize', '-q', type=str, default='fp16',
                        choices=['none', 'fp16', 'int8'],
                        help='TFLite quantization (default: fp16)')
    args = parser.parse_args()

    success = convert(args.source, args.name, args.imgsz,
                      step=args.step, quantize=args.quantize)
    sys.exit(0 if success else 1)
