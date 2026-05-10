#!/usr/bin/env python3
"""Export YOLO and DA2 to TensorRT FP16 engines for faster inference.

Run once from the repo root (with venv active):
    python scripts/export_trt.py [--yolo] [--da2] [--all]

After export, web_navigator.py auto-detects and uses the engines.
"""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'frodo_ai', 'perception'))
sys.path.insert(0, os.path.join(_ROOT, 'third_party', 'Depth-Anything-V2'))
sys.path.insert(0, os.path.join(_ROOT, 'third_party', 'Depth-Anything-V2', 'metric_depth'))

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def export_yolo():
    print("\n── YOLO 26m → TensorRT FP16 ──")
    from ultralytics import YOLO
    model = YOLO('yolo26m.pt')
    # Exports to yolo26m.engine in the current directory
    model.export(format='engine', device='cuda', half=True, imgsz=640, batch=1)
    src = 'yolo26m.engine'
    dst = os.path.join(SCRIPTS_DIR, 'yolo26m.engine')
    if os.path.exists(src) and src != dst:
        os.rename(src, dst)
    print(f"Saved: {dst}")
    print("web_navigator.py will use this engine automatically on next start.")


def export_da2(model_size='base'):
    print(f"\n── DA2 {model_size} → TensorRT FP16 ──")
    import torch
    import numpy as np
    from depth_estimator import DepthEstimator

    # Load the PyTorch model
    estimator = DepthEstimator(model_size=model_size, max_depth=10.0, device='cuda')
    model = estimator.model
    model.eval()

    # Export to ONNX with fixed 640x480 input (multiples of 14 → 630x476)
    H, W = 476, 630
    dummy = torch.randn(1, 3, H, W, device='cuda').half()
    model = model.half()

    onnx_path = os.path.join(SCRIPTS_DIR, f'da2_{model_size}.onnx')
    engine_path = os.path.join(SCRIPTS_DIR, f'da2_{model_size}.engine')

    print(f"Exporting ONNX → {onnx_path}")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['image'],
        output_names=['depth'],
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"Converting ONNX → TensorRT engine → {engine_path}")
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(i)}")
                return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
        config.set_flag(trt.BuilderFlag.FP16)

        serialized = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(serialized)

        print(f"Saved: {engine_path}")
        print("Update DepthWorker model_size to use engine path directly if needed.")

    except ImportError:
        print("tensorrt Python package not found.")
        print("Install via: pip install tensorrt")
        print(f"ONNX model saved at {onnx_path} — convert manually with trtexec:")
        print(f"  trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', action='store_true', help='Export YOLO to TRT')
    parser.add_argument('--da2',  action='store_true', help='Export DA2 to TRT')
    parser.add_argument('--all',  action='store_true', help='Export both')
    args = parser.parse_args()

    if not any([args.yolo, args.da2, args.all]):
        parser.print_help()
        print("\nExample: python scripts/export_trt.py --all")
        return

    if args.yolo or args.all:
        export_yolo()
    if args.da2 or args.all:
        export_da2(model_size='base')

    print("\nDone. Restart web_navigator.py to use the new engines.")


if __name__ == '__main__':
    main()
