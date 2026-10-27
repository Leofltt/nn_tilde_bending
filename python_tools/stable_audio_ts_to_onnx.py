#!/usr/bin/env python3
"""
stable_audio_ts_to_onnx.py - Export Stable Audio 1.0 / Open VAE TorchScript (.ts) to ONNX and Bendable ONNX.

Takes any existing TorchScript Stable Audio VAE model (e.g. models/stable_audio_1_vae.ts),
converts it to standard ONNX with dynamic buffer sizes, promotes initializers into bendable
graph inputs for real-time NetBender circuit bending, and slices companion encode/decode subgraphs.
"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Warning: 'onnx' library is not installed. Run `pip install onnx` for full verification.", file=sys.stderr)


def slice_stable_audio_onnx_subgraphs(
    full_onnx_path: str,
    base_path: str,
    methods: List[str],
    latent_dim: int = 64,
    verbose: bool = True,
):
    """
    Slices the bendable Stable Audio ONNX graph into companion models for 'encode' and 'decode'
    using the bottleneck latent tensor between encoder_module and decoder_module.
    """
    try:
        full_model = onnx.load(full_onnx_path)
    except Exception as e:
        if verbose:
            print(f"  [Warning] Could not load {full_onnx_path} for companion slicing: {e}")
        return

    # Automatically identify the bottleneck latent tensor
    latent_tensor_name = None
    for n in full_model.graph.node:
        if "encoder" in n.name and "Slice" in n.op_type and len(n.output) > 0:
            latent_tensor_name = n.output[0]
            break

    if not latent_tensor_name:
        for n in full_model.graph.node:
            if "decoder" in n.name and "Conv" in n.op_type and len(n.input) > 0:
                latent_tensor_name = n.input[0]
                break

    if not latent_tensor_name:
        latent_tensor_name = "/encoder_module/Slice_output_0"

    if verbose:
        print(f"  -> Identified Stable Audio latent bottleneck tensor: '{latent_tensor_name}'")

    # 1. ENCODE SUBGRAPH
    if "encode" in methods:
        enc_path = f"{base_path}_encode.onnx"
        if verbose:
            print(f"  -> Slicing companion method 'encode' -> {enc_path}...")
        enc_model = copy.deepcopy(full_model)
        del enc_model.graph.output[:]
        latent_out = helper.make_tensor_value_info("latent_out", TensorProto.FLOAT, ["batch", latent_dim, "frames"])
        enc_model.graph.output.append(latent_out)

        for n in enc_model.graph.node:
            for i, out in enumerate(n.output):
                if out == latent_tensor_name:
                    n.output[i] = "latent_out"

        needed = {"latent_out"}
        nodes_to_keep = []
        for n in reversed(enc_model.graph.node):
            if any(out in needed for out in n.output):
                needed.update(n.input)
                nodes_to_keep.append(n)
        nodes_to_keep.reverse()

        del enc_model.graph.node[:]
        enc_model.graph.node.extend(nodes_to_keep)

        enc_inits = [init for init in enc_model.graph.initializer if init.name in needed]
        del enc_model.graph.initializer[:]
        enc_model.graph.initializer.extend(enc_inits)

        enc_inputs = [inp for inp in enc_model.graph.input if inp.name in needed]
        del enc_model.graph.input[:]
        enc_model.graph.input.extend(enc_inputs)

        onnx.save(enc_model, enc_path)
        if verbose:
            print(f"  -> Companion 'encode' exported successfully ({len(enc_model.graph.node)} nodes).")

    # 2. DECODE SUBGRAPH
    if "decode" in methods:
        dec_path = f"{base_path}_decode.onnx"
        if verbose:
            print(f"  -> Slicing companion method 'decode' -> {dec_path}...")
        dec_model = copy.deepcopy(full_model)
        latent_in = helper.make_tensor_value_info("latent_in", TensorProto.FLOAT, ["batch", latent_dim, "frames"])
        dec_model.graph.input.insert(0, latent_in)

        for n in dec_model.graph.node:
            for i, inp in enumerate(n.input):
                if inp == latent_tensor_name:
                    n.input[i] = "latent_in"

        needed = {o.name for o in dec_model.graph.output}
        nodes_to_keep = []
        for n in reversed(dec_model.graph.node):
            if any(out in needed for out in n.output):
                needed.update(n.input)
                nodes_to_keep.append(n)
        nodes_to_keep.reverse()

        del dec_model.graph.node[:]
        dec_model.graph.node.extend(nodes_to_keep)

        needed.add("latent_in")
        dec_inits = [init for init in dec_model.graph.initializer if init.name in needed]
        del dec_model.graph.initializer[:]
        dec_model.graph.initializer.extend(dec_inits)

        dec_inputs = [inp for inp in dec_model.graph.input if inp.name in needed]
        del dec_model.graph.input[:]
        dec_model.graph.input.extend(dec_inputs)

        onnx.save(dec_model, dec_path)
        if verbose:
            print(f"  -> Companion 'decode' exported successfully ({len(dec_model.graph.node)} nodes).")


def convert_stable_audio_ts_to_onnx(
    ts_model_path: str,
    output_onnx_path: Optional[str] = None,
    buffer_size: int = 8192,
    channels: int = 2,
    opset_version: int = 17,
    promote_for_bending: bool = True,
    methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> str:
    """
    Loads a Stable Audio TorchScript model (.ts) and exports it to bendable ONNX (.onnx).
    Supports exporting forward pass as well as companion subgraphs ('encode', 'decode').
    """
    if not os.path.exists(ts_model_path):
        raise FileNotFoundError(f"Model file not found: {ts_model_path}")

    if output_onnx_path is None:
        base, _ = os.path.splitext(ts_model_path)
        output_onnx_path = f"{base}_bendable.onnx" if promote_for_bending else f"{base}.onnx"

    base, ext = os.path.splitext(output_onnx_path)
    std_onnx_path = output_onnx_path if not promote_for_bending else f"{base}_raw.onnx"
    os.makedirs(os.path.dirname(os.path.abspath(std_onnx_path)), exist_ok=True)

    if verbose:
        print(f"\n[1/3] Loading Stable Audio TorchScript model: {ts_model_path}")

    model = torch.jit.load(ts_model_path, map_location="cpu")
    model.eval()

    # Extract metadata properties if available on the model
    sr = 44100
    try:
        if hasattr(model, "sr"):
            val = model.sr() if callable(model.sr) else model.sr
            sr = int(val)
        elif hasattr(model, "sample_rate"):
            val = model.sample_rate() if callable(model.sample_rate) else model.sample_rate
            sr = int(val)
    except Exception:
        pass

    ratio = 2048
    latent_dim = 64
    in_channels = channels
    out_channels = channels

    try:
        if hasattr(model, "encode_params"):
            ep = model.encode_params() if callable(model.encode_params) else model.encode_params
            if hasattr(ep, "__iter__") or isinstance(ep, (list, tuple, torch.Tensor)):
                in_channels = int(ep[0])
                latent_dim = int(ep[2])
                ratio = int(ep[3])
        elif hasattr(model, "decode_params"):
            dp = model.decode_params() if callable(model.decode_params) else model.decode_params
            if hasattr(dp, "__iter__") or isinstance(dp, (list, tuple, torch.Tensor)):
                latent_dim = int(dp[0])
                ratio = int(dp[1])
                out_channels = int(dp[2])
    except Exception:
        pass

    dummy_audio = torch.zeros(1, in_channels, buffer_size, dtype=torch.float32)

    if verbose:
        print(f"[2/3] Exporting to standard ONNX (opset {opset_version}): {std_onnx_path}...")
        print(f"      Channels: {in_channels} -> {out_channels} | Latent Dim: {latent_dim} | Ratio: {ratio} | SR: {sr} Hz")

    torch.onnx.export(
        model,
        (dummy_audio,),
        std_onnx_path,
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes={
            "audio_in": {0: "batch", 2: "samples"},
            "audio_out": {0: "batch", 2: "samples"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    if verbose:
        print("  -> Exported standard ONNX successfully.")

    extra_meta = {
        "netbender.sample_rate": str(sr),
        "netbender.ratio": str(ratio),
        "netbender.latent_dim": str(latent_dim),
        "netbender.in_channels": str(in_channels),
        "netbender.out_channels": str(out_channels),
    }

    if promote_for_bending:
        if verbose:
            print(f"[3/3] Promoting weights to graph inputs for real-time bending: {output_onnx_path}...")

        from python_tools.onnx_bend_prep import promote_initializers_to_inputs

        promote_initializers_to_inputs(
            input_path=std_onnx_path,
            output_path=output_onnx_path,
            keep_initializers_as_defaults=True,
            verbose=verbose,
            extra_metadata=extra_meta,
        )

        if std_onnx_path != output_onnx_path and os.path.exists(std_onnx_path):
            try:
                os.remove(std_onnx_path)
            except OSError:
                pass

    if methods:
        candidate_methods = ["encode", "decode"]
        available_ts_methods = [m for m in candidate_methods if hasattr(model, m)]
        export_list = []
        if "all" in methods:
            export_list = ["encode", "decode"]
        else:
            export_list = [m for m in methods if m in candidate_methods and m != "forward"]

        if export_list:
            slice_stable_audio_onnx_subgraphs(
                output_onnx_path,
                base,
                export_list,
                latent_dim=latent_dim,
                verbose=verbose,
            )

    if verbose:
        print(f"\n[DONE] Ready for NetBender ONNX inference: {output_onnx_path}\n")

    return output_onnx_path


def main():
    parser = argparse.ArgumentParser(description="Convert Stable Audio TorchScript (.ts) to NetBender ONNX")
    parser.add_argument("-i", "--input", required=True, help="Input TorchScript model (.ts)")
    parser.add_argument("-o", "--output", default=None, help="Output ONNX model (.onnx)")
    parser.add_argument("--buffer-size", type=int, default=8192, help="Buffer size for tracing (default: 8192)")
    parser.add_argument("--channels", type=int, default=2, help="Audio channels (default: 2 for stereo)")
    parser.add_argument("--no-bending", action="store_true", help="Skip promoting initializers to graph inputs")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Companion methods to export (e.g. all, encode, decode). Default: all",
    )
    args = parser.parse_args()

    convert_stable_audio_ts_to_onnx(
        ts_model_path=args.input,
        output_onnx_path=args.output,
        buffer_size=args.buffer_size,
        channels=args.channels,
        promote_for_bending=not args.no_bending,
        methods=args.methods,
        verbose=True,
    )


if __name__ == "__main__":
    main()
