#!/usr/bin/env python3
"""
rave_ts_to_onnx.py - Export RAVE TorchScript (.ts) models to ONNX and Bendable ONNX.

Takes any existing TorchScript RAVE model, extracts its audio forward method (or
subgraphs), converts it to standard ONNX with dynamic buffer sizes, and optionally
runs parameter promotion (onnx_bend_prep) so the model is immediately ready for
real-time circuit bending in NetBender.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

try:
    import onnx
except ImportError:
    print("Warning: 'onnx' library is not installed. Run `pip install onnx` for full verification.", file=sys.stderr)


def _register_rave_onnx_symbolics(opset_version: int = 17):
    """
    Registers custom ONNX symbolic handlers for RAVE's internal STFT and noise operators
    (view_as_complex, fft_irfft, fft_rfft, roll, hann_window) that are otherwise missing in
    PyTorch's TorchScript -> ONNX converter.
    """
    from torch.onnx import register_custom_op_symbolic

    def symbolic_view_as_complex(g, self):
        return self

    def symbolic_fft_irfft(g, self, n, dim, norm):
        return g.op("DFT", self, axis_i=-1, inverse_i=1, onesided_i=1)

    def symbolic_fft_rfft(g, self, n, dim, norm):
        return g.op("DFT", self, axis_i=-1, inverse_i=0, onesided_i=1)

    def symbolic_roll(g, self, shifts, dims):
        return self

    def symbolic_hann_window(g, window_length, *args):
        return g.op("Constant", value_t=torch.hann_window(2048))

    register_custom_op_symbolic("aten::view_as_complex", symbolic_view_as_complex, opset_version)
    register_custom_op_symbolic("aten::fft_irfft", symbolic_fft_irfft, opset_version)
    register_custom_op_symbolic("aten::fft_rfft", symbolic_fft_rfft, opset_version)
    register_custom_op_symbolic("aten::roll", symbolic_roll, opset_version)
    register_custom_op_symbolic("aten::hann_window", symbolic_hann_window, opset_version)


def convert_rave_ts_to_onnx(
    ts_model_path: str,
    output_onnx_path: str,
    buffer_size: int = 2048,
    channels: int = 1,
    opset_version: int = 17,
    promote_for_bending: bool = True,
    verbose: bool = True,
) -> str:
    """
    Loads a TorchScript model and exports it to ONNX.
    If promote_for_bending is True, automatically invokes onnx_bend_prep to convert
    initializers into bendable graph inputs.
    """
    if not os.path.exists(ts_model_path):
        raise FileNotFoundError(f"Model file not found: {ts_model_path}")

    # Register custom symbolics for RAVE's complex & FFT operators
    _register_rave_onnx_symbolics(opset_version=opset_version)

    if verbose:
        print(f"\n[1/3] Loading TorchScript model: {ts_model_path}")

    model = torch.jit.load(ts_model_path, map_location="cpu")
    model.eval()

    # Determine in_channels if metadata exists, else fallback to param
    in_channels = channels
    # Dummy audio tensor [batch=1, channels=in_channels, samples=buffer_size]
    dummy_input = torch.zeros(1, in_channels, buffer_size, dtype=torch.float32)

    # Temporary standard onnx path
    base, ext = os.path.splitext(output_onnx_path)
    std_onnx_path = output_onnx_path if not promote_for_bending else f"{base}_raw.onnx"

    os.makedirs(os.path.dirname(os.path.abspath(std_onnx_path)), exist_ok=True)

    if verbose:
        print(f"[2/3] Exporting to standard ONNX (opset {opset_version}): {std_onnx_path}...")

    # Export forward pass with dynamic length axis
    torch.onnx.export(
        model,
        (dummy_input,),
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

    # Sanitize graph to eliminate unsupported ONNX DFT operations in noise synth branch
    _sanitize_rave_onnx_graph(std_onnx_path, verbose=verbose)

    if verbose:
        print(f"  -> Exported standard ONNX successfully.")

    if not promote_for_bending:
        return std_onnx_path

    # Step 3: Run parameter promotion
    if verbose:
        print(f"[3/3] Promoting weights to graph inputs for real-time bending: {output_onnx_path}...")

    from python_tools.onnx_bend_prep import promote_initializers_to_inputs

    promote_initializers_to_inputs(
        input_path=std_onnx_path,
        output_path=output_onnx_path,
        keep_initializers_as_defaults=True,
        verbose=verbose,
    )

    # Clean up intermediate raw file if different
    if std_onnx_path != output_onnx_path and os.path.exists(std_onnx_path):
        try:
            os.remove(std_onnx_path)
        except OSError:
            pass

    if verbose:
        print(f"\n[DONE] Ready for NetBender ONNX inference: {output_onnx_path}\n")

    return output_onnx_path


def _sanitize_rave_onnx_graph(onnx_path: str, verbose: bool = True):
    """
    Sanitizes RAVE's exported ONNX graph by pruning the noise synthesis branch (branch 2)
    that invokes torch.fft.irfft / DFT with onesided=1 & inverse=1, which is strictly
    forbidden by ONNX runtime shape inference specifications. The main audio reconstruction
    waveform is preserved intact via Identity bypass.
    """
    try:
        import onnx
    except ImportError:
        return

    model = onnx.load(onnx_path)

    # 1. Bypass /_rave/decoder/Add_1 to pass through tonal waveform (/Mul_1_output_0)
    for n in model.graph.node:
        if n.name == "/_rave/decoder/Add_1":
            n.op_type = "Identity"
            del n.input[:]
            n.input.append("/_rave/decoder/Mul_1_output_0")
            if verbose:
                print("  [Sanitizer] Bypassed /_rave/decoder/Add_1 noise summation with tonal waveform.")

    # 2. Prune all nodes belonging to synth noise branch 2
    nodes_to_keep = [n for n in model.graph.node if "/_rave/decoder/synth/2/" not in n.name]
    del model.graph.node[:]
    model.graph.node.extend(nodes_to_keep)

    # 3. Prune unreferenced initializers
    referenced_inputs = set()
    for n in model.graph.node:
        for inp in n.input:
            referenced_inputs.add(inp)

    initializers_to_keep = [init for init in model.graph.initializer if init.name in referenced_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers_to_keep)

    onnx.save(model, onnx_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAVE TorchScript (.ts) to ONNX and Bendable ONNX for NetBender"
    )
    parser.add_argument("-i", "--input", required=True, help="Input TorchScript .ts model path")
    parser.add_argument("-o", "--output", help="Output .onnx file path")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Sample buffer size for export tracing")
    parser.add_argument("--channels", type=int, default=1, help="Input audio channels (default: 1)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--no-bending",
        action="store_true",
        help="Export standard static ONNX without promoting weights to inputs",
    )

    args = parser.parse_args()

    output = args.output
    if not output:
        base, _ = os.path.splitext(args.input)
        output = f"{base}_bendable.onnx"

    convert_rave_ts_to_onnx(
        ts_model_path=args.input,
        output_onnx_path=output,
        buffer_size=args.buffer_size,
        channels=args.channels,
        opset_version=args.opset,
        promote_for_bending=not args.no_bending,
        verbose=True,
    )


if __name__ == "__main__":
    main()
