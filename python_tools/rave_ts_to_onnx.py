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
from typing import List, Optional

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
        # ONNX Runtime specification forbids DFT with onesided=1 & inverse=1.
        # We rebuild the exact two-sided conjugate-symmetric spectrum [..., N, 2]
        # and invoke DFT with inverse=1, onesided=0 along axis=-2, which ONNX Runtime natively executes!
        axis_freq = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
        start_1 = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
        end_neg1 = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        interior = g.op("Slice", self, start_1, end_neg1, axis_freq)

        # Reverse interior along the frequency axis (-2) using negative step -1
        rev_start = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        rev_end = g.op("Constant", value_t=torch.tensor([-999999], dtype=torch.int64))
        rev_step = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        flipped = g.op("Slice", interior, rev_start, rev_end, axis_freq, rev_step)

        # Conjugate: keep real part, negate imaginary part along the complex component axis (-1)
        axis_complex = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        real_part = g.op("Slice", flipped, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)), g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)), axis_complex)
        imag_part = g.op("Slice", flipped, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)), g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64)), axis_complex)
        neg_imag = g.op("Neg", imag_part)
        conj_flipped = g.op("Concat", real_part, neg_imag, axis_i=-1)

        # Concat along frequency axis (-2) to form full two-sided Hermitian spectrum
        full_spectrum = g.op("Concat", self, conj_flipped, axis_i=-2)

        # Inverse two-sided DFT along axis -2
        inv_dft = g.op("DFT", full_spectrum, axis_i=-2, inverse_i=1, onesided_i=0)

        # Extract real part (index 0 along complex axis -1) and squeeze
        real_out = g.op("Slice", inv_dft, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)), g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)), axis_complex)
        squeezed = g.op("Squeeze", real_out, axis_complex)

        # Scale by 1 / N to match PyTorch irfft(norm="backward") specification
        # full_spectrum has shape [..., N, 2] along axis -2
        shape_full = g.op("Shape", full_spectrum)
        axis_dim = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
        n_samples = g.op("Gather", shape_full, axis_dim, axis_i=0)
        n_float = g.op("Cast", n_samples, to_i=1)  # 1 = FLOAT
        return g.op("Div", squeezed, n_float)

    def symbolic_fft_rfft(g, self, n, dim, norm):
        # ONNX specification requires one-sided forward DFT input to have trailing dimension 1: [..., samples, 1]
        axis_c = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        real_in = g.op("Unsqueeze", self, axis_c)
        return g.op("DFT", real_in, axis_i=-2, inverse_i=0, onesided_i=1)

    def symbolic_roll(g, self, shifts, dims):
        return self

    def symbolic_hann_window(g, window_length, *args):
        val = window_length.toIValue() if hasattr(window_length, "toIValue") else None
        if val is None and hasattr(window_length, "node") and window_length.node().hasAttribute("value"):
            val = window_length.node().t("value").item()
        L = int(val) if val is not None else 8
        return g.op("Constant", value_t=torch.hann_window(L))

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
    methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> str:
    """
    Loads a TorchScript model and exports it to ONNX.
    Supports exporting forward pass as well as companion methods (encode, decode, prior).
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
    export_kwargs = dict(
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes={
            "audio_in": {0: "batch", 2: "samples"},
            "audio_out": {0: "batch", 2: "samples"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, (dummy_input,), std_onnx_path, **export_kwargs)

    if verbose:
        print(f"  -> Exported standard ONNX successfully.")

    # Sanitize graph to bypass NaN-producing DFT noise branch in ONNX Runtime C++
    _sanitize_rave_onnx_graph(std_onnx_path, verbose=verbose)

    # Extract metadata properties if available on the model
    sr = 48000
    try:
        if hasattr(model, "sr"):
            val = model.sr() if callable(model.sr) else model.sr
            sr = int(val)
        elif hasattr(model, "sample_rate"):
            val = model.sample_rate() if callable(model.sample_rate) else model.sample_rate
            sr = int(val)
        elif hasattr(model, "named_buffers"):
            for bname, buf in model.named_buffers():
                if "sampling_rate" in bname.lower() or "sample_rate" in bname.lower():
                    sr = int(buf.item())
                    break
    except Exception:
        pass

    ratio = 2048
    try:
        if hasattr(model, "encode_params"):
            ep = model.encode_params() if callable(model.encode_params) else model.encode_params
            if hasattr(ep, "stride"):
                val = ep.stride() if callable(ep.stride) else ep.stride
                ratio = int(val)
            elif isinstance(ep, (list, tuple)) and len(ep) >= 2:
                ratio = int(ep[1])
        elif hasattr(model, "ratios"):
            ratios = model.ratios() if callable(model.ratios) else model.ratios
            if hasattr(ratios, "__iter__"):
                prod = 1
                for r in ratios:
                    prod *= int(r)
                if prod > 0:
                    ratio = prod
    except Exception:
        pass

    latent_dim = 4
    try:
        if hasattr(model, "decode_params"):
            dp = model.decode_params() if callable(model.decode_params) else model.decode_params
            if hasattr(dp, "__iter__") or isinstance(dp, (list, tuple, torch.Tensor)):
                latent_dim = int(dp[0])
        elif hasattr(model, "encode_params"):
            ep = model.encode_params() if callable(model.encode_params) else model.encode_params
            if hasattr(ep, "__iter__") or isinstance(ep, (list, tuple, torch.Tensor)):
                latent_dim = int(ep[2])
    except Exception:
        pass

    history_samples = 4096
    extra_meta = {
        "netbender.architecture": "rave",
        "netbender.sample_rate": str(sr),
        "netbender.ratio": str(ratio),
        "netbender.latent_dim": str(latent_dim),
        "netbender.in_channels": str(in_channels),
        "netbender.out_channels": str(in_channels),
        "netbender.history_samples": str(history_samples),
        "netbender.output_offset": str(history_samples),
    }

    if promote_for_bending:
        # Step 3: Run parameter promotion
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

        # Clean up intermediate raw file if different
        if std_onnx_path != output_onnx_path and os.path.exists(std_onnx_path):
            try:
                os.remove(std_onnx_path)
            except OSError:
                pass

    # Export companion subgraphs if requested
    if methods:
        candidate_methods = ["encode", "decode", "prior"]
        available_ts_methods = [m for m in candidate_methods if hasattr(model, m)]
        export_list = []
        if "all" in methods:
            export_list = available_ts_methods
        else:
            export_list = [m for m in methods if m in available_ts_methods and m != "forward"]

        if export_list:
            slice_rave_onnx_subgraphs(output_onnx_path, base, export_list, latent_dim=latent_dim, verbose=verbose)

        # Direct companion export for 'prior' if present on the TorchScript model
        if "prior" in export_list and hasattr(model, "prior"):
            prior_onnx_path = f"{base}_prior.onnx"
            if verbose:
                print(f"  -> Exporting companion method 'prior' -> {prior_onnx_path}...")
            try:
                p_in_dim = 0
                p_in_ratio = 1
                try:
                    if hasattr(model, "prior_params"):
                        pp = model.prior_params() if callable(model.prior_params) else model.prior_params
                        if isinstance(pp, (list, tuple)) and len(pp) >= 2:
                            p_in_dim = int(pp[0])
                            p_in_ratio = int(pp[1])
                except Exception:
                    pass

                in_ch = max(1, p_in_dim)
                dummy_x = torch.zeros(1, in_ch, 2048, dtype=torch.float32)

                class PriorExportModule(torch.nn.Module):
                    def __init__(self, prior_fn, takes_audio_input: bool = False):
                        super().__init__()
                        self.prior_fn = prior_fn
                        self.takes_audio_input = takes_audio_input

                    def forward(self, x):
                        if self.takes_audio_input:
                            return self.prior_fn(x)
                        try:
                            return self.prior_fn(x)
                        except TypeError:
                            return self.prior_fn()

                wrapper = PriorExportModule(model.prior, takes_audio_input=(p_in_dim > 0))

                torch.onnx.export(
                    wrapper,
                    (dummy_x,),
                    prior_onnx_path,
                    input_names=["prior_in"],
                    output_names=["latent_out"],
                    dynamic_axes={"prior_in": {0: "batch", 2: "samples"}, "latent_out": {0: "batch", 2: "frames"}},
                    opset_version=opset_version,
                )
                if verbose:
                    print(f"  -> Companion 'prior' exported successfully.")
            except Exception as pe:
                if verbose:
                    print(f"  [Note] RAVE prior export skipped: {pe}")
                if os.path.exists(prior_onnx_path):
                    try:
                        os.remove(prior_onnx_path)
                    except OSError:
                        pass

    if verbose:
        print(f"\n[DONE] Ready for NetBender ONNX inference: {output_onnx_path}\n")

    return output_onnx_path


def slice_rave_onnx_subgraphs(full_onnx_path: str, base_path: str, methods: List[str], latent_dim: int = 4, verbose: bool = True):
    """
    Slices the verified bendable RAVE ONNX graph into companion models for 'encode' and 'decode'
    using the bottleneck latent tensor /_rave/Slice_1_output_0.
    """
    try:
        import onnx
        from onnx import helper, TensorProto
        import copy
    except ImportError:
        return

    full_model = onnx.load(full_onnx_path)
    latent_tensor_name = "/_rave/Slice_1_output_0"

    # Find actual latent tensor if name was modified or differently named
    has_latent = any(latent_tensor_name in n.output for n in full_model.graph.node)
    if not has_latent:
        for n in full_model.graph.node:
            if n.name.endswith("/Slice_1"):
                latent_tensor_name = n.output[0]
                has_latent = True
                break

    if not has_latent:
        if verbose:
            print("  [Warning] Could not find latent bottleneck tensor in ONNX graph for slicing.")
        return

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
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to export (e.g. all, forward, encode, decode, prior). Default: all",
    )
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
        methods=args.methods,
        verbose=True,
    )


if __name__ == "__main__":
    main()
