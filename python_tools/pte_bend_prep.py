#!/usr/bin/env python3
"""
pte_bend_prep.py - NetBender ExecuTorch (.pte) Parameter Promotion & Export Tool

Prepares PyTorch nn.Modules for real-time, zero-copy circuit bending in ExecuTorch (.pte)
using Input-Promoted Parameters (`torch.func.functional_call`) and hardware delegates
(`coreml`, `mps`, `xnnpack`, `portable`).

Produces three synchronized artifacts:
1. `<model>.pte`: Compiled ExecuTorch flatbuffer program
2. `<model>.bin`: NetBender `NBND` binary weights sidecar for zero-copy C++ bending
3. `<model>.json`: Model & method metadata + promoted parameter ordering
"""

import argparse
import json
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DELEGATE_SAFE_KEYWORDS = (
    "norm",
    "ln",
    "gn",
    "bn",
    "bias",
    "snake",
    "alpha",
    "beta",
    "gamma",
    "scale",
    "shift",
    "gain",
    "weight_g",
    "act",
)


def is_bendable_parameter(
    name: str,
    param: torch.Tensor,
    bend_mode: str = "delegate",
    layer_filter: Optional[List[str]] = None,
    min_elements: int = 1,
) -> bool:
    """
    Determine whether a named parameter should be promoted to a dynamic graph input.

    - bend_mode == "none": promote nothing (static graph)
    - bend_mode == "full": promote all floating-point parameters (best with xnnpack/portable)
    - bend_mode == "delegate": promote hardware-delegate-safe parameters (norms, biases,
      activations/snake alpha/beta, scales, weight_g) so heavy conv/matmul weights remain
      compile-time constants inside CoreML / ANE / XNNPACK subgraphs.
    """
    if bend_mode == "none":
        return False

    if not param.is_floating_point():
        return False

    if param.numel() < min_elements:
        return False

    if layer_filter:
        return any(f.lower() in name.lower() for f in layer_filter)

    if bend_mode == "full":
        return True

    # Default: "delegate" mode — promote elementwise / normalization / activation / bias / channel-scale tensors
    # while keeping multi-dimensional Conv / ConvTranspose / Linear weight matrices constant inside CoreML/XNNPACK
    non_singleton_dims = sum(1 for d in param.shape if d > 1)
    if non_singleton_dims > 1:
        return False

    lower_name = name.lower()
    if any(k in lower_name for k in DELEGATE_SAFE_KEYWORDS):
        return True

    # Also promote 1D or channelwise parameter vectors (e.g., per-channel scales, biases, snake alpha/beta)
    if non_singleton_dims <= 1 and param.numel() <= 4096:
        return True

    return False


class FunctionalPromotedWrapper(nn.Module):
    """
    Wraps an existing `nn.Module` so that selected parameters become explicit positional
    `forward(audio_in, *promoted_params)` arguments routed via `torch.func.functional_call`.
    Also decouples dynamic `bias` additions on `Conv1d`/`Conv2d`/`ConvTranspose1d` layers into
    explicit elementwise `+ bias` ops so CoreML/XNNPACK partitioners can compile convolutions
    while keeping biases dynamically bendable at runtime.
    """

    def __init__(
        self,
        base_module: nn.Module,
        promoted_names: List[str],
        method_name: str = "forward",
    ):
        super().__init__()
        self.base_module = base_module
        self.promoted_names = list(promoted_names)
        self.method_name = method_name
        self._decouple_conv_biases(self.base_module)

    @staticmethod
    def _decouple_conv_biases(root_mod: nn.Module) -> None:
        for sub in root_mod.modules():
            if isinstance(sub, nn.Conv1d):
                def _conv1d_fwd(self_mod, inp):
                    out = F.conv1d(
                        inp, self_mod.weight, None, self_mod.stride,
                        self_mod.padding, self_mod.dilation, self_mod.groups
                    )
                    if self_mod.bias is not None:
                        out = out + self_mod.bias.view(1, -1, 1)
                    return out
                sub.forward = types.MethodType(_conv1d_fwd, sub)
            elif isinstance(sub, nn.Conv2d):
                def _conv2d_fwd(self_mod, inp):
                    out = F.conv2d(
                        inp, self_mod.weight, None, self_mod.stride,
                        self_mod.padding, self_mod.dilation, self_mod.groups
                    )
                    if self_mod.bias is not None:
                        out = out + self_mod.bias.view(1, -1, 1, 1)
                    return out
                sub.forward = types.MethodType(_conv2d_fwd, sub)
            elif isinstance(sub, nn.ConvTranspose1d):
                def _conv_t1d_fwd(self_mod, inp, output_size=None):
                    out_pad = self_mod._output_padding(
                        inp, output_size, self_mod.stride, self_mod.padding,
                        self_mod.kernel_size, self_mod.num_spatial_dims, self_mod.dilation
                    )
                    out = F.conv_transpose1d(
                        inp, self_mod.weight, None, self_mod.stride,
                        self_mod.padding, out_pad, self_mod.groups, self_mod.dilation
                    )
                    if self_mod.bias is not None:
                        out = out + self_mod.bias.view(1, -1, 1)
                    return out
                sub.forward = types.MethodType(_conv_t1d_fwd, sub)

    def forward(self, x: torch.Tensor, *promoted_tensors: torch.Tensor) -> torch.Tensor:
        param_override: Dict[str, torch.Tensor] = {}
        for name, tensor in zip(self.promoted_names, promoted_tensors):
            param_override[name] = tensor

        if self.method_name == "forward":
            return torch.func.functional_call(self.base_module, param_override, (x,))
        else:
            # Temporarily bind forward to the target method for functional_call
            orig_forward = self.base_module.forward
            try:
                self.base_module.forward = getattr(self.base_module, self.method_name)
                return torch.func.functional_call(self.base_module, param_override, (x,))
            finally:
                self.base_module.forward = orig_forward


def select_promoted_parameters(
    module: nn.Module,
    bend_mode: str = "delegate",
    layer_filter: Optional[List[str]] = None,
) -> Tuple[List[str], List[torch.Tensor]]:
    """Returns `(promoted_names, promoted_tensors)` in deterministic model order."""
    promoted_names: List[str] = []
    promoted_tensors: List[torch.Tensor] = []

    for name, param in module.named_parameters():
        if is_bendable_parameter(name, param, bend_mode=bend_mode, layer_filter=layer_filter):
            promoted_names.append(name)
            promoted_tensors.append(param.detach().clone().float().contiguous())

    # If "delegate" mode matched nothing on a very simple conv-only network, fall back to all params
    if bend_mode == "delegate" and len(promoted_names) == 0:
        for name, param in module.named_parameters():
            if param.is_floating_point() and param.numel() >= 1:
                promoted_names.append(name)
                promoted_tensors.append(param.detach().clone().float().contiguous())

    return promoted_names, promoted_tensors


def write_nbnd_sidecar(
    sidecar_bin_path: str,
    promoted_names: List[str],
    promoted_tensors: List[torch.Tensor],
) -> None:
    """Writes the NetBender `NBND` v1 binary weights sidecar."""
    os.makedirs(os.path.dirname(os.path.abspath(sidecar_bin_path)), exist_ok=True)
    with open(sidecar_bin_path, "wb") as f:
        f.write(b"NBND\x01\x00\x00\x00")
        f.write(struct.pack("<I", len(promoted_names)))
        for name, tensor in zip(promoted_names, promoted_tensors):
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(arr.shape)))
            for dim in arr.shape:
                f.write(struct.pack("<q", int(dim)))
            f.write(struct.pack("<Q", int(arr.size)))
            f.write(arr.tobytes())


def get_partitioner_for_delegate(delegate: str):
    """Returns the appropriate ExecuTorch partitioner list for the requested hardware delegate."""
    delegate = delegate.lower().strip()
    if delegate == "portable":
        return []

    if delegate == "xnnpack":
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            return [XnnpackPartitioner()]
        except ImportError as e:
            raise ImportError(
                f"ExecuTorch XNNPACK partitioner not available ({e}). Install executorch with XNNPACK support."
            )

    if delegate == "coreml":
        try:
            from executorch.backends.apple.coreml.partition import CoreMLPartitioner
            return [CoreMLPartitioner()]
        except ImportError as e:
            raise ImportError(
                f"ExecuTorch CoreML partitioner not available ({e}). Install coremltools and executorch CoreML backend."
            )

    if delegate == "mps":
        try:
            from executorch.backends.apple.mps.partition import MPSPartitioner
            return [MPSPartitioner()]
        except ImportError as e:
            raise ImportError(
                f"ExecuTorch MPS partitioner not available ({e}). Install executorch MPS backend."
            )

    raise ValueError(f"Unsupported delegate '{delegate}'. Choose from: portable, xnnpack, coreml, mps.")


def export_bendable_pte(
    module: nn.Module,
    sample_audio_input: torch.Tensor,
    output_pte_path: str,
    delegate: str = "coreml",
    bend_mode: str = "delegate",
    layer_filter: Optional[List[str]] = None,
    method_name: str = "forward",
    companion_modules: Optional[Dict[str, Tuple[nn.Module, torch.Tensor]]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    write_sidecar: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Exports a PyTorch `nn.Module` (and optional companion methods like `encode` and `decode`)
    to an ExecuTorch `.pte` file with input-promoted bendable parameters, alongside
    `<base>.bin` (`NBND` weights sidecar) and `<base>.json` metadata.
    """
    try:
        from executorch.exir import to_edge, to_edge_transform_and_lower
    except ImportError:
        raise ImportError(
            "The 'executorch' Python package is required to export .pte models. "
            "Install with: pip install executorch"
        )

    module = module.eval().cpu()
    sample_audio_input = sample_audio_input.detach().cpu().float().contiguous()

    promoted_names, promoted_tensors = select_promoted_parameters(
        module, bend_mode=bend_mode, layer_filter=layer_filter
    )

    if verbose:
        print(
            f"[ExecuTorch] Preparing '{method_name}' | delegate={delegate} | "
            f"bend_mode={bend_mode} | promoted_layers={len(promoted_names)}"
        )

    if len(promoted_names) > 0:
        wrapped = FunctionalPromotedWrapper(module, promoted_names, method_name=method_name).eval()
        example_args = (sample_audio_input, *promoted_tensors)
    else:
        if method_name == "forward":
            wrapped = module
        else:
            wrapped = FunctionalPromotedWrapper(module, [], method_name=method_name).eval()
        example_args = (sample_audio_input,)

    with torch.no_grad():
        exported_program = torch.export.export(wrapped, example_args, strict=False)

    safe_to_orig = getattr(module, "safe_to_orig", {})
    display_names = [safe_to_orig.get(n, n) for n in promoted_names]
    master_display_names: List[str] = list(display_names)
    master_promoted_tensors: List[torch.Tensor] = list(promoted_tensors)
    master_name_to_tensor: Dict[str, torch.Tensor] = {
        dn: pt for dn, pt in zip(display_names, promoted_tensors)
    }
    method_param_inputs: Dict[str, List[str]] = {method_name: list(display_names)}
    programs_dict: Dict[str, Any] = {method_name: exported_program}

    if companion_modules:
        for comp_name, (comp_mod, comp_sample_in) in companion_modules.items():
            try:
                comp_mod = comp_mod.eval().cpu()
                comp_sample_in = comp_sample_in.detach().cpu().float().contiguous()
                c_names, c_tensors = select_promoted_parameters(
                    comp_mod, bend_mode=bend_mode, layer_filter=layer_filter
                )
                c_safe_to_orig = getattr(comp_mod, "safe_to_orig", {})
                c_display_names: List[str] = []
                for cn, ct in zip(c_names, c_tensors):
                    dn = c_safe_to_orig.get(cn, cn)
                    if dn in master_name_to_tensor:
                        existing = master_name_to_tensor[dn]
                        if existing.shape != ct.shape or not torch.allclose(existing, ct, atol=1e-5):
                            dn = f"{comp_name}.{dn}"
                    if dn not in master_name_to_tensor:
                        master_name_to_tensor[dn] = ct
                        master_display_names.append(dn)
                        master_promoted_tensors.append(ct)
                    c_display_names.append(dn)

                if verbose:
                    print(
                        f"[ExecuTorch] Preparing '{comp_name}' | delegate={delegate} | "
                        f"bend_mode={bend_mode} | promoted_layers={len(c_names)}"
                    )

                if len(c_names) > 0:
                    c_wrapped = FunctionalPromotedWrapper(comp_mod, c_names, method_name="forward").eval()
                    c_args = (comp_sample_in, *c_tensors)
                else:
                    c_wrapped = comp_mod
                    c_args = (comp_sample_in,)

                with torch.no_grad():
                    c_ep = torch.export.export(c_wrapped, c_args, strict=False)
                programs_dict[comp_name] = c_ep
                method_param_inputs[comp_name] = c_display_names
            except Exception as comp_err:
                if verbose:
                    print(f"[ExecuTorch] Warning: skipping companion method '{comp_name}': {comp_err}")

    base_path, _ = os.path.splitext(output_pte_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_pte_path)), exist_ok=True)

    # Note: ExecuTorch's Apple CoreML runtime rejects multifunction CoreML .pte programs
    # unless built with macOS 15.0+ deployment target. For CoreML, export companion methods
    # to separate single-method `<base>_encode.pte` / `<base>_decode.pte` files, which
    # ExecuTorchBackend::load automatically discovers and binds.
    if delegate.lower().strip() == "coreml" and len(programs_dict) > 1:
        fwd_partitioners = get_partitioner_for_delegate(delegate)
        fwd_edge = (
            to_edge_transform_and_lower(exported_program, partitioner=fwd_partitioners)
            if fwd_partitioners
            else to_edge(exported_program)
        )
        with open(output_pte_path, "wb") as f:
            fwd_edge.to_executorch().write_to_file(f)

        for comp_name, comp_ep in programs_dict.items():
            if comp_name == method_name:
                continue
            comp_pte_path = f"{base_path}_{comp_name}.pte"
            comp_partitioners = get_partitioner_for_delegate(delegate)
            comp_edge = (
                to_edge_transform_and_lower(comp_ep, partitioner=comp_partitioners)
                if comp_partitioners
                else to_edge(comp_ep)
            )
            with open(comp_pte_path, "wb") as cf:
                comp_edge.to_executorch().write_to_file(cf)
            if verbose:
                print(f"[ExecuTorch] Saved companion '{comp_name}' .pte to: {comp_pte_path}")
    else:
        partitioners = get_partitioner_for_delegate(delegate)
        export_target = programs_dict if len(programs_dict) > 1 else exported_program
        if partitioners:
            edge_program = to_edge_transform_and_lower(export_target, partitioner=partitioners)
        else:
            edge_program = to_edge(export_target)

        et_program = edge_program.to_executorch()
        with open(output_pte_path, "wb") as f:
            et_program.write_to_file(f)

    if write_sidecar and len(master_display_names) > 0:
        sidecar_bin_path = f"{base_path}.bin"
        write_nbnd_sidecar(sidecar_bin_path, master_display_names, master_promoted_tensors)
        if verbose:
            print(f"[ExecuTorch] Saved NBND weights sidecar to: {sidecar_bin_path}")

    meta_dict: Dict[str, Any] = {
        "netbender.bendable": "1" if len(master_display_names) > 0 else "0",
        "netbender.delegate": delegate,
        "netbender.bend_mode": bend_mode,
        "netbender.promoted_layers_count": len(master_display_names),
        "promoted_inputs": master_display_names,
        "method_param_inputs": method_param_inputs,
    }
    if extra_metadata:
        meta_dict.update(extra_metadata)
    meta_dict["methods"] = list(programs_dict.keys())
    meta_dict["method_param_inputs"] = method_param_inputs

    if write_sidecar:
        json_path = f"{base_path}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(meta_dict, jf, indent=2)
        if verbose:
            print(f"[ExecuTorch] Saved metadata JSON to: {json_path}")

    if verbose:
        print(f"[ExecuTorch] Saved .pte model to: {output_pte_path}")

    return meta_dict

