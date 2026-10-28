#!/usr/bin/env python3
"""
migrate_ts_to_pte.py - Migrate TorchScript (.ts) or ONNX (.onnx) audio models to ExecuTorch (.pte)

Inspired by `neural_tilde.migrate`, extended with NetBender's Input-Promoted Parameter
circuit-bending architecture (`NBND` `.bin` sidecar + `.json` metadata).

Works with all NetBender models (`darbouka.ts`, `speech.ts`, `encodec_bendable.ts`,
`stable_audio_1_vae.ts`, or their `.onnx` counterparts) by lowering through a sanitized ONNX
graph into an eager `torch.nn.Module` that `torch.export.export()` and ExecuTorch delegates
(`coreml`, `mps`, `xnnpack`, `portable`) can compile.

Usage:
    python -m python_tools.migrate_ts_to_pte models/darbouka.ts -o models/darbouka_bendable.pte --delegate coreml --bend-mode delegate
"""

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from python_tools.pte_bend_prep import export_bendable_pte


class OnnxGraphEagerModule(nn.Module):
    """
    Reconstructs an eager `torch.nn.Module` directly from an ONNX ModelProto.
    - All floating-point initializers are registered as `nn.Parameter` (preserving original names
      via a parameter map so `torch.func.functional_call` can override any weight by name).
    - Non-float initializers (indices, shapes, axes) are registered as buffers or constants.
    - Executes standard ONNX operators using pure ATen / `torch.nn.functional` primitives so
      `torch.export.export()` can trace the full computation graph cleanly.
    """

    def __init__(self, onnx_model, sample_input: Optional[torch.Tensor] = None):
        super().__init__()
        from onnx import numpy_helper

        self.graph = onnx_model.graph
        self.orig_to_safe: Dict[str, str] = {}
        self.safe_to_orig: Dict[str, str] = {}
        self.constants: Dict[str, Any] = {}
        self.const_buffers: Dict[str, str] = {}

        for init in self.graph.initializer:
            arr = numpy_helper.to_array(init).copy()
            t = torch.from_numpy(arr)
            if t.is_floating_point():
                safe_name = self._sanitize_name(init.name)
                is_cache_or_const = ("cache" in init.name.lower()) or (
                    init.name.startswith("onnx::") and not any(k in init.name for k in ("Conv", "LSTM", "MatMul", "Gemm"))
                )
                if is_cache_or_const:
                    self.register_buffer(safe_name, t.float().contiguous())
                    self.const_buffers[init.name] = safe_name
                else:
                    self.orig_to_safe[init.name] = safe_name
                    self.safe_to_orig[safe_name] = init.name
                    self.register_parameter(safe_name, nn.Parameter(t.float().contiguous()))
            else:
                self.constants[init.name] = arr

        self.nodes = []
        for node in self.graph.node:
            attrs = self._parse_attrs(node)
            if node.op_type == "Constant" and "value" in attrs:
                val = attrs["value"]
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    safe_c = f"_cnode_{self._sanitize_name(node.output[0])}"
                    self.register_buffer(safe_c, val.float().contiguous())
                    self.const_buffers[node.output[0]] = safe_c
                elif isinstance(val, torch.Tensor):
                    self.constants[node.output[0]] = val.detach().cpu().numpy()
                else:
                    self.constants[node.output[0]] = val
            else:
                self.nodes.append((node.op_type, list(node.input), list(node.output), attrs))

        # Primary input/output names (excluding promoted initializers)
        init_names = set(self.orig_to_safe.keys()) | set(self.constants.keys()) | set(self.const_buffers.keys())
        self.primary_inputs = [
            inp.name for inp in self.graph.input if inp.name not in init_names
        ]
        self.primary_outputs = [out.name for out in self.graph.output]
        self.active_nodes = list(self.nodes)

        if sample_input is not None:
            self.precompute_static_subgraphs(sample_input)

    @staticmethod
    def _to_int_list(val: Any) -> List[int]:
        if isinstance(val, np.ndarray):
            return [int(v) for v in val.reshape(-1).tolist()]
        if isinstance(val, (list, tuple)):
            return [int(v) for v in val]
        if isinstance(val, (int, np.integer)):
            return [int(val)]
        return [int(v) for v in val.detach().cpu().numpy().reshape(-1).tolist()]

    @staticmethod
    def _to_scalar_int(val: Any) -> int:
        if isinstance(val, np.ndarray):
            return int(val.reshape(-1)[0])
        if isinstance(val, (int, np.integer)):
            return int(val)
        return int(val.item())

    @staticmethod
    def _to_scalar_float(val: Any) -> float:
        if isinstance(val, np.ndarray):
            return float(val.reshape(-1)[0])
        if isinstance(val, (float, int, np.number)):
            return float(val)
        return float(val.item())

    @staticmethod
    def _ensure_tensor(val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val)
        return val

    @torch.no_grad()
    def precompute_static_subgraphs(self, sample_input: torch.Tensor) -> None:
        """
        Folds all shape-only and constant subgraphs (Shape, ConstantOfShape, Range, ScatterND,
        Where, Equal, and integer index arithmetic) into concrete NumPy constants or buffers for
        the target buffer size so `torch.export.export()` traces a clean, static-shape ATen graph.
        """
        env: Dict[str, Any] = {}
        is_dynamic: Dict[str, bool] = {}

        if self.primary_inputs:
            env[self.primary_inputs[0]] = sample_input
            is_dynamic[self.primary_inputs[0]] = True
        for orig_name, safe_name in self.orig_to_safe.items():
            env[orig_name] = getattr(self, safe_name)
            is_dynamic[orig_name] = True
        for orig_name, buf_name in self.const_buffers.items():
            env[orig_name] = getattr(self, buf_name)
            is_dynamic[orig_name] = False
        for cname, cval in self.constants.items():
            env[cname] = cval
            is_dynamic[cname] = False

        active = []
        for op, inputs, outputs, attrs in self.nodes:
            args = [env.get(inp, None) if inp != "" else None for inp in inputs]
            out = self._exec_node(op, args, attrs)
            out_list = [out] if len(outputs) == 1 else list(out)
            for o_name, o_val in zip(outputs, out_list):
                if o_name:
                    env[o_name] = o_val

            node_is_static = (op == "Shape") or not any(
                is_dynamic.get(inp, False) for inp in inputs if inp != ""
            )
            if node_is_static:
                for o_name, o_val in zip(outputs, out_list):
                    if not o_name or o_val is None:
                        continue
                    is_dynamic[o_name] = False
                    if isinstance(o_val, torch.Tensor) and o_val.is_floating_point():
                        buf_name = f"_cbuf_{self._sanitize_name(o_name)}"
                        self.register_buffer(buf_name, o_val.detach().clone().float().contiguous())
                        self.const_buffers[o_name] = buf_name
                    elif isinstance(o_val, torch.Tensor):
                        self.constants[o_name] = o_val.detach().cpu().numpy()
                    else:
                        self.constants[o_name] = o_val
            else:
                for o_name in outputs:
                    if o_name:
                        is_dynamic[o_name] = True
                active.append((op, inputs, outputs, attrs))

        self.active_nodes = active

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace(".", "__dot__").replace("/", "__slash__").replace(":", "__colon__")

    @staticmethod
    def _parse_attrs(node) -> Dict[str, Any]:
        from onnx import numpy_helper

        attrs: Dict[str, Any] = {}
        for a in node.attribute:
            if a.type == 1:  # FLOAT
                attrs[a.name] = float(a.f)
            elif a.type == 2:  # INT
                attrs[a.name] = int(a.i)
            elif a.type == 3:  # STRING
                attrs[a.name] = a.s.decode("utf-8", errors="ignore")
            elif a.type == 4:  # TENSOR
                arr = numpy_helper.to_array(a.t).copy()
                attrs[a.name] = torch.from_numpy(arr)
            elif a.type == 6:  # FLOATS
                attrs[a.name] = [float(x) for x in a.floats]
            elif a.type == 7:  # INTS
                attrs[a.name] = [int(x) for x in a.ints]
        return attrs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        env: Dict[str, Any] = {}
        if self.primary_inputs:
            env[self.primary_inputs[0]] = x
        for orig_name, safe_name in self.orig_to_safe.items():
            env[orig_name] = getattr(self, safe_name)
        for cname, cval in self.constants.items():
            env[cname] = cval
        for orig_name, buf_name in self.const_buffers.items():
            env[orig_name] = getattr(self, buf_name)

        for op, inputs, outputs, attrs in self.active_nodes:
            args = [env.get(inp, None) if inp != "" else None for inp in inputs]
            out = self._exec_node(op, args, attrs)
            if len(outputs) == 1:
                env[outputs[0]] = out
            else:
                for o_name, o_val in zip(outputs, out):
                    if o_name:
                        env[o_name] = o_val

        return env[self.primary_outputs[0]]

    def _exec_node(self, op: str, args: List[Any], attrs: Dict[str, Any]) -> Any:
        if op == "Identity":
            return self._ensure_tensor(args[0])
        if op == "Add":
            return self._ensure_tensor(args[0]) + self._ensure_tensor(args[1])
        if op == "Sub":
            return self._ensure_tensor(args[0]) - self._ensure_tensor(args[1])
        if op == "Mul":
            return self._ensure_tensor(args[0]) * self._ensure_tensor(args[1])
        if op == "Div":
            return self._ensure_tensor(args[0]) / self._ensure_tensor(args[1])
        if op == "Neg":
            return -self._ensure_tensor(args[0])
        if op == "Sin":
            return torch.sin(args[0])
        if op == "Cos":
            return torch.cos(args[0])
        if op == "Tanh":
            return torch.tanh(args[0])
        if op == "Sigmoid":
            return torch.sigmoid(args[0])
        if op == "Relu":
            return F.relu(args[0])
        if op == "Elu":
            return F.elu(args[0], alpha=attrs.get("alpha", 1.0))
        if op == "LeakyRelu":
            return F.leaky_relu(args[0], negative_slope=attrs.get("alpha", 0.01))
        if op == "Pow":
            return torch.pow(self._ensure_tensor(args[0]), self._ensure_tensor(args[1]))
        if op == "Sqrt":
            return torch.sqrt(args[0])
        if op == "Exp":
            return torch.exp(args[0])
        if op == "Reciprocal":
            return torch.reciprocal(args[0])
        if op == "Abs":
            return torch.abs(args[0])
        if op == "Clip":
            lo = self._to_scalar_float(args[1]) if len(args) > 1 and args[1] is not None else attrs.get("min", None)
            hi = self._to_scalar_float(args[2]) if len(args) > 2 and args[2] is not None else attrs.get("max", None)
            return torch.clamp(args[0], min=lo, max=hi)
        if op == "Conv":
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else None
            strides = tuple(attrs.get("strides", [1]))
            dilations = tuple(attrs.get("dilations", [1]))
            group = int(attrs.get("group", 1))
            pads = attrs.get("pads", [0, 0])
            if len(pads) == 2:
                if pads[0] == pads[1]:
                    padding = (pads[0],)
                else:
                    x = F.pad(x, (pads[0], pads[1]))
                    padding = (0,)
            elif len(pads) == 4:
                if pads[0] == pads[2] and pads[1] == pads[3]:
                    padding = (pads[0], pads[1])
                else:
                    x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]))
                    padding = (0, 0)
            else:
                padding = (0,)
            if w.ndim == 3:
                res = F.conv1d(x, w, None, stride=strides, padding=padding, dilation=dilations, groups=group)
                if b is not None:
                    res = res + b.view(1, -1, 1)
                return res
            res = F.conv2d(x, w, None, stride=strides, padding=padding, dilation=dilations, groups=group)
            if b is not None:
                res = res + b.view(1, -1, 1, 1)
            return res
        if op == "ConvTranspose":
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else None
            strides = tuple(attrs.get("strides", [1]))
            dilations = tuple(attrs.get("dilations", [1]))
            output_padding = tuple(attrs.get("output_padding", [0]))
            group = int(attrs.get("group", 1))
            pads = attrs.get("pads", [0, 0])
            padding = (pads[0],) if len(pads) == 2 and pads[0] == pads[1] else (0,)
            res = F.conv_transpose1d(
                x, w, None, stride=strides, padding=padding, output_padding=output_padding, groups=group, dilation=dilations
            )
            if b is not None:
                res = res + b.view(1, -1, 1)
            if len(pads) == 2 and pads[0] != pads[1]:
                end = res.shape[-1] - pads[1] if pads[1] > 0 else res.shape[-1]
                res = res[..., pads[0]:end]
            return res
        if op == "Pad":
            x = self._ensure_tensor(args[0])
            pads = self._to_int_list(args[1]) if len(args) > 1 and args[1] is not None else attrs.get("pads", [])
            mode = attrs.get("mode", "constant")
            val = self._to_scalar_float(args[2]) if len(args) > 2 and args[2] is not None else float(attrs.get("value", 0.0))
            n_dims = len(pads) // 2
            torch_pads = []
            for d in reversed(range(n_dims)):
                torch_pads.extend([int(pads[d]), int(pads[d + n_dims])])
            while len(torch_pads) >= 2 and torch_pads[-1] == 0 and torch_pads[-2] == 0 and len(torch_pads) > 2:
                torch_pads = torch_pads[:-2]
            return F.pad(x, tuple(torch_pads), mode=mode, value=val if mode == "constant" else None)
        if op == "Slice":
            x = self._ensure_tensor(args[0])
            starts = self._to_int_list(args[1])
            ends = self._to_int_list(args[2])
            axes = self._to_int_list(args[3]) if len(args) > 3 and args[3] is not None else list(range(len(starts)))
            steps = self._to_int_list(args[4]) if len(args) > 4 and args[4] is not None else [1] * len(starts)
            res = x
            for s, e, ax, st in zip(starts, ends, axes, steps):
                if ax < 0:
                    ax += res.ndim
                dim_len = res.shape[ax]
                if st > 0:
                    if s < 0:
                        s = max(0, dim_len + s)
                    else:
                        s = min(dim_len, s)
                    if e < 0:
                        e = max(0, dim_len + e)
                    elif e > dim_len:
                        e = dim_len
                    if st == 1:
                        res = torch.narrow(res, ax, s, max(0, e - s))
                    else:
                        idx = torch.arange(s, e, st, device=res.device)
                        res = torch.index_select(res, ax, idx)
                else:
                    if s < 0:
                        s = max(-1, dim_len + s)
                    elif s >= dim_len:
                        s = dim_len - 1
                    if e < -dim_len:
                        e = -1
                    elif e < 0:
                        e = dim_len + e
                    elif e > dim_len:
                        e = dim_len
                    if s == dim_len - 1 and e == -1 and st == -1:
                        res = torch.flip(res, dims=[ax])
                    else:
                        idx = torch.arange(s, e, st, device=res.device)
                        res = torch.index_select(res, ax, idx)
            return res
        if op == "Concat":
            axis = int(attrs.get("axis", 0))
            return torch.cat([self._ensure_tensor(a) for a in args if a is not None], dim=axis)
        if op == "Split":
            axis = int(attrs.get("axis", 0))
            split = self._to_int_list(args[1]) if len(args) > 1 and args[1] is not None else attrs.get("split", None)
            if split is not None:
                return torch.split(self._ensure_tensor(args[0]), split, dim=axis)
            num_outputs = int(attrs.get("num_outputs", 2))
            return torch.chunk(self._ensure_tensor(args[0]), num_outputs, dim=axis)
        if op == "Reshape":
            shape = self._to_int_list(args[1])
            t0 = self._ensure_tensor(args[0])
            in_shape = t0.shape
            final_shape = [in_shape[i] if s == 0 and i < len(in_shape) else s for i, s in enumerate(shape)]
            return torch.reshape(t0, final_shape)
        if op == "Transpose":
            perm = attrs.get("perm", None)
            t0 = self._ensure_tensor(args[0])
            return t0.permute(*perm) if perm else t0.t()
        if op == "Squeeze":
            axes = self._to_int_list(args[1]) if len(args) > 1 and args[1] is not None else attrs.get("axes", [])
            res = self._ensure_tensor(args[0])
            for ax in sorted([a if a >= 0 else res.ndim + a for a in axes], reverse=True):
                res = torch.squeeze(res, dim=ax)
            return res
        if op == "Unsqueeze":
            axes = self._to_int_list(args[1]) if len(args) > 1 and args[1] is not None else attrs.get("axes", [0])
            res = self._ensure_tensor(args[0])
            for ax in sorted(axes):
                res = torch.unsqueeze(res, dim=ax)
            return res
        if op == "Shape":
            return torch.tensor(list(args[0].shape), dtype=torch.int64)
        if op == "Gather":
            axis = int(attrs.get("axis", 0))
            x = self._ensure_tensor(args[0])
            idx = self._ensure_tensor(args[1])
            if idx.ndim == 0:
                i = self._to_scalar_int(args[1])
                return x.select(axis, i)
            return torch.index_select(x, axis, idx.reshape(-1).long()).reshape(*x.shape[:axis], *idx.shape, *x.shape[axis + 1:])
        if op == "Cast":
            to_type = int(attrs.get("to", 1))
            t0 = self._ensure_tensor(args[0])
            if to_type == 1:
                return t0.float()
            if to_type in (6, 7):
                return t0.long()
            if to_type == 9:
                return t0.bool()
            return t0
        if op == "ConstantOfShape":
            shape = self._to_int_list(args[0])
            val_t = self._ensure_tensor(attrs.get("value", torch.tensor([0.0], dtype=torch.float32)))
            fill_val = val_t.reshape(-1)[0].item()
            return torch.full(shape, fill_val, dtype=val_t.dtype)
        if op == "Expand":
            shape = self._to_int_list(args[1])
            return self._ensure_tensor(args[0]).expand(*shape)
        if op == "Where":
            return torch.where(self._ensure_tensor(args[0]).bool(), self._ensure_tensor(args[1]), self._ensure_tensor(args[2]))
        if op == "ReduceMean":
            axes = self._to_int_list(args[1]) if len(args) > 1 and args[1] is not None else attrs.get("axes", [-1])
            keepdims = bool(attrs.get("keepdims", 1))
            return torch.mean(self._ensure_tensor(args[0]), dim=axes, keepdim=keepdims)
        if op == "MatMul":
            return torch.matmul(self._ensure_tensor(args[0]), self._ensure_tensor(args[1]))
        if op == "Gemm":
            a, b = self._ensure_tensor(args[0]), self._ensure_tensor(args[1])
            c = self._ensure_tensor(args[2]) if len(args) > 2 and args[2] is not None else None
            if attrs.get("transA", 0):
                a = a.t()
            if attrs.get("transB", 0):
                b = b.t()
            res = attrs.get("alpha", 1.0) * torch.matmul(a, b)
            if c is not None:
                res = res + attrs.get("beta", 1.0) * c
            return res
        if op == "Equal":
            return torch.eq(self._ensure_tensor(args[0]), self._ensure_tensor(args[1]))
        if op == "Range":
            return torch.arange(self._to_scalar_int(args[0]), self._to_scalar_int(args[1]), self._to_scalar_int(args[2]))
        if op == "Softplus":
            return F.softplus(args[0], beta=attrs.get("beta", 1.0), threshold=attrs.get("threshold", 20.0))
        if op == "RandomNormalLike":
            return torch.zeros_like(args[0])
        if op == "Tile":
            repeats = self._to_int_list(args[1])
            return self._ensure_tensor(args[0]).repeat(*repeats)
        if op == "SplitToSequence":
            axis = int(attrs.get("axis", 0))
            keepdims = int(attrs.get("keepdims", 1))
            t0 = self._ensure_tensor(args[0])
            if len(args) > 1 and args[1] is not None:
                split_sizes = self._to_int_list(args[1])
                if len(split_sizes) == 1 and (isinstance(args[1], (int, np.integer)) or getattr(args[1], "ndim", 0) == 0):
                    return list(torch.split(t0, split_sizes[0], dim=axis))
                return list(torch.split(t0, split_sizes, dim=axis))
            chunks = torch.split(t0, 1, dim=axis)
            return [c.squeeze(axis) if not keepdims else c for c in chunks]
        if op == "SequenceAt":
            seq, idx = args[0], self._to_scalar_int(args[1])
            return seq[idx]
        if op == "ScatterND":
            data, indices, updates = self._ensure_tensor(args[0]), self._ensure_tensor(args[1]), self._ensure_tensor(args[2])
            res = data.clone()
            idx_tuple = tuple(indices[..., i].long() for i in range(indices.shape[-1]))
            res[idx_tuple] = updates
            return res
        if op == "LSTM":
            # ONNX LSTM gate order: [i, o, f, c]
            x = args[0]  # [seq_len, batch, input_size]
            w = args[1]  # [num_dir, 4*hidden_size, input_size]
            r = args[2]  # [num_dir, 4*hidden_size, hidden_size]
            b = args[3] if len(args) > 3 else None  # [num_dir, 8*hidden_size]
            h0 = args[5] if len(args) > 5 and args[5] is not None else None
            c0 = args[6] if len(args) > 6 and args[6] is not None else None
            hidden_size = int(attrs.get("hidden_size", r.shape[-1]))
            seq_len, batch_size, _ = x.shape
            h_t = h0[0] if h0 is not None else torch.zeros(batch_size, hidden_size, dtype=x.dtype, device=x.device)
            c_t = c0[0] if c0 is not None else torch.zeros(batch_size, hidden_size, dtype=x.dtype, device=x.device)
            w_0 = w[0].t()  # [input_size, 4*hidden_size]
            r_0 = r[0].t()  # [hidden_size, 4*hidden_size]
            bias_0 = (b[0, : 4 * hidden_size] + b[0, 4 * hidden_size :]) if b is not None else None
            x_proj_all = torch.matmul(x, w_0)
            if bias_0 is not None:
                x_proj_all = x_proj_all + bias_0
            outputs_y = []
            for t in range(seq_len):
                gates = x_proj_all[t] + torch.matmul(h_t, r_0)
                i_g, o_g, f_g, c_g = torch.chunk(gates, 4, dim=-1)
                i_t = torch.sigmoid(i_g)
                f_t = torch.sigmoid(f_g)
                c_cand = torch.tanh(c_g)
                c_t = f_t * c_t + i_t * c_cand
                o_t = torch.sigmoid(o_g)
                h_t = o_t * torch.tanh(c_t)
                outputs_y.append(h_t.unsqueeze(0).unsqueeze(1))
            y_all = torch.cat(outputs_y, dim=0)  # [seq_len, 1, batch, hidden_size]
            return y_all, h_t.unsqueeze(0), c_t.unsqueeze(0)
        raise NotImplementedError(f"Unsupported ONNX op '{op}' in OnnxGraphEagerModule")


def extract_nn_tilde_metadata(ts_mod: torch.jit.ScriptModule, default_buffer_size: int = 2048) -> Dict[str, Any]:
    """Extracts `nn~` / `netBender~` method & attribute metadata from a TorchScript module."""
    methods: List[str] = ["forward", "encode", "decode"]
    if hasattr(ts_mod, "_methods"):
        try:
            for m in getattr(ts_mod, "_methods"):
                if m not in methods:
                    methods.append(m)
        except Exception:
            pass

    sr = 44100
    for sr_attr in ("sr", "_sr", "sample_rate", "sampling_rate"):
        if hasattr(ts_mod, sr_attr):
            try:
                val = getattr(ts_mod, sr_attr)
                sr = int(val() if callable(val) else val)
                break
            except Exception:
                pass

    higher_ratio = 1
    method_props: Dict[str, Any] = {}
    for m in methods:
        params_attr = f"{m}_params"
        if hasattr(ts_mod, params_attr):
            try:
                raw_p = getattr(ts_mod, params_attr)
                p = [int(v) for v in (raw_p() if callable(raw_p) else raw_p)]
                if len(p) >= 4:
                    in_labels = [f"(signal) model input {i}" for i in range(p[0])]
                    out_labels = [f"(signal) model output {i}" for i in range(p[2])]
                    if hasattr(ts_mod, f"{m}_input_labels"):
                        in_labels = [str(x) for x in getattr(ts_mod, f"{m}_input_labels")]
                    if hasattr(ts_mod, f"{m}_output_labels"):
                        out_labels = [str(x) for x in getattr(ts_mod, f"{m}_output_labels")]
                    method_props[m] = {
                        "in_dim": p[0],
                        "in_ratio": p[1],
                        "out_dim": p[2],
                        "out_ratio": p[3],
                        "input_labels": in_labels,
                        "output_labels": out_labels,
                    }
                    higher_ratio = max(higher_ratio, p[1], p[3])
            except Exception:
                pass

    if hasattr(ts_mod, "higher_ratio"):
        try:
            higher_ratio = max(higher_ratio, int(getattr(ts_mod, "higher_ratio")))
        except Exception:
            pass

    if "forward" not in method_props:
        fwd_in_dim = method_props["encode"]["in_dim"] if "encode" in method_props else 1
        fwd_out_dim = method_props["decode"]["out_dim"] if "decode" in method_props else fwd_in_dim
        method_props["forward"] = {
            "in_dim": fwd_in_dim,
            "in_ratio": 1,
            "out_dim": fwd_out_dim,
            "out_ratio": 1,
            "input_labels": [f"(signal) model input {i}" for i in range(fwd_in_dim)],
            "output_labels": [f"(signal) model output {i}" for i in range(fwd_out_dim)],
        }

    f_props = method_props["forward"]
    meta: Dict[str, Any] = {
        "sr": sr,
        "buffer_size": max(default_buffer_size, higher_ratio),
        "higher_ratio": higher_ratio,
        "in_dim": f_props["in_dim"],
        "in_ratio": f_props["in_ratio"],
        "out_dim": f_props["out_dim"],
        "out_ratio": f_props["out_ratio"],
        "methods": list(method_props.keys()),
        "method_properties": method_props,
        "attribute_properties": {},
    }

    if "encode" in method_props:
        meta["encode_in_dim"] = method_props["encode"]["in_dim"]
        meta["encode_in_ratio"] = method_props["encode"]["in_ratio"]
        meta["encode_out_dim"] = method_props["encode"]["out_dim"]
        meta["encode_out_ratio"] = method_props["encode"]["out_ratio"]

    if "decode" in method_props:
        meta["decode_in_dim"] = method_props["decode"]["in_dim"]
        meta["decode_in_ratio"] = method_props["decode"]["in_ratio"]
        meta["decode_out_dim"] = method_props["decode"]["out_dim"]
        meta["decode_out_ratio"] = method_props["decode"]["out_ratio"]

    return meta


def _build_eager_module_from_onnx(onnx_path: str, sample_input: Optional[torch.Tensor] = None) -> nn.Module:
    import onnx

    onnx_model = onnx.load(onnx_path)
    return OnnxGraphEagerModule(onnx_model, sample_input=sample_input).eval()


class TorchScriptGraphEagerModule(nn.Module):
    """
    Directly translates a `torch.jit.ScriptModule`'s inlined TorchScript IR (`inlined_graph`)
    into an eager `torch.nn.Module` (matching `neural_tilde.migrate`'s direct TS -> ExportedProgram
    approach without going through ONNX).
    - Preserves the exact parameter & buffer hierarchy (`named_parameters()`, `named_buffers()`)
      so `torch.func.functional_call` can promote any parameter by its original TorchScript name.
    - Interprets `prim::Constant`, `prim::GetAttr`, `prim::ListConstruct`, `prim::TupleUnpack`,
      and `aten::*` operations directly via `torch.ops.aten` / `torch.*` so `torch.export.export()`
      captures a native ATen FX graph.
    """

    def __init__(self, ts_mod: torch.jit.ScriptModule, method_name: str = "forward"):
        super().__init__()
        target_fn = getattr(ts_mod, method_name) if method_name != "forward" else ts_mod
        self._ts_graph = target_fn.inlined_graph
        top_input_name = list(self._ts_graph.inputs())[0].debugName()
        used_top_attrs = {
            node.s("name")
            for node in self._ts_graph.nodes()
            if node.kind() == "prim::GetAttr" and list(node.inputs())[0].debugName() == top_input_name
        }
        for mod_name, sub_ts in ts_mod.named_modules():
            sub_eg = self._ensure_submodule(mod_name) if mod_name else self
            sub_eg.__dict__["_ts_ref"] = sub_ts
            self._copy_scalar_attrs(sub_ts, sub_eg)
        for name, param in ts_mod.named_parameters():
            top_prefix = name.split(".")[0]
            keep_as_param = (method_name == "forward") or (not used_top_attrs) or (top_prefix in used_top_attrs)
            self._register_nested(name, param.detach().clone().float().contiguous(), is_param=keep_as_param)
        for name, buf in ts_mod.named_buffers():
            self._register_nested(name, buf.detach().clone(), is_param=False)

    def _ensure_submodule(self, dotted_name: str) -> nn.Module:
        curr = self
        for p in dotted_name.split("."):
            if not hasattr(curr, p):
                curr.add_module(p, nn.Module())
            curr = getattr(curr, p)
        return curr

    def _register_nested(self, dotted_name: str, tensor: torch.Tensor, is_param: bool):
        parts = dotted_name.split(".")
        curr = self
        for p in parts[:-1]:
            if not hasattr(curr, p):
                curr.add_module(p, nn.Module())
            curr = getattr(curr, p)
        if is_param and tensor.is_floating_point():
            curr.register_parameter(parts[-1], nn.Parameter(tensor))
        else:
            curr.register_buffer(parts[-1], tensor)

    @classmethod
    def _copy_scalar_attrs(cls, ts_sub, eager_sub):
        for attr_name in dir(ts_sub):
            if attr_name.startswith("__"):
                continue
            if hasattr(eager_sub, attr_name):
                sub_ts = getattr(ts_sub, attr_name, None)
                sub_eg = getattr(eager_sub, attr_name, None)
                if isinstance(sub_ts, torch.jit.ScriptModule) and isinstance(sub_eg, nn.Module):
                    cls._copy_scalar_attrs(sub_ts, sub_eg)
                continue
            try:
                val = getattr(ts_sub, attr_name)
                if isinstance(val, (int, float, bool, str, list, tuple)):
                    setattr(eager_sub, attr_name, val)
            except Exception:
                pass

    def _exec_block(self, block, env: Dict[str, Any]) -> List[Any]:
        for node in block.nodes():
            kind = node.kind()
            in_vals = [env.get(i.debugName(), None) for i in node.inputs()]
            out_names = [o.debugName() for o in node.outputs()]

            if kind == "prim::Constant":
                val = node.output().toIValue()
                env[out_names[0]] = val
            elif kind == "prim::GetAttr":
                attr_name = node.s("name")
                target_obj = in_vals[0]
                if attr_name == "_flat_weights" and isinstance(target_obj, nn.Module) and "_ts_ref" in target_obj.__dict__:
                    ts_ref = target_obj.__dict__["_ts_ref"]
                    if hasattr(ts_ref, "_flat_weights_names"):
                        env[out_names[0]] = [getattr(target_obj, wn) for wn in ts_ref._flat_weights_names]
                        continue
                if hasattr(target_obj, attr_name):
                    env[out_names[0]] = getattr(target_obj, attr_name)
                elif isinstance(target_obj, nn.Module) and "_ts_ref" in target_obj.__dict__:
                    env[out_names[0]] = getattr(target_obj.__dict__["_ts_ref"], attr_name)
                else:
                    raise AttributeError(f"'{type(target_obj).__name__}' object has no attribute '{attr_name}'")
            elif kind == "prim::SetAttr":
                attr_name = node.s("name")
                setattr(in_vals[0], attr_name, in_vals[1])
            elif kind == "prim::ListConstruct":
                env[out_names[0]] = list(in_vals)
            elif kind == "prim::TupleConstruct":
                env[out_names[0]] = tuple(in_vals)
            elif kind == "prim::TupleIndex":
                env[out_names[0]] = in_vals[0][int(in_vals[1])]
            elif kind == "prim::max":
                env[out_names[0]] = max(in_vals[0]) if len(in_vals) == 1 and isinstance(in_vals[0], (list, tuple)) else max(*in_vals)
            elif kind == "prim::min":
                env[out_names[0]] = min(in_vals[0]) if len(in_vals) == 1 and isinstance(in_vals[0], (list, tuple)) else min(*in_vals)
            elif kind in ("prim::ListUnpack", "prim::TupleUnpack"):
                for o_name, v in zip(out_names, in_vals[0]):
                    env[o_name] = v
            elif kind == "prim::NumToTensor":
                env[out_names[0]] = torch.tensor(in_vals[0])
            elif kind == "prim::dtype":
                env[out_names[0]] = in_vals[0].dtype
            elif kind == "prim::device":
                env[out_names[0]] = in_vals[0].device
            elif kind == "prim::If":
                cond = bool(in_vals[0])
                blocks = list(node.blocks())
                chosen_block = blocks[0] if cond else blocks[1]
                block_outs = self._exec_block(chosen_block, env)
                for o_name, v in zip(out_names, block_outs):
                    env[o_name] = v
            elif kind == "prim::Loop":
                max_trip = int(in_vals[0])
                cond = bool(in_vals[1])
                loop_vars = list(in_vals[2:])
                body_block = list(node.blocks())[0]
                body_inputs = list(body_block.inputs())
                it = 0
                while cond and it < max_trip:
                    env[body_inputs[0].debugName()] = it
                    for bi, lv in zip(body_inputs[1:], loop_vars):
                        env[bi.debugName()] = lv
                    body_outs = self._exec_block(body_block, env)
                    cond = bool(body_outs[0])
                    loop_vars = list(body_outs[1:])
                    it += 1
                for o_name, lv in zip(out_names, loop_vars):
                    env[o_name] = lv
            elif kind == "prim::RaiseException":
                raise RuntimeError(str(in_vals[0]) if in_vals else "TorchScript RaiseException")
            elif kind == "prim::unchecked_cast":
                env[out_names[0]] = in_vals[0]
            elif kind.startswith("aten::"):
                op_name = kind.split("::")[1]
                out = self._exec_aten(op_name, in_vals)
                if len(out_names) == 1:
                    env[out_names[0]] = out
                else:
                    for o_name, v in zip(out_names, out):
                        env[o_name] = v
            else:
                raise NotImplementedError(f"Unsupported TorchScript node kind: {kind}")

        return [env.get(o.debugName(), None) for o in block.outputs()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        env: Dict[str, Any] = {}
        graph_inputs = list(self._ts_graph.inputs())
        env[graph_inputs[0].debugName()] = self
        if len(graph_inputs) > 1:
            env[graph_inputs[1].debugName()] = x

        outs = self._exec_block(self._ts_graph, env)
        return outs[0]

    @staticmethod
    def _exec_aten(op_name: str, args: List[Any]) -> Any:
        if op_name == "Int":
            return int(args[0])
        if op_name == "Float":
            return float(args[0])
        if op_name == "Bool":
            return bool(args[0])
        if op_name == "str":
            return str(args[0])
        if op_name == "list":
            return list(args[0])
        if op_name == "__contains__":
            return args[1] in args[0]
        if op_name == "__getitem__":
            return args[0][args[1]]
        if op_name == "len":
            return len(args[0])
        if op_name == "__not__":
            return not bool(args[0])
        if op_name == "__is__":
            return args[0] is args[1]
        if op_name == "__isnot__":
            return args[0] is not args[1]
        if op_name == "eq" and not isinstance(args[0], torch.Tensor):
            return args[0] == args[1]
        if op_name == "ne" and not isinstance(args[0], torch.Tensor):
            return args[0] != args[1]
        if op_name == "lt" and not isinstance(args[0], torch.Tensor):
            return args[0] < args[1]
        if op_name == "le" and not isinstance(args[0], torch.Tensor):
            return args[0] <= args[1]
        if op_name == "gt" and not isinstance(args[0], torch.Tensor):
            return args[0] > args[1]
        if op_name == "ge" and not isinstance(args[0], torch.Tensor):
            return args[0] >= args[1]
        if op_name == "add":
            if len(args) >= 3 and args[2] != 1:
                return args[0] + args[1] * args[2]
            return args[0] + args[1]
        if op_name == "sub":
            if len(args) >= 3 and args[2] != 1:
                return args[0] - args[1] * args[2]
            return args[0] - args[1]
        if op_name == "mul":
            return args[0] * args[1]
        if op_name == "div":
            return args[0] / args[1]
        if op_name == "elu":
            alpha = float(args[1]) if len(args) > 1 and args[1] is not None else 1.0
            return F.elu(args[0], alpha=alpha)
        if op_name == "ceil" and not isinstance(args[0], torch.Tensor):
            return math.ceil(args[0])
        if op_name == "floordiv" and not isinstance(args[0], torch.Tensor):
            return args[0] // args[1]
        if op_name == "remainder" and not isinstance(args[0], torch.Tensor):
            return args[0] % args[1]
        if op_name == "append":
            args[0].append(args[1])
            return args[0]
        if op_name == "dim":
            return args[0].ndim
        if op_name == "size":
            return int(args[0].shape[args[1]]) if len(args) > 1 and args[1] is not None else [int(s) for s in args[0].shape]
        if op_name == "warn":
            return None
        if op_name == "format":
            return str(args[0]).format(*args[1:])
        if op_name == "zeros":
            size = args[0]
            dtype = args[1] if len(args) > 1 and isinstance(args[1], torch.dtype) else torch.float32
            return torch.zeros(size, dtype=dtype)
        if op_name == "lstm":
            return TorchScriptGraphEagerModule._run_aten_lstm(*args)
        if op_name == "to":
            # Handle aten::to.dtype / aten::to.device overloaded signatures
            tensor = args[0]
            if len(args) >= 2 and isinstance(args[1], torch.dtype):
                return tensor.to(dtype=args[1])
            if len(args) >= 3 and isinstance(args[2], torch.dtype):
                return tensor.to(dtype=args[2])
            return tensor
        if op_name == "pad":
            return F.pad(args[0], args[1], mode=args[2] if len(args) > 2 and args[2] is not None else "constant", value=args[3] if len(args) > 3 and args[3] is not None else 0.0)
        if op_name in ("_convolution", "convolution", "conv1d", "conv2d", "conv_transpose1d"):
            bias = args[2] if len(args) > 2 else None
            args_no_bias = list(args)
            if len(args_no_bias) > 2:
                args_no_bias[2] = None
            if op_name == "_convolution":
                res = torch.ops.aten.convolution(*args_no_bias[:9])
            else:
                aten_packet = getattr(torch.ops.aten, op_name)
                res = aten_packet(*args_no_bias)
            if bias is not None:
                view_shape = [1, -1] + [1] * (res.ndim - 2)
                res = res + bias.view(*view_shape)
            return res
        if hasattr(torch.ops.aten, op_name):
            aten_packet = getattr(torch.ops.aten, op_name)
            return aten_packet(*args)
        if hasattr(torch, op_name):
            return getattr(torch, op_name)(*args)
        raise NotImplementedError(f"Unsupported aten op: aten::{op_name}")

    @staticmethod
    def _run_aten_lstm(
        inp: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor],
        flat_weights: List[torch.Tensor],
        has_biases: bool,
        num_layers: int,
        dropout: float,
        train: bool,
        bidirectional: bool,
        batch_first: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bidirectional:
            return torch.ops.aten.lstm.input(inp, hx, flat_weights, has_biases, num_layers, dropout, train, bidirectional, batch_first)
        x = inp.transpose(0, 1) if batch_first else inp
        seq_len = int(x.shape[0])
        h_0, c_0 = hx[0], hx[1]
        weights_per_layer = 4 if has_biases else 2
        h_n_list = []
        c_n_list = []
        layer_input = x
        for layer in range(num_layers):
            base = layer * weights_per_layer
            w_ih = flat_weights[base]
            w_hh = flat_weights[base + 1]
            b_ih = flat_weights[base + 2] if has_biases else None
            b_hh = flat_weights[base + 3] if has_biases else None
            h_t = h_0[layer]
            c_t = c_0[layer]
            ih_all = torch.matmul(layer_input, w_ih.t())
            if b_ih is not None:
                ih_all = ih_all + b_ih
            outputs = []
            for t in range(seq_len):
                gates = ih_all[t] + torch.matmul(h_t, w_hh.t())
                if b_hh is not None:
                    gates = gates + b_hh
                i_g, f_g, g_g, o_g = gates.chunk(4, dim=-1)
                i_g = torch.sigmoid(i_g)
                f_g = torch.sigmoid(f_g)
                g_g = torch.tanh(g_g)
                o_g = torch.sigmoid(o_g)
                c_t = f_g * c_t + i_g * g_g
                h_t = o_g * torch.tanh(c_t)
                outputs.append(h_t)
            layer_input = torch.stack(outputs, dim=0)
            h_n_list.append(h_t)
            c_n_list.append(c_t)
        out = layer_input.transpose(0, 1) if batch_first else layer_input
        return out, torch.stack(h_n_list, dim=0), torch.stack(c_n_list, dim=0)


def migrate_ts_to_pte(
    ts_model_path: str,
    output_pte_path: Optional[str] = None,
    delegate: str = "coreml",
    bend_mode: str = "delegate",
    buffer_size: Optional[int] = None,
    channels: Optional[int] = None,
    layer_filter: Optional[List[str]] = None,
    eager_module: Optional[nn.Module] = None,
    verbose: bool = True,
) -> str:
    """
    Converts a TorchScript `.ts` or `.onnx` model into a bendable ExecuTorch `.pte` model
    + `.bin` NBND sidecar + `.json` metadata.
    """
    if not os.path.exists(ts_model_path):
        raise FileNotFoundError(f"Model not found: {ts_model_path}")

    if not output_pte_path:
        base, _ = os.path.splitext(ts_model_path)
        output_pte_path = f"{base}.pte"

    companion_modules: Dict[str, tuple[nn.Module, torch.Tensor]] = {}

    ext = os.path.splitext(ts_model_path)[1].lower()
    if ext == ".onnx":
        import onnx

        if verbose:
            print(f"Loading ONNX model for .pte conversion: {ts_model_path}")
        onnx_mod = onnx.load(ts_model_path)
        meta_props = {p.key: p.value for p in onnx_mod.metadata_props}
        sr = int(meta_props.get("netbender.sample_rate", "44100"))
        ratio = int(meta_props.get("netbender.ratio", "2048"))
        latent_dim = int(meta_props.get("netbender.latent_dim", "16"))
        in_dim = channels or int(meta_props.get("netbender.in_channels", "1"))
        out_dim = int(meta_props.get("netbender.out_channels", str(in_dim)))
        history_samples = int(meta_props.get("netbender.history_samples", "0"))
        output_offset = int(meta_props.get("netbender.output_offset", str(history_samples)))
        is_rave_onnx = any(n.name.startswith("/_rave/") for n in onnx_mod.graph.node) or "rave" in ts_model_path.lower() or "darbouka" in ts_model_path.lower() or "speech" in ts_model_path.lower()
        if is_rave_onnx and history_samples == 0 and ratio >= 1024:
            history_samples = 28672 if latent_dim >= 8 else 4096
            output_offset = history_samples
        seq_len = buffer_size or max(2048, ratio)
        decode_latent_steps = max(1, (history_samples + seq_len + ratio - 1) // max(1, ratio))
        meta = {
            "sr": sr,
            "sample_rate": sr,
            "buffer_size": seq_len,
            "higher_ratio": ratio,
            "ratio": ratio,
            "latent_dim": latent_dim,
            "in_dim": in_dim,
            "in_channels": in_dim,
            "in_ratio": 1,
            "out_dim": out_dim,
            "out_channels": out_dim,
            "out_ratio": 1,
            "history_samples": history_samples,
            "output_offset": output_offset,
            "decode_latent_steps": decode_latent_steps,
            "methods": ["forward"],
        }
        sample_input = torch.zeros(1, in_dim, history_samples + seq_len, dtype=torch.float32)
        target_module = eager_module or _build_eager_module_from_onnx(ts_model_path, sample_input=sample_input)

        base_onnx, _ = os.path.splitext(ts_model_path)
        enc_onnx_path = f"{base_onnx}_encode.onnx"
        dec_onnx_path = f"{base_onnx}_decode.onnx"
        if os.path.exists(enc_onnx_path) and os.path.exists(dec_onnx_path):
            try:
                enc_mod = _build_eager_module_from_onnx(enc_onnx_path, sample_input=sample_input)
                with torch.no_grad():
                    sample_latent = enc_mod(sample_input)
                dec_mod = _build_eager_module_from_onnx(dec_onnx_path, sample_input=sample_latent)
                companion_modules["encode"] = (enc_mod, sample_input)
                companion_modules["decode"] = (dec_mod, sample_latent)
                meta["decode_latent_steps"] = int(sample_latent.shape[2])
            except Exception as e:
                if verbose:
                    print(f"  [Note] Companion ONNX encode/decode skip: {e}")
    else:
        if verbose:
            print(f"Loading TorchScript model: {ts_model_path}")
        ts_mod = torch.jit.load(ts_model_path, map_location="cpu").eval()
        meta = extract_nn_tilde_metadata(ts_mod, default_buffer_size=buffer_size or 2048)

        if buffer_size is not None:
            meta["buffer_size"] = buffer_size
        if channels is not None:
            meta["in_dim"] = channels

        in_dim = int(meta["in_dim"])
        seq_len = int(meta["buffer_size"])
        latent_dim = int(meta.get("encode_out_dim", meta.get("decode_in_dim", 16)))
        is_rave = hasattr(ts_mod, "_rave") or "rave" in ts_model_path.lower() or "darbouka" in ts_model_path.lower() or "speech" in ts_model_path.lower()
        history_samples = (28672 if latent_dim >= 8 else 4096) if (is_rave and int(meta["higher_ratio"]) >= 1024) else 0
        ratio = int(meta["higher_ratio"])
        meta["sample_rate"] = meta["sr"]
        meta["ratio"] = ratio
        meta["latent_dim"] = latent_dim
        meta["in_channels"] = in_dim
        meta["out_channels"] = int(meta["out_dim"])
        meta["history_samples"] = history_samples
        meta["output_offset"] = history_samples
        meta["decode_latent_steps"] = max(1, (history_samples + seq_len + ratio - 1) // max(1, ratio))
        sample_input = torch.zeros(1, in_dim, history_samples + seq_len, dtype=torch.float32)
        target_module = eager_module
        has_enc_dec = hasattr(ts_mod, "encode") and hasattr(ts_mod, "decode")

        if target_module is None:
            # 1. Try direct TorchScript IR -> Eager ATen Module translation first (like neural_tilde.migrate)
            try:
                candidate_mod = TorchScriptGraphEagerModule(ts_mod, method_name="forward").eval()
                with torch.no_grad():
                    _ = candidate_mod(sample_input)
                target_module = candidate_mod
                if verbose:
                    print("  -> Directly translated TorchScript IR (inlined_graph) to eager PyTorch module.")
                if has_enc_dec:
                    try:
                        enc_mod = TorchScriptGraphEagerModule(ts_mod, method_name="encode").eval()
                        with torch.no_grad():
                            sample_latent = enc_mod(sample_input)
                        dec_mod = TorchScriptGraphEagerModule(ts_mod, method_name="decode").eval()
                        with torch.no_grad():
                            _ = dec_mod(sample_latent)
                        companion_modules["encode"] = (enc_mod, sample_input)
                        companion_modules["decode"] = (dec_mod, sample_latent)
                        meta["decode_latent_steps"] = int(sample_latent.shape[2])
                    except Exception as enc_dec_err:
                        if verbose:
                            print(f"  [Note] Direct encode/decode translation skipped: {enc_dec_err}")
            except Exception as direct_err:
                if verbose:
                    print(
                        f"  [Note] Direct TorchScript IR trace encountered non-exportable op ({direct_err}); "
                        f"falling back to sanitized graph lowering..."
                    )
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_onnx = os.path.join(tmpdir, "model_sanitized.onnx")
                    comp_methods = ["encode", "decode"] if has_enc_dec else []
                    if is_rave:
                        from python_tools.rave_ts_to_onnx import convert_rave_ts_to_onnx

                        convert_rave_ts_to_onnx(
                            ts_model_path=ts_model_path,
                            output_onnx_path=tmp_onnx,
                            buffer_size=history_samples + seq_len,
                            channels=in_dim,
                            promote_for_bending=False,
                            methods=comp_methods,
                            verbose=verbose,
                        )
                    else:
                        from python_tools.stable_audio_ts_to_onnx import convert_stable_audio_ts_to_onnx

                        convert_stable_audio_ts_to_onnx(
                            ts_model_path=ts_model_path,
                            output_onnx_path=tmp_onnx,
                            buffer_size=history_samples + seq_len,
                            channels=in_dim,
                            promote_for_bending=False,
                            methods=comp_methods,
                            verbose=verbose,
                        )
                    target_module = _build_eager_module_from_onnx(tmp_onnx, sample_input=sample_input)
                    tmp_enc = os.path.join(tmpdir, "model_sanitized_encode.onnx")
                    tmp_dec = os.path.join(tmpdir, "model_sanitized_decode.onnx")
                    if os.path.exists(tmp_enc) and os.path.exists(tmp_dec):
                        try:
                            enc_mod = _build_eager_module_from_onnx(tmp_enc, sample_input=sample_input)
                            with torch.no_grad():
                                sample_latent = enc_mod(sample_input)
                            dec_mod = _build_eager_module_from_onnx(tmp_dec, sample_input=sample_latent)
                            companion_modules["encode"] = (enc_mod, sample_input)
                            companion_modules["decode"] = (dec_mod, sample_latent)
                            meta["decode_latent_steps"] = int(sample_latent.shape[2])
                        except Exception as enc_dec_err:
                            if verbose:
                                print(f"  [Note] Sanitized encode/decode lowering skipped: {enc_dec_err}")

    export_bendable_pte(
        module=target_module,
        sample_audio_input=sample_input,
        output_pte_path=output_pte_path,
        delegate=delegate,
        bend_mode=bend_mode,
        layer_filter=layer_filter,
        method_name="forward",
        companion_modules=companion_modules,
        extra_metadata=meta,
        write_sidecar=True,
        verbose=verbose,
    )
    return output_pte_path


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a TorchScript (.ts) or ONNX (.onnx) audio model to ExecuTorch (.pte) with NetBender sidecars."
    )
    parser.add_argument("input", help="Path to input .ts or .onnx file")
    parser.add_argument("-o", "--output", default=None, help="Output .pte path (default: <input>.pte)")
    parser.add_argument(
        "--delegate",
        choices=["portable", "xnnpack", "coreml", "mps"],
        default="coreml",
        help="ExecuTorch hardware delegate (default: coreml)",
    )
    parser.add_argument(
        "--bend-mode",
        choices=["delegate", "full", "none"],
        default="delegate",
        help="Parameter promotion strategy: delegate (fast CoreML/XNNPACK + safe bending), full (all weights), or none",
    )
    parser.add_argument("--buffer-size", type=int, default=None, help="Override audio buffer size")
    parser.add_argument("--channels", type=int, default=None, help="Override input channel count")
    parser.add_argument("--filter", nargs="+", default=None, help="Optional layer name substrings to promote")

    args = parser.parse_args()
    migrate_ts_to_pte(
        ts_model_path=args.input,
        output_pte_path=args.output,
        delegate=args.delegate,
        bend_mode=args.bend_mode,
        buffer_size=args.buffer_size,
        channels=args.channels,
        layer_filter=args.filter,
        verbose=True,
    )


if __name__ == "__main__":
    main()
