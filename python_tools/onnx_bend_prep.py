#!/usr/bin/env python3
"""
onnx_bend_prep.py - NetBender ONNX Parameter Promotion Tool

Converts frozen ONNX initializers (weights, biases, scale/shift vectors) into
explicit graph inputs ("Exposed Ports"). This allows real-time, zero-copy
circuit bending (matrix drawing, stochastic drift, cross-talk bridging)
without recreating the ONNX Runtime session during live audio callbacks.
"""

import argparse
import sys
import os
from typing import List, Optional

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
except ImportError:
    print("Error: 'onnx' Python package is required. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)


def is_bendable_tensor(name: str, tensor: TensorProto, min_elements: int = 1) -> bool:
    """
    Check if an initializer qualifies as a bendable neural network parameter.
    Includes weights, biases, scale/shift vectors, normalization coefficients.
    Excludes shape constants, sequence indices, or integer scalars.
    """
    # Only float / double / float16 parameters are bendable audio weights
    if tensor.data_type not in (
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.DOUBLE,
        TensorProto.BFLOAT16,
    ):
        return False

    dims = list(tensor.dims)
    total_elements = 1
    for d in dims:
        total_elements *= d

    if total_elements < min_elements:
        return False

    return True


def promote_initializers_to_inputs(
    input_path: str,
    output_path: str,
    layer_filter: Optional[List[str]] = None,
    keep_initializers_as_defaults: bool = True,
    verbose: bool = True,
) -> int:
    """
    Reads an ONNX model, finds specified or all eligible float initializers,
    and converts them to explicit graph inputs.
    """
    if verbose:
        print(f"Loading ONNX model from: {input_path}")

    model = onnx.load(input_path)
    graph = model.graph

    existing_input_names = {inp.name for inp in graph.input}
    initializers = list(graph.initializer)

    promoted_count = 0
    new_inputs = []

    for init in initializers:
        name = init.name

        # If already an input, skip
        if name in existing_input_names:
            continue

        # Optional layer name filtering
        if layer_filter:
            matched = any(f.lower() in name.lower() for f in layer_filter)
            if not matched:
                continue

        # Verify candidate tensor type and dimensions
        if not is_bendable_tensor(name, init):
            continue

        # Create corresponding ValueInfoProto for graph.input
        dims = list(init.dims)
        value_info = helper.make_tensor_value_info(
            name=name,
            elem_type=init.data_type,
            shape=dims,
        )
        new_inputs.append(value_info)
        promoted_count += 1

        if verbose:
            print(f"  [+] Promoted: {name} | Shape: {dims} | Type: {init.data_type}")

    # Append new inputs to graph.input
    graph.input.extend(new_inputs)

    # In ONNX IR >= 4, having an initializer with a matching graph.input name marks it as an
    # input with a default initializer value. If keep_initializers_as_defaults is False, we strip it.
    if not keep_initializers_as_defaults:
        name_set = {inp.name for inp in new_inputs}
        remaining_inits = [init for init in graph.initializer if init.name not in name_set]
        del graph.initializer[:]
        graph.initializer.extend(remaining_inits)

    # Add metadata marker to document that this model is NetBender-promoted
    meta = model.metadata_props.add()
    meta.key = "netbender.bendable"
    meta.value = "1"

    meta_count = model.metadata_props.add()
    meta_count.key = "netbender.promoted_layers_count"
    meta_count.value = str(promoted_count)

    # Verify model validity
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"Warning: ONNX model check raised an issue: {e}", file=sys.stderr)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)

    if verbose:
        print(f"\nSuccessfully promoted {promoted_count} parameters to graph inputs.")
        print(f"Saved bendable model to: {output_path}")

    return promoted_count


def inspect_onnx_model(input_path: str):
    """Prints all inputs, outputs, and initializers of an ONNX model."""
    print(f"\n--- Inspecting ONNX Model: {input_path} ---")
    model = onnx.load(input_path)
    graph = model.graph

    print(f"Graph Name: {graph.name}")
    print("\n[Graph Inputs]:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  - {inp.name}: shape={shape}, type={inp.type.tensor_type.elem_type}")

    print("\n[Graph Outputs]:")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  - {out.name}: shape={shape}, type={out.type.tensor_type.elem_type}")

    print(f"\n[Graph Initializers / Weights]: {len(graph.initializer)} total")
    for i, init in enumerate(graph.initializer[:15]):
        print(f"  - {init.name}: shape={list(init.dims)}")
    if len(graph.initializer) > 15:
        print(f"  ... and {len(graph.initializer) - 15} more.")


def main():
    parser = argparse.ArgumentParser(
        description="NetBender ONNX Parameter Promotion Tool - Prepares ONNX models for real-time audio circuit-bending."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input .onnx file")
    parser.add_argument("-o", "--output", help="Path to output .onnx file (defaults to <name>_bendable.onnx)")
    parser.add_argument("--inspect", action="store_true", help="Inspect model structure without writing changes")
    parser.add_argument(
        "--filter",
        nargs="+",
        help="Optional name substrings to selectively promote (e.g. --filter conv weight bias)",
    )
    parser.add_argument(
        "--strip-initializers",
        action="store_true",
        help="Completely remove promoted weights from initializers (pure dynamic inputs without defaults)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.inspect:
        inspect_onnx_model(args.input)
        return

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_bendable{ext}"

    promote_initializers_to_inputs(
        input_path=args.input,
        output_path=output_path,
        layer_filter=args.filter,
        keep_initializers_as_defaults=not args.strip_initializers,
        verbose=True,
    )


if __name__ == "__main__":
    main()
