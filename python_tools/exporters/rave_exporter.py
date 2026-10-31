"""
rave_exporter.py - Exporter for RAVE models from training checkpoints (.ckpt).

Wraps RAVE checkpoint loading and provides export to:
  1. Streaming TorchScript (.ts) for nn~ / NetBender TorchBackend (using RAVE's native streaming mode).
  2. Bendable ONNX (.onnx) for NetBender OnnxBackend (with weight promotion for circuit bending).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch


def load_rave_model(checkpoint_path: str):
    """
    Loads a RAVE model from a PyTorch Lightning checkpoint (.ckpt) or run directory.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        import rave
    except ImportError as e:
        raise ImportError(
            f"Could not import 'rave'. Ensure 'acids-rave' is installed: {e}"
        )

    # Automatically locate and parse config.gin if present in the checkpoint directory
    import gin
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    gin_path = os.path.join(ckpt_dir, "config.gin")
    if os.path.exists(gin_path):
        logging.info(f"Parsing Gin configuration: {gin_path}...")
        gin.parse_config_file(gin_path, skip_unknown=True)
    else:
        logging.warning(f"No config.gin found in {ckpt_dir}. If loading fails, ensure config.gin is in the same directory.")

    logging.info(f"Instantiating RAVE and loading weights from {checkpoint_path}...")
    model = rave.RAVE()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "callbacks" in checkpoint and "EMA" in checkpoint["callbacks"]:
        logging.info("Loading EMA weights from checkpoint...")
        model.load_state_dict(checkpoint["callbacks"]["EMA"], strict=False)
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

    model.eval()
    return model


def export_rave_to_torchscript(
    model,
    output_path: str,
    streaming: bool = True,
):
    """
    Exports a RAVE model to TorchScript (.ts) using RAVE's export pipeline.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logging.info(f"Exporting RAVE to TorchScript (streaming={streaming}) -> {output_path}...")

    import cached_conv as cc
    import rave.blocks as blocks
    from scripts import export as rave_export

    cc.use_cached_conv(streaming)

    # Remove weight norm prior to scripting
    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)

    if isinstance(model.encoder, blocks.VariationalEncoder):
        script_class = rave_export.VariationalScriptedRAVE
    elif isinstance(model.encoder, blocks.DiscreteEncoder):
        script_class = rave_export.DiscreteScriptedRAVE
    elif isinstance(model.encoder, blocks.WasserteinEncoder):
        script_class = rave_export.WasserteinScriptedRAVE
    elif isinstance(model.encoder, blocks.SphericalEncoder):
        script_class = rave_export.SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(model.encoder)} not supported for export.")

    scripted_rave = script_class(
        pretrained=model,
        channels=getattr(model, "n_channels", 1),
        fidelity=0.95,
        target_sr=getattr(model, "sampling_rate", 48000),
        prior=None,
    )

    logging.info(f"Exporting scripted RAVE to: {output_path}...")
    scripted_rave.export_to_ts(output_path)
    logging.info(f"Successfully saved TorchScript model to: {output_path}")
    return output_path


def export_rave_to_onnx(
    model,
    output_path: str,
    buffer_size: int = 2048,
    channels: int = 1,
    opset_version: int = 17,
    promote_for_bending: bool = True,
):
    """
    Exports a RAVE model forward pass to ONNX and optionally promotes weights to inputs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base, ext = os.path.splitext(output_path)
    raw_onnx = output_path if not promote_for_bending else f"{base}_raw.onnx"

    dummy_input = torch.zeros(1, channels, buffer_size, dtype=torch.float32)

    logging.info(f"Exporting RAVE to ONNX (opset {opset_version}): {raw_onnx}...")

    # We wrap the model forward to ensure standard [batch, channels, time] signature
    class RAVEWrapper(torch.nn.Module):
        def __init__(self, rave_module):
            super().__init__()
            self.rave = rave_module

        def forward(self, x):
            # If RAVE takes (x), return reconstructed audio
            return self.rave(x)

    wrapped = RAVEWrapper(model)
    wrapped.eval()

    torch.onnx.export(
        wrapped,
        (dummy_input,),
        raw_onnx,
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes={
            "audio_in": {0: "batch", 2: "samples"},
            "audio_out": {0: "batch", 2: "samples"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    logging.info(f"Exported raw ONNX: {raw_onnx}")

    if promote_for_bending:
        logging.info(f"Promoting initializers to inputs for circuit bending: {output_path}...")
        from python_tools.onnx_bend_prep import promote_initializers_to_inputs
        promote_initializers_to_inputs(
            input_path=raw_onnx,
            output_path=output_path,
            keep_initializers_as_defaults=True,
            verbose=True,
        )
        if raw_onnx != output_path and os.path.exists(raw_onnx):
            try:
                os.remove(raw_onnx)
            except OSError:
                pass

    logging.info(f"Successfully exported RAVE ONNX model to: {output_path}")
    return output_path


def export_rave(
    checkpoint_path: str,
    output_path: str,
    to_onnx: bool = False,
    streaming: bool = True,
    promote_for_bending: bool = True,
    buffer_size: int = 2048,
    channels: int = 1,
):
    """
    Main export dispatcher for RAVE checkpoints.
    """
    model = load_rave_model(checkpoint_path)

    if to_onnx:
        return export_rave_to_onnx(
            model=model,
            output_path=output_path,
            buffer_size=buffer_size,
            channels=channels,
            promote_for_bending=promote_for_bending,
        )
    else:
        return export_rave_to_torchscript(
            model=model,
            output_path=output_path,
            streaming=streaming,
        )
