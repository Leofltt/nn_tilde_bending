"""
Exporter for Stable Audio 1.0 / Stable Audio Open VAE (Oobleck / DAC architecture).

Wraps the continuous latent convolutional variational autoencoder into the nn~
streaming format for real-time circuit bending in nn_tilde_bending.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Union, Tuple

# Add project root to sys.path so direct script execution works
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from python_tools.exporters.base_exporter import AudioCodecWrapper, export_torchscript
from python_tools.stable_audio_ts_to_onnx import convert_stable_audio_ts_to_onnx


class OobleckEncoderAdapter(nn.Module):
    """
    Adapts AutoencoderOobleck / DAC encoder to pure tensor I/O:
    Input: [B, C, T] -> Output: [B, latent_dim, T // ratio]
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class OobleckDecoderAdapter(nn.Module):
    """
    Adapts AutoencoderOobleck / DAC decoder to pure tensor I/O:
    Input: [B, latent_dim, T // ratio] -> Output: [B, C, T]
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class StableAudioVAEWrapper(AudioCodecWrapper):
    """
    nn~ wrapper for Stable Audio Open / 1.0 VAE (Stereo 44.1 kHz, 64-dim latents, 2048x ratio).
    """
    def __init__(
        self,
        autoencoder_or_vae: nn.Module,
        sr: int = 44100,
        latent_dim: int = 64,
        ratio: int = 2048,
    ):
        encoder = None
        decoder = None

        # 1. Check for diffusers AutoencoderOobleck
        if hasattr(autoencoder_or_vae, "encoder") and hasattr(autoencoder_or_vae, "decoder"):
            encoder = OobleckEncoderAdapter(autoencoder_or_vae.encoder)
            decoder = OobleckDecoderAdapter(autoencoder_or_vae.decoder)
        elif hasattr(autoencoder_or_vae, "model"):
            target = autoencoder_or_vae.model
            if hasattr(target, "encoder") and hasattr(target, "decoder"):
                encoder = OobleckEncoderAdapter(target.encoder)
                decoder = OobleckDecoderAdapter(target.decoder)

        if encoder is None or decoder is None:
            raise ValueError(
                "Could not extract encoder/decoder from provided autoencoder model."
            )

        # Enable cached_conv streaming if available (eliminates boundary clicking artifacts)
        try:
            import cached_conv as cc
            logging.info("Enabling cached_conv for seamless streaming...")
            cc.use_cached_conv(True)
        except Exception as cc_err:
            logging.info(f"Proceeding without cached_conv: {cc_err}")

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            sr=sr,
            in_channels=2,
            out_channels=2,
            latent_dim=latent_dim,
            ratio=ratio,
            test_method=False,
        )


def load_stable_audio_1_vae(
    pretrained_name_or_path: str = "stabilityai/stable-audio-open-1.0",
    download_dir: str = "models/checkpoints/stable_audio_1_vae",
) -> nn.Module:
    """
    Loads the Stable Audio 1.0 / Open VAE (AutoencoderOobleck) via diffusers
    or local directory.
    """
    try:
        from diffusers import AutoencoderOobleck
        logging.info(f"Loading AutoencoderOobleck from '{pretrained_name_or_path}' (subfolder='vae')...")
        os.makedirs(download_dir, exist_ok=True)
        vae = AutoencoderOobleck.from_pretrained(
            pretrained_name_or_path,
            subfolder="vae",
            cache_dir=download_dir,
        )
        vae.eval()
        return vae
    except ImportError:
        raise ImportError(
            "diffusers is required to export Stable Audio 1.0 / Open VAE.\n"
            "Install it via: uv pip install diffusers"
        )


def export_stable_audio(
    model_checkpoint_or_instance=None,
    output_path: Optional[str] = None,
    sr: int = 44100,
    latent_dim: int = 64,
    ratio: int = 2048,
    repo_id: str = "stabilityai/stable-audio-open-1.0",
    download_dir: str = "models/checkpoints/stable_audio_1_vae",
    to_onnx: bool = False,
    format: Optional[str] = None,
    promote_for_bending: bool = True,
    methods: Optional[List[str]] = None,
    buffer_size: int = 8192,
    channels: int = 2,
    force: bool = False,
    filename: Optional[str] = None,
    verbose: bool = True,
) -> Union[str, Tuple[str, str]]:
    """
    Exports Stable Audio 1.0 / Open VAE to TorchScript (.ts), ONNX (.onnx), or both.
    If an existing TorchScript model already exists or is provided, converts it directly
    to ONNX without re-downloading or re-loading the heavy checkpoint.
    """
    # 1. Determine requested format
    target_format = "ts"
    if format:
        target_format = format.lower()
    elif to_onnx or (output_path and output_path.endswith(".onnx")):
        target_format = "onnx"

    # 2. Determine target filepaths
    default_base = "models/stable_audio_1_vae"
    if output_path is None:
        if target_format == "onnx":
            output_path = f"{default_base}_bendable.onnx" if promote_for_bending else f"{default_base}.onnx"
        else:
            output_path = f"{default_base}.ts"

    if output_path.endswith(".onnx"):
        onnx_path = output_path
        base = output_path.removesuffix(".onnx").removesuffix("_bendable").removesuffix("_raw")
        ts_path = f"{base}.ts"
    elif output_path.endswith(".ts"):
        ts_path = output_path
        base = output_path.removesuffix(".ts")
        onnx_path = f"{base}_bendable.onnx" if promote_for_bending else f"{base}.onnx"
    else:
        ts_path = f"{output_path}.ts"
        onnx_path = f"{output_path}_bendable.onnx" if promote_for_bending else f"{output_path}.onnx"

    # 3. Check if input checkpoint is an existing TorchScript model file
    if isinstance(model_checkpoint_or_instance, str) and os.path.isfile(model_checkpoint_or_instance):
        is_ts_file = model_checkpoint_or_instance.endswith((".ts", ".pt"))
        if is_ts_file and target_format in ("onnx", "both"):
            if verbose:
                logging.info(f"Input '{model_checkpoint_or_instance}' is an existing TorchScript model.")
                logging.info(f"Converting directly to ONNX: {onnx_path}")
            res_onnx = convert_stable_audio_ts_to_onnx(
                ts_model_path=model_checkpoint_or_instance,
                output_onnx_path=onnx_path,
                buffer_size=buffer_size,
                channels=channels,
                promote_for_bending=promote_for_bending,
                methods=methods,
                verbose=verbose,
            )
            if target_format == "both":
                return (model_checkpoint_or_instance, res_onnx)
            return res_onnx
        elif is_ts_file and target_format == "ts":
            if verbose:
                logging.info(f"Using existing TorchScript model at '{model_checkpoint_or_instance}'")
            return model_checkpoint_or_instance

    # 4. If ONNX is requested and a .ts model already exists on disk, convert directly!
    if target_format in ("onnx", "both") and os.path.exists(ts_path) and not force and model_checkpoint_or_instance is None:
        if verbose:
            logging.info(f"Found existing TorchScript model at '{ts_path}'.")
            logging.info(f"Converting directly to ONNX without re-downloading: {onnx_path}")
        res_onnx = convert_stable_audio_ts_to_onnx(
            ts_model_path=ts_path,
            output_onnx_path=onnx_path,
            buffer_size=buffer_size,
            channels=channels,
            promote_for_bending=promote_for_bending,
            methods=methods,
            verbose=verbose,
        )
        if target_format == "both":
            return (ts_path, res_onnx)
        return res_onnx

    # 5. Otherwise, export/load the TorchScript model
    if target_format == "ts" and os.path.exists(ts_path) and not force and model_checkpoint_or_instance is None:
        if verbose:
            logging.info(f"TorchScript model already exists at '{ts_path}'. Skipping export (use --force to overwrite).")
        return ts_path

    if model_checkpoint_or_instance is None or isinstance(model_checkpoint_or_instance, str):
        path = model_checkpoint_or_instance or repo_id
        vae = load_stable_audio_1_vae(path, download_dir=download_dir)
    else:
        vae = model_checkpoint_or_instance

    wrapper = StableAudioVAEWrapper(
        autoencoder_or_vae=vae,
        sr=sr,
        latent_dim=latent_dim,
        ratio=ratio,
    )

    if verbose:
        logging.info(f"Exporting Stable Audio wrapper to TorchScript: {ts_path}...")
    res_ts = export_torchscript(wrapper, ts_path, test_buffer_size=buffer_size)

    # 6. If ONNX was also requested, convert the freshly exported TorchScript
    if target_format in ("onnx", "both"):
        if verbose:
            logging.info(f"Converting freshly exported TorchScript to ONNX: {onnx_path}...")
        res_onnx = convert_stable_audio_ts_to_onnx(
            ts_model_path=ts_path,
            output_onnx_path=onnx_path,
            buffer_size=buffer_size,
            channels=channels,
            promote_for_bending=promote_for_bending,
            methods=methods,
            verbose=verbose,
        )
        if target_format == "both":
            return (res_ts, res_onnx)
        return res_onnx

    return res_ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Stable Audio 1.0 / Open VAE for nn_tilde_bending")
    parser.add_argument("-i", "--input", "--checkpoint", dest="checkpoint", type=str, default=None,
                        help="Path to checkpoint/model (.ts, .pt) or HF repo id (default: stabilityai/stable-audio-open-1.0)")
    parser.add_argument("--repo-id", type=str, default="stabilityai/stable-audio-open-1.0", help="Hugging Face repo id")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path (.ts or .onnx)")
    parser.add_argument("--format", type=str, choices=["ts", "onnx", "both"], default=None,
                        help="Export format: ts, onnx, or both (default: auto based on output extension or --onnx)")
    parser.add_argument("--onnx", action="store_true", help="Export as ONNX instead of TorchScript")
    parser.add_argument("--no-bending", action="store_true", help="Skip promoting initializers for ONNX bending")
    parser.add_argument("--methods", nargs="+", default=["all"],
                        help="Companion methods for ONNX: all, forward, encode, decode (default: all)")
    parser.add_argument("--buffer-size", type=int, default=8192, help="Buffer size (default: 8192)")
    parser.add_argument("--channels", type=int, default=2, help="Channels (default: 2)")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate (default: 44100)")
    parser.add_argument("--ratio", type=int, default=2048, help="Temporal compression ratio (default: 2048)")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimensionality (default: 64)")
    parser.add_argument("--force", action="store_true", help="Force re-export even if model already exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    export_stable_audio(
        model_checkpoint_or_instance=args.checkpoint,
        output_path=args.output,
        sr=args.sr,
        latent_dim=args.latent_dim,
        ratio=args.ratio,
        repo_id=args.repo_id,
        to_onnx=args.onnx,
        format=args.format,
        promote_for_bending=not args.no_bending,
        methods=args.methods,
        buffer_size=args.buffer_size,
        channels=args.channels,
        force=args.force,
    )
